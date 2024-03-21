# Copyright (2023) Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
import torch.nn as nn
import torch.distributed as dist

from einops import rearrange

from timm.models.layers import trunc_normal_

from ..dataset.tokenizers import build_tokenizer


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )


allgather = AllGather.apply


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LynxBase(nn.Module):
    def __init__(self, config=None, freeze_vit=True, freeze_llm=True, load_bridge=False):
        super().__init__()

        self.freeze_vit = freeze_vit
        self.freeze_llm = freeze_llm

        self.init_params = []

        self.vision_encoder, missing_keys = self.build_vision_encoder(config, freeze_params=freeze_vit)
        self.update_init_params([f'vision_encoder.{k}' for k in missing_keys])

        self.tokenizer, num_new_tokens = build_tokenizer(config['LLM'], use_left_pad=config['use_left_pad'])
        self.LLM, missing_keys = self.build_LLM(config, freeze_params=freeze_llm, num_new_tokens=num_new_tokens)
        self.update_init_params([f'LLM.{k}' for k in missing_keys])

        # Bridge: Vision2Text
        self.bridge, missing_keys = self.build_bridge(config, load_params=load_bridge)
        self.update_init_params([f'bridge.{k}' for k in missing_keys])

        # Video
        self.video_encoding = config.get("video_encoding", "")
        if self.video_encoding:
            self.add_frame_pos = config['add_frame_pos']
            if self.add_frame_pos:
                self.absolute_frame_pos_embed = nn.Parameter(
                    torch.zeros(1, config['data']['num_frames'], 1, self.vision_width))
                trunc_normal_(self.absolute_frame_pos_embed, std=.02)
                self.update_init_params(['absolute_frame_pos_embed'])

            elif self.video_encoding == 'concate':
                # concate all video frames
                pass

            else:
                raise NotImplementedError(f"video_encoding == {self.video_encoding}")

        if os.path.exists(config.get('checkpoint', '')):
            self.load_pretrained(config['checkpoint'], config)

    def build_vision_encoder(self, config, freeze_params=True):
        """
        Args:
            load_params: False when building fine-tuning models
        """
        print(f"### Building ViT (Freeze: {freeze_params})", flush=True)

        if config['vision_encoder'] == 'eva_vit_1b':
            from .vits.eva_vit import create_eva_vit_g
            model, missing_keys = create_eva_vit_g(config['image_res'], config.get('drop_path_rate', 0.0),
                                                   load_params=True)
            # set attrs
            self.vision_width = model.embed_dim

        else:
            raise NotImplementedError("Vision Encoder: ", config['vision_encoder'])

        if freeze_params:

            assert len(missing_keys) == 0

            for name, param in model.named_parameters():
                param.requires_grad = False

            model = model.eval()
            model.train = disabled_train

        return model, missing_keys

    def build_LLM(self, config, freeze_params=True, num_new_tokens=0):
        print(f"### Building LLM (Freeze: {freeze_params})", flush=True)

        self.use_dec_only = True
        self.use_adapter = config.get('use_adapter', False)

        if 'vicuna-7b' in config['LLM']:
            from .llms.llama.modeling_llama import LlamaForCausalLM, LlamaConfig

            # rpath = os.path.join("data/", config['LLM'])
            rpath = config['LLM']
            assert os.path.exists(rpath)

            text_config = LlamaConfig.from_json_file(os.path.join(rpath, "config.json"))

            text_config.use_flash_attn = config.get("use_flash_attn", False)

            text_config.use_adapter = self.use_adapter
            text_config.adapter_freq = config.get('adapter_freq', -1)
            text_config.freeze_params = freeze_params
            text_config.label_smoothing = config.get("label_smoothing", 0.0)

            model = LlamaForCausalLM.from_pretrained(rpath, config=text_config)

            model.model.padding_idx = self.tokenizer.pad_token_id

            missing_keys = [n for n, _ in model.named_parameters() if 'adapter' in n]

            # set attrs
            self.text_width = model.config.hidden_size
            decoder_layers_attr_name = "model.layers"

        else:
            raise NotImplementedError("LLM: ", config['LLM'])

        if num_new_tokens > 0:
            print("### LLM Vocab Size: ", model.config.vocab_size, flush=True)
            print("### num_new_tokens: ", num_new_tokens, flush=True)
            vocab_size = model.config.vocab_size + num_new_tokens
            assert vocab_size == len(self.tokenizer)

            model.resize_token_embeddings(vocab_size)
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

        if freeze_params:
            print("### Freeze LLM", flush=True)
            for name, param in model.named_parameters():
                if 'adapter' in name:
                    pass
                else:
                    param.requires_grad = False

        return model, missing_keys

    def build_bridge(self, config, load_params=True):
        """
        Bridge for Vision to Text
        """
        print("### Building Bridge", flush=True)
        missing_keys = []

        if config['bridge'] == 'resampler':
            from .bridges.resampler import PerceiverResampler
            model = PerceiverResampler(self.vision_width, self.text_width,
                                       depth=config["bridge_depth"], num_latents=config["num_bridge_tokens"])
            assert load_params is False, "no param to load for Resampler"
            missing_keys = [n for (n, p) in model.named_parameters()]
        else:
            raise NotImplementedError("Bridge: ", config['bridge'])

        if load_params:
            print("missing_keys: ", missing_keys, flush=True)

        return model, missing_keys

    def update_init_params(self, missing_keys=None):
        if missing_keys is not None:
            assert isinstance(missing_keys, list)
            for k in missing_keys:
                if k not in self.init_params:
                    self.init_params.append(k)

        named_parameters = set([n for n, _ in self.named_parameters()])
        for n in set(self.init_params):
            if n not in named_parameters:
                self.init_params.remove(n)

    def load_pretrained(self, ckpt_rpath, config):
        """
        Load Pre-trained 
        """
        print('### load params from: ', ckpt_rpath, flush=True)
        state_dict = torch.load(ckpt_rpath, map_location='cpu')
        state_dict = state_dict['model'] if 'model' in state_dict.keys() else state_dict

        state_dict = {k.lstrip("model."): v for k, v in state_dict.items()}

        msg = self.load_state_dict(state_dict, strict=False)

        missing_keys = msg.missing_keys[:]
        if config["freeze_vit"]:
            missing_keys = [p for p in missing_keys if not p.startswith("vision_encoder.")]

        if config["freeze_llm"]:
            tmp = []
            for p in missing_keys:
                if p.startswith("LLM."):
                    if config.get("use_adapter", False):
                        if "adapter" in p:
                            tmp.append(p)
                else:  # bridge
                    tmp.append(p)

            missing_keys = tmp

        self.init_params = [p for p in self.init_params if p in missing_keys]

    def _encode_video_frames(self, frames):
        assert frames.dim() == 5  # (bsz, frame_len, c, h, w)
        bsz = frames.shape[0]

        # Encode Each Frame
        frames = rearrange(frames, 'b f c h w -> (b f) c h w')
        frame_embeds = self.vision_encoder(frames)
        frame_embeds = rearrange(frame_embeds, '(b f) p d -> b f p d', b=bsz)

        if self.add_frame_pos:
            frame_embeds = frame_embeds + self.absolute_frame_pos_embed

        return frame_embeds

    def get_video_embeds(self, frames):
        assert frames.dim() == 5  # (bsz, frame_len, c, h, w)

        if self.video_encoding == 'concate':
            video_embeds = self._encode_video_frames(frames)
            video_embeds = rearrange(video_embeds, 'b f p d -> b (f p) d')
            video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(video_embeds.device)
        else:
            raise NotImplementedError

        return video_embeds, video_atts

    def get_image_embeds(self, image):
        assert image.dim() == 4
        image_embeds = self.vision_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        return image_embeds, image_atts  # full attention

    def get_vision_embeds(self, vision):
        if vision.dim() == 5:  # encode video
            return self.get_video_embeds(vision)

        else:
            assert vision.dim() == 4
            return self.get_image_embeds(vision)

    def embed_tokens(self, input_ids):
        if self.use_dec_only:
            text_embeds = self.LLM.model.embed_tokens(input_ids)
        else:
            raise NotImplementedError

        return text_embeds

    def get_mixed_v2t_feats(self, image, video):

        assert ((image is None) and (video is None)) is False

        with torch.set_grad_enabled(not self.freeze_vit):
            if image is not None:
                image_embeds, image_atts = self.get_vision_embeds(image)

            if video is not None:
                video_embeds, video_atts = self.get_vision_embeds(video)

        img_v2t_feats, img_v2t_atts = None, None
        if image is not None:
            img_v2t_feats, img_v2t_atts = self.bridge(vision_embeds=image_embeds, vision_atts=image_atts)

        video_v2t_feats, video_v2t_atts = None, None
        if video is not None:
            video_v2t_feats, video_v2t_atts = self.bridge(vision_embeds=video_embeds, vision_atts=video_atts)

        def _concate(video, image):
            assert ((video is None) and (image is None)) is False
            if video is None:
                return image
            elif image is None:
                return video
            else:
                return torch.concat([video, image], dim=0)  # video first

        return _concate(video_v2t_feats, img_v2t_feats), _concate(video_v2t_atts, img_v2t_atts)

    @torch.no_grad()
    def generate(self, vision_input, input_ids, input_atts,
                 use_nucleus_sampling=False, num_beams=5, max_new_tokens=64, min_length=2, top_p=0.9, top_k=3,
                 repetition_penalty=1.0, no_repeat_ngram_size=3,
                 length_penalty=1.0, num_return_sequences=1, temperature=1, apply_lemmatizer=False):
        text_embeds = self.embed_tokens(input_ids)

        if vision_input is not None:
            vision_embeds, vision_atts = self.get_vision_embeds(vision_input)
            v2t_feats, v2t_atts = self.bridge(vision_embeds=vision_embeds, vision_atts=vision_atts)

            inputs_embeds = torch.cat([v2t_feats, text_embeds], dim=1)
            attention_mask = torch.cat([v2t_atts, input_atts], dim=1)

        else:
            inputs_embeds = text_embeds
            attention_mask = input_atts

        outputs = self.LLM.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        if apply_lemmatizer:
            output_text = self._lemmatize(output_text)

        return output_text

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        if not hasattr(self, "lemmatizer"):
            import spacy
            self.lemmatizer = spacy.load("en_core_web_sm")

        return [apply(answer) for answer in answers]
