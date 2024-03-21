import os
import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from ..common.registry import registry
from .blip2 import Blip2Base, disabled_train
from .modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer
from .Qformer import BertConfig

from collections import OrderedDict
from ..common.dist_utils import download_cached_file
from ..common.utils import is_url

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

@registry.register_model("cheetah_vicuna")
class Cheetah_Vicuna(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/cheetah_vicuna.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        freeze_llama_proj=True,
        num_query_token=32,
        llama_model="",
        # prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        update_layer = 16,
        # low_resource=False,  # use 8 bit and put vit in cpu
        # device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        super().__init__()
        print("the vicuna version of cheetah!")
        self.tokenizer = self.init_tokenizer()
        
        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")
        print('Loading Q-Former Done')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        self.llama_model = LlamaForCausalLM.from_pretrained(
            llama_model,
            torch_dtype=torch.float16,
        )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        if freeze_llama_proj:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
        
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.prompt_template = prompt_template
        
        new_query_tokens = self.init_query_tokens(num_query_token)
        
        qformer_proj = zero_module(nn.Linear(
            self.llama_model.config.hidden_size, self.Qformer.config.hidden_size
        ))
        
        new_llm_proj = zero_module(nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        ))

        self.llama_model.set_qformer_and_proj(self.Qformer, qformer_proj, new_llm_proj, new_query_tokens)
        self.init_query_tokens_value(url_or_filename=q_former_model)
        self.update_layer = update_layer
    
    def init_query_tokens_value(self, url_or_filename="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth"):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")
        
        state_dict = checkpoint["model"]
        new_state_dict = OrderedDict()
        for k in list(state_dict.keys()):
            if 'query_tokens' in k:
                new_state_dict[f'llama_model.model.query_tokens'] = state_dict[k]
        self.load_state_dict(new_state_dict, strict=False)

    
    @classmethod
    def init_query_tokens(self, num_query_token):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return query_tokens
    
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def encode_img(self, image):
        device = image.device

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_llama = self.llama_proj(query_output.last_hidden_state)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama, image_embeds

    def prompt_process(self, img_embeds, img_attens, input_text):
        split_prompt = [txt.split('<HereForImage>') for txt in input_text]
        for i in range(len(split_prompt)):
            assert len(split_prompt[i]) == len(img_embeds) + 1, f"Unmatched numbers of image placeholders and images."
        prompt_segs = []
        for i in range(len(img_embeds) + 1):
            prompt_segs.append([p[i] for p in split_prompt])
        
        seg_tokens = [
            self.llama_tokenizer(
                seg, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            for i, seg in enumerate(prompt_segs)
        ]
        
        seg_embs = [self.llama_model.model.embed_tokens(seg_t.input_ids) for seg_t in seg_tokens]
        seg_attns = [seg_t.attention_mask for seg_t in seg_tokens]
        
        mixed_embs = []
        mixed_attns = []
        img_position_list = []
        img_start = 1
        for i in range(len(prompt_segs)):
            mixed_embs.append(seg_embs[i])
            mixed_attns.append(seg_attns[i])
            if i != len(img_embeds):
                mixed_embs.append(img_embeds[i])
                mixed_attns.append(img_attens[i])
                img_start += seg_embs[i].size(1)
                img_end = img_start + img_embeds[i].size(1)
                img_position_list.append((img_start, img_end))
                img_start = img_end
        
        mixed_embs = torch.cat(mixed_embs, dim=1)
        mixed_attns = torch.cat(mixed_attns, dim=1)
        
        return mixed_embs, mixed_attns, img_position_list
    
    def concat_text_input_output(self, input_attns, input_embeds, output_attns, output_embeds, target_ids):
        input_part_targets_len = []
        empty_targets = (
            torch.ones(input_attns.size(), dtype=torch.long).to(input_attns.device).fill_(-100)
        )
        llm_inputs = {"inputs_embeds": [], "attention_mask": [], "targets":[]}
        for i in range(input_attns.size(0)):
            this_input_ones = (torch.nonzero(input_attns[i]).squeeze())[-1]
            input_part_targets_len.append(this_input_ones)
            this_input_ones = this_input_ones + 1
            
            llm_inputs["targets"].append(
                torch.cat([
                    empty_targets[i][:this_input_ones],
                    target_ids[i][:],
                    empty_targets[i][this_input_ones:]
                ])
            )
            
            llm_inputs["inputs_embeds"].append(
                torch.cat([
                    input_embeds[i][:this_input_ones, :],
                    output_embeds[i][:, :],
                    input_embeds[i][this_input_ones:, :]
                ])
            )
            
            llm_inputs["attention_mask"].append(
                torch.cat([
                    input_attns[i][:this_input_ones],
                    output_attns[i][:],
                    input_attns[i][this_input_ones:]
                ])
            )
            
        llm_inputs["inputs_embeds"] = torch.stack(llm_inputs["inputs_embeds"], dim=0)
        llm_inputs["targets"] = torch.stack(llm_inputs["targets"], dim=0)
        llm_inputs["attention_mask"] = torch.stack(llm_inputs["attention_mask"], dim=0)
        
        return llm_inputs, input_part_targets_len
     
    def forward(self, samples):
        image = samples["image"]
        img_list, vit_list, att_list = [], [], []
        if image.dim() == 5:
            for j in range(image.size(2)):
                this_image = image[:,:,j,:,:]
                image_emb, image_att, vit_emb = self.encode_img(this_image)
                img_list.append(image_emb)
                att_list.append(image_att)
                vit_list.append(vit_emb)
        else:
            image_emb, image_att, vit_emb = self.encode_img(image)
            img_list.append(image_emb)
            att_list.append(image_att)
            vit_list.append(vit_emb)

        prompt = samples["text_input"]
        prompt = [self.prompt_template.format(p) for p in prompt]
        
        self.llama_tokenizer.padding_side = "right"
        self.llama_tokenizer.truncation_side = 'left'
        img_embeds, atts_img, img_position_list = self.prompt_process(img_list, att_list, prompt)

        self.llama_tokenizer.padding_side = "right"
        self.llama_tokenizer.truncation_side = 'right' 
        
        text = [t + self.end_sym for t in samples["text_output"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = torch.ones((atts_img.size(0), 1), dtype=torch.long).to(atts_img.device)
        
        img_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        atts_img = torch.cat([atts_bos, atts_img], dim=1)
        
        llm_inputs, input_part_targets_len = self.concat_text_input_output(atts_img, 
                                                                           img_embeds, 
                                                                           to_regress_tokens['attention_mask'], 
                                                                           to_regress_embeds, 
                                                                           targets)

        input_part_targets_len = torch.tensor(input_part_targets_len)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=llm_inputs['inputs_embeds'],
                attention_mask=llm_inputs['attention_mask'],
                return_dict=True,
                labels=llm_inputs['targets'],
                update_layer = self.update_layer,
                image_position_list = img_position_list,
                input_part_targets_len = input_part_targets_len,
                all_image_embeds = torch.stack(vit_list,dim=1),
            )
        loss = outputs.loss

        return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        freeze_llama_proj = cfg.get("freeze_llama_proj", True)
        
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        update_layer = cfg.get("update_layer", 16)

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            freeze_llama_proj=freeze_llama_proj,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            update_layer=update_layer,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of Cheetah
        if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
