"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import pickle

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from ...common.registry import registry
from .blip2 import Blip2Base, disabled_train
from .modeling_llama import LlamaConfig, LlamaForCausalLM
from transformers import AutoTokenizer
import random
from transformers import StoppingCriteria, StoppingCriteriaList
import torch.nn.functional as F
class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[]):
        super().__init__()
        self.stops = stops
        self.prompt_len = 0

    def _contains_subsequence(self, large_tensor, small_tensor):
        len_small = len(small_tensor)
        for i in range(0, len(large_tensor)-len_small+1):
            flag = torch.all((small_tensor == large_tensor[i: i+len_small])).item()
            if flag:
                return True
        return False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for x in input_ids:
            end_now = False
            for stop in self.stops:
                stop = stop.to(x.device)
                end_now |= self._contains_subsequence(x[self.prompt_len:], stop)
                # if torch.all((stop == input_ids[i][-len(stop):])).item():
                #     return True
            if not end_now:
                return False
        return True

@registry.register_model("blip2_vicuna")
class Blip2Vicuna(Blip2Base):
    """
    BLIP2 LLaMA model.
    Supported model types:
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_vicuna", "pretrain_vicuna")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna7b": "configs/models/blip2/blip2_pretrain_vicuna7b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        only_proj=False,
        num_query_token=32,
        llama_model=None,
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
        end_sym="\n",
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        # freeze ln_vision and qformer
        if only_proj: # use False to calculate the trainable parameters of VPGTrans
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision.eval()
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer.eval()
            self.query_tokens.requires_grad = False
            logging.info("train only projection layer")

        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side="left"
        self.llama_model = LlamaForCausalLM.from_pretrained(
            llama_model, torch_dtype=torch.float16
        )
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        self.eos_token_id = self.llama_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[-1]
        self.end_sym = end_sym

        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.llama_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        prompt_template = "###Human: {} ###Assistant: "
        raw_prompts= ['<Img><ImageHere></Img> Describe this image in detail.',
                      '<Img><ImageHere></Img> Take a look at this image and describe what you notice.',
                      '<Img><ImageHere></Img> Please provide a detailed description of the picture.',
                      '<Img><ImageHere></Img> Could you describe the contents of this image for me?']
        filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
        self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
        print('Load {} training prompts'.format(len(self.prompt_list)))
        print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))

        stop_words_ids = [torch.tensor([835]),
                          torch.tensor([2277, 29937])]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])


    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()


    def encode_img(self, image):
        device = image.device
        # if self.low_resource:
        #     self.vit_to_cpu()
        #     image = image.to("cpu")

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
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def forward(self, samples):
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        if hasattr(samples, 'question_split'):  # VQA dataset
            print('VQA Batch')
            vqa_prompt = '###Human: <Img><ImageHere></Img> '
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)

        self.llama_tokenizer.padding_side = "right"

        text = [t + self.end_sym for t in samples["text_input"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}


    @torch.no_grad()
    def forward_lm(self, samples):
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        if hasattr(samples, 'question_split'):  # VQA dataset
            print('VQA Batch')
            vqa_prompt = '###Human: <Img><ImageHere></Img> '
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)

        self.llama_tokenizer.padding_side = "right"

        #text = [t + self.end_sym for t in samples["text_input"]]
        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        prompt = [prompt] * image.size(0)
        to_regress_tokens = self.llama_tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        start_loc = samples["start_loc"]
        targets[0, :start_loc] = -100
        targets[0, start_loc + 1:] = -100  # the location of candidates


        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=300,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        stop_criteria=True,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        with self.maybe_autocast():
            image = samples["image"]
            img_embeds, atts_img = self.encode_img(image)
            vqa_prompt = '###Human: <Img><ImageHere></Img> '
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)


            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            prompt = [prompt] * image.size(0)

            llama_tokens = self.llama_tokenizer(prompt, return_tensors="pt").to(
                image.device
            )
            input_ids = llama_tokens.input_ids
            prompt_length = llama_tokens.input_ids.shape[1]
            attention_mask = torch.cat([atts_img, llama_tokens.attention_mask], dim=1)

            if use_nucleus_sampling:
                query_embeds = img_embeds#.repeat_interleave(num_captions, dim=0)
                num_beams = 1
            else:
                query_embeds = img_embeds#.repeat_interleave(num_beams, dim=0)
            if stop_criteria:
                self.stopping_criteria[0].prompt_len = prompt_length
                outputs = self.llama_model.generate(
                    input_ids=input_ids,
                    query_embeds=query_embeds,
                    attention_mask=attention_mask,
                    do_sample=use_nucleus_sampling,
                    stopping_criteria=self.stopping_criteria,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=max_length,
                    min_length=min_length,
                    # eos_token_id=self.eos_token_id,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                )
            else:
                outputs = self.llama_model.generate(
                    input_ids=input_ids,
                    query_embeds=query_embeds,
                    attention_mask=attention_mask,
                    do_sample=use_nucleus_sampling,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=max_length,
                    min_length=min_length,
                    # eos_token_id=self.eos_token_id,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                )

            prompt_length = llama_tokens.input_ids.shape[1]
            output_text = self.llama_tokenizer.batch_decode(
                outputs[:, prompt_length:], skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
            return output_text

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        **kwargs
    ):

        image = samples["image"]
        with torch.cuda.amp.autocast(
                enabled=(self.device != torch.device("cpu"))
        ):
            # import pdb
            # pdb.set_trace()
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_llama = self.llama_proj(query_output.last_hidden_state)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)

            if isinstance(samples["text_input"], str):
                samples["text_input"] = [samples["text_input"]]
            if prompt:
                text_input = [prompt.format(question) for question in samples["text_input"]]
            else:
                text_input = samples["text_input"]

            llama_tokens = self.llama_tokenizer(text_input, padding="longest", return_tensors="pt").to(image.device)
            input_ids = llama_tokens.input_ids
            attention_mask = torch.cat([atts_llama, llama_tokens.attention_mask], dim=1)

            query_embeds = inputs_llama#.repeat_interleave(num_beams, dim=0)

            outputs = self.llama_model.generate(
                input_ids=input_ids,
                query_embeds=query_embeds,
                attention_mask=attention_mask,
                stopping_criteria=self.stopping_criteria,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                eos_token_id=self.eos_token_id,
                length_penalty=length_penalty,
            )
            prompt_length = llama_tokens.input_ids.shape[1]
            output_text = self.llama_tokenizer.batch_decode(
                outputs[:, prompt_length:], skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
            if self._apply_lemmatizer:
                output_text = self._lemmatize(output_text)
            return output_text

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

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

        return [apply(answer) for answer in answers]

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 160)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        only_proj = cfg.get("only_proj", False)
        qformer_weight_path = cfg.get("qformer_weight_path", None)
        proj_weight_path = cfg.get("proj_weight_path", None)
        end_sym = cfg.get("end_sym", '\n')
        print(cfg.get("pretrained", "none"))
        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            only_proj=only_proj,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt=prompt,
            max_txt_len=160,
            apply_lemmatizer=apply_lemmatizer,
            end_sym=end_sym
        )
        model.load_checkpoint_from_config(cfg)

        def _keys2str(keys):
            keys = [k.split(".")[0] for k in keys]
            from collections import Counter
            return Counter(keys)
        if qformer_weight_path is not None and qformer_weight_path!="":
            ckpt = torch.load(qformer_weight_path, "cpu")
            ckpt = ckpt['model']
            for k in list(ckpt.keys()):
                if k.split(".")[0] not in ["query_tokens", "ln_vision", "Qformer"]:
                    ckpt.pop(k)
            logging.info(f"load qformer weights from {qformer_weight_path}. weights are:")
            logging.info(_keys2str(ckpt.keys()))
            msg = model.load_state_dict(ckpt, strict=False)
            del ckpt
        if proj_weight_path is not None and proj_weight_path!="":
            ckpt = torch.load(proj_weight_path, "cpu")
            ckpt = ckpt["model"]
            for k in list(ckpt.keys()):
                if "proj" not in k:
                    ckpt.pop(k)
            logging.info(f"load projection weights from {proj_weight_path}. weights are:")
            logging.info(_keys2str(ckpt.keys()))
            msg = model.load_state_dict(ckpt, strict=False)
            del ckpt
        torch.cuda.empty_cache()
        return model
