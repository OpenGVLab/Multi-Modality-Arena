"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast

from ...common.registry import registry
from .blip2 import Blip2Base, disabled_train
from .modeling_t5 import T5Config, T5ForConditionalGeneration


@registry.register_model("blip2_t5")
class Blip2T5(Blip2Base):
    """
    BLIP2 T5 model.
    Supported model types:
        - pretrain_flant5xl: pretrained model with FlanT5-XL
        - pretrain_flant5xl_vitL: pretrained model with FlanT5-XL
        - pretrain_flant5xxl: pretrained model with FlanT5-XXL
        - caption_coco_flant5xl: fintuned image captioning model with FlanT5-XL
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_t5", "pretrain_flant5xl")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5small_vitL": "configs/models/blip2/blip2_pretrain_flant5small_vitL.yaml",
        "pretrain_flant5base_vitL": "configs/models/blip2/blip2_pretrain_flant5base_vitL.yaml",
        "pretrain_flant5large_vitL": "configs/models/blip2/blip2_pretrain_flant5large_vitL.yaml",
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
        "pretrain_flant5xl_vitL": "configs/models/blip2/blip2_pretrain_flant5xl_vitL.yaml",
        "pretrain_flant5xxl": "configs/models/blip2/blip2_pretrain_flant5xxl.yaml",
        "caption_coco_flant5xl": "configs/models/blip2/blip2_caption_flant5xl.yaml",
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
        t5_model="google/flan-t5-xl",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
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

        if only_proj:
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision.eval()
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer.eval()
            self.query_tokens.requires_grad = False
            logging.info("train only projection layer")

        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
        )

        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            param.data = param.data.bfloat16()

        self.t5_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.max_text_length = max_txt_len
        self.prompt = prompt

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        self.g = torch.Generator()
        self.g.manual_seed(0)

    def forward(self, samples):
        image = samples["image"]

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        image_embeds = image_embeds.float()
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                samples["text_input"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            output_tokens = self.t5_tokenizer(
                samples["text_output"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            # context_embeds, context_atts = self.context_sample(inputs_t5, atts_t5, samples['text_input'])

            # encoder_atts = torch.cat([context_atts, atts_t5, input_tokens.attention_mask], dim=1)
            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            # inputs_embeds = torch.cat([context_embeds, inputs_t5, inputs_embeds], dim=1)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

            return {"loss": loss}

    @torch.no_grad()
    def encode_images(self, image):
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu"))):
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
        # inputs_t5 = query_output.last_hidden_state
        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        return inputs_t5

    def context_sample(self, inputs_t5, atts_t5, tokens, n=4):
        batch_size = inputs_t5.size(0)
        tokens = self.t5_tokenizer(
                tokens,
                padding="longest",
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(inputs_t5.device)
        token_embeds = self.t5_model.encoder.embed_tokens(tokens.input_ids)

        indices = [torch.randperm(batch_size, generator=self.g) for i in range(n)]
        ret_inputs = []
        ret_atts = []
        for idx in indices:
            ret_inputs.append(inputs_t5[idx])
            ret_inputs.append(token_embeds[idx])
            ret_atts.append(atts_t5[idx])
            ret_atts.append(tokens.attention_mask[idx])
        ret_inputs = torch.cat(ret_inputs, 1)
        ret_atts = torch.cat(ret_atts, 1)
        return ret_inputs, ret_atts


    @torch.no_grad()
    def combine_encoding(self, input_list, device="cuda:0"):
        image_list = [x for x in input_list if type(x) is not str]
        image_idxs = [i for i, x in enumerate(input_list) if type(x) is not str]

        text_list = [x for x in input_list if type(x) is str]
        text_idxs = [i for i, x in enumerate(input_list) if type(x) is str]

        input_tensor_list = [None for x in input_list]
        input_textid_list = [None for x in input_list]
        input_atts_list = [None for x in input_list]

        if len(image_list) > 0:
            image = torch.cat(image_list, 0)
            with torch.cuda.amp.autocast(
                enabled=(self.device != torch.device("cpu"))
            ):
                print(image.dtype)
                image_embeds = self.ln_vision(self.visual_encoder(image))
                print(image.dtype)
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

            inputs_opt = self.t5_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)
            inputs_opt = [x.unsqueeze(0) for x in inputs_opt]
            atts_opt = [x.unsqueeze(0) for x in atts_opt]
            for i, idx in enumerate(image_idxs):
                input_tensor_list[idx] = inputs_opt[i]
                input_atts_list[idx] = atts_opt[i]

        # text embeddings and atts
        raw_text_tokens = [self.t5_tokenizer(x, return_tensors="pt", add_special_tokens=False,).to(device) for x in text_list]
        text_input_ids = [x.input_ids for x in raw_text_tokens]
        text_attention_masks = [x.attention_mask for x in raw_text_tokens]
        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            text_input_embeddings = [self.t5_model.encoder.embed_tokens(x) for x in text_input_ids]
            for i, idx in enumerate(text_idxs):
                input_tensor_list[idx] = text_input_embeddings[i]
                input_atts_list[idx] = text_attention_masks[i]
                input_textid_list[idx] = text_input_ids[i]

            input_tensor = torch.cat(input_tensor_list, 1)
        input_atts = torch.cat(input_atts_list, 1)
        # input_textids = torch.cat(input_textid_list, 1)
        return input_tensor, input_atts

    def in_context_predict_answers(
        self,
        context,
        image,
        prompt,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        length_penalty=-1,
        **kwargs
    ):
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu"))):
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
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

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        if prompt:
            text_input = [prompt]

        input_tokens = self.t5_tokenizer(
            text_input, padding="longest", return_tensors="pt"
        ).to(image.device)

        if context is not None:
            encoder_atts = torch.cat([context[1], atts_t5, input_tokens.attention_mask], dim=1)
        else:
            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            if context is not None:
                inputs_embeds = torch.cat([context[0], inputs_t5, inputs_embeds], dim=1)
            else:
                inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                length_penalty=length_penalty,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        if self._apply_lemmatizer:
            output_text = self._lemmatize(output_text)

        return output_text

    def text_gen(
        self,
        context,
        prompt,
        device="cuda:0",
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        length_penalty=-1,
        **kwargs
    ):
        if prompt:
            text_input = [prompt]

        input_tokens = self.t5_tokenizer(
            text_input, padding="longest", return_tensors="pt"
        ).to(device)

        if context is not None:
            encoder_atts = torch.cat([context[1], input_tokens.attention_mask], dim=1)
        else:
            encoder_atts = torch.cat([input_tokens.attention_mask], dim=1)

        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            if context is not None:
                inputs_embeds = torch.cat([context[0], inputs_embeds], dim=1)
            else:
                inputs_embeds = torch.cat([inputs_embeds], dim=1)
            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                length_penalty=length_penalty,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        if self._apply_lemmatizer:
            output_text = self._lemmatize(output_text)

        return output_text

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
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
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
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

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        if isinstance(prompt, str):
            prompt = [prompt] * image.size(0)
        else:
            assert len(prompt) == image.size(
                0
            ), "The number of prompts must be equal to the batch size."

        input_tokens = self.t5_tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        with self.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

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
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
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

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]
        if prompt:
            text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        input_tokens = self.t5_tokenizer(
            text_input, padding="longest", return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        with self.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                length_penalty=length_penalty,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        if self._apply_lemmatizer:
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

        return [apply(answer) for answer in answers]

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

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        only_proj = cfg.get("only_proj", False)
        qformer_weight_path = cfg.get("qformer_weight_path", None)
        proj_weight_path = cfg.get("proj_weight_path", None)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            only_proj=only_proj,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
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
                if k.startswith("t5_model"):
                    ckpt.pop(k)
            for k in list(ckpt.keys()):
                if "proj" not in k:
                    ckpt.pop(k)
            logging.info(f"load projection weights from {proj_weight_path}. weights are:")
            logging.info(_keys2str(ckpt.keys()))
            msg = model.load_state_dict(ckpt, strict=False)
            del ckpt

        return model
