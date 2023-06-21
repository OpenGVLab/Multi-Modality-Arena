import torch
import contextlib
from types import MethodType
from lavis.models import load_model_and_preprocess
from lavis.models.eva_vit import convert_weights_to_fp16
from . import get_image


def new_maybe_autocast(self, dtype=None):
    enable_autocast = self.device != torch.device("cpu")
    if not enable_autocast:
        return contextlib.nullcontext()
    elif dtype is torch.bfloat16:
        if torch.cuda.is_bf16_supported():
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
        else:
            return torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        return torch.cuda.amp.autocast(dtype=dtype)


@torch.no_grad()
def forward_lm(self, samples):
    image = samples["image"]

    with self.maybe_autocast():
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

    inputs_t5 = self.t5_proj(query_output.last_hidden_state)
    atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

    with self.maybe_autocast(dtype=torch.bfloat16):
        output_tokens = self.t5_tokenizer(
            samples["prompt"],
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)

        targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
        )

        start_loc = samples["start_loc"]
        targets[0, :start_loc] = -100
        targets[0, start_loc + 1:] = -100

        empty_targets = (
            torch.ones(atts_t5.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.opt_model.model.decoder.embed_tokens(output_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_t5, output_tokens.attention_mask], dim=1)

        outputs = self.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )

    return outputs


class TestBlip2:
    def __init__(self, device=None) -> None:
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device='cpu'
        )
        self.model.maybe_autocast = MethodType(new_maybe_autocast, self.model)
        self.model.forward_lm = MethodType(forward_lm, self.model)

        if device is not None:
            self.move_to_device(device)

    def move_to_device(self, device):
        if device is not None and 'cuda' in device.type:
            self.dtype = torch.float16
            self.device = device
            convert_weights_to_fp16(self.model.visual_encoder)
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
            self.model.visual_encoder = self.model.visual_encoder.to(self.device, dtype=self.dtype)
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=30):
        image = get_image(image)
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device, dtype=self.dtype)
        answer = self.model.generate({
            "image": image, "prompt": f"Question: {question} Answer:"
        }, max_length=max_new_tokens)

        return answer[0]
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=30):
        imgs = [get_image(img) for img in image_list]
        imgs = [self.vis_processors["eval"](x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device, dtype=self.dtype)
        prompts = [f"Question: {question} Answer:" for question in question_list]
        output = self.model.generate({"image": imgs, "prompt": prompts}, max_length=max_new_tokens)

        return output

    # NOTE: ImageNetVC required
    def forward_lm(self, image, prompt, start_loc):
        image = get_image(image)
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device, dtype=self.dtype)
        output = self.model.forward_lm({"image": image, "prompt": prompt, "start_loc": start_loc})
        loss = output.loss
        return loss