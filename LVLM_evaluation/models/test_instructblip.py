import torch
from types import MethodType
from .instruct_blip.models import load_model_and_preprocess
from .instruct_blip.models.eva_vit import convert_weights_to_fp16
from . import get_image


@torch.no_grad()
def forward_lm(self, samples):
    image = samples["image"]
    with self.maybe_autocast():
        image_embeds = self.ln_vision(self.visual_encoder(image))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

    bs = image.size(0)

    query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
    if self.qformer_text_input:
        text_Qformer = self.tokenizer(
            samples["text_input"],
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)

        query_output = self.Qformer.bert(
            text_Qformer.input_ids,
            attention_mask=Qformer_atts,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
    else:
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

    inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
    atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

    self.llm_tokenizer.padding_side = "right"
    self.llm_tokenizer.truncation_side = 'left'
    llm_tokens = self.llm_tokenizer(
        [samples['prompt']] * bs,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=self.max_txt_len,
    ).to(image.device)

    # do not apply loss to the padding
    targets = llm_tokens['input_ids'].masked_fill(
        llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
    )
    start_loc = samples["start_loc"]
    targets[0, :start_loc] = -100
    targets[0, start_loc + 1:] = -100

    # do not apply loss to the query tokens
    empty_targets = (
        torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)
    )
    targets = torch.cat([empty_targets, targets], dim=1)

    inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
    inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
    attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

    with self.maybe_autocast():
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )

    return outputs


class TestInstructBLIP:
    def __init__(self, device=None) -> None:
        self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device='cpu')
        self.model.forward_lm = MethodType(forward_lm, self.model)

        if device is not None:
            self.move_to_device(device)

    def move_to_device(self, device=None):
        if device is not None and 'cuda' in device.type:
            self.dtype = torch.float16
            self.device = device
            convert_weights_to_fp16(self.model.visual_encoder)
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
            self.model.visual_encoder = self.model.visual_encoder.to(self.device, dtype=self.dtype)
        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.llm_model = self.model.llm_model.to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def generate(self, image, question):
        image = get_image(image)
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        output = self.model.generate({"image": image, "prompt": question})[0]

        return output
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list):
        imgs = [get_image(img) for img in image_list]
        imgs = [self.vis_processors["eval"](x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = question_list
        output = self.model.generate({"image": imgs, "prompt": prompts})

        return output
    
    def forward_lm(self, image, prompt, start_loc):
        image = get_image(image)
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        output = self.model.forward_lm({"image": image, "prompt": prompt, "start_loc": start_loc})
        loss = output.loss
        return loss
    