import torch
from types import MethodType

from .mplug_owl.processing_mplug_owl import MplugOwlProcessor, MplugOwlImageProcessor
from .mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration, get_media_indices
from .mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
from transformers import AutoTokenizer
from . import get_image


prompt_template = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <image>\nHuman: {}\nAI:"


@torch.no_grad()
def forward_lm(self, start_loc, pixel_values, input_ids):
    attention_mask = input_ids.new_ones(*input_ids.shape)
    media_token_indices = [get_media_indices(input_ids[i]) for i in range(batch_size)]
    num_images_per_sample = [len(x) for x in media_token_indices]
    input_ids = input_ids.clone()  # prevent inplace modify
    input_ids[input_ids < 0] = 0  # Not used

    if hasattr(self, "hf_device_map"):
        self._preprocess_accelerate()
    batch_size = input_ids.shape[0]
    # get text embedding
    inputs_embeds = self.get_input_embeddings()(input_ids)
    # get visual embedding
    if pixel_values is not None:
        pixel_values = pixel_values.to(input_ids.device)
        image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_features = self.abstractor(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
        )["last_hidden_state"]
        torch.ones(query_features.size()[:-1], dtype=torch.long).to(query_features.device)
        img_seq_length = query_features.shape[1]
        image_embeds = query_features
        img_seq_length = image_embeds.shape[1]


    text_chunk_embeds = []
    text_chunk_attns = []
    img_idx = 0
    for b in range(batch_size):
        start = 0
        result = []
        result_attn = []
        for i, pos in enumerate(media_token_indices[b]):
            if pos > start:
                result.append(inputs_embeds[b, start:pos])
                result_attn.append(attention_mask[b, start:pos])
            result.append(image_embeds[img_idx + i])
            result_attn.append(torch.ones(image_embeds[img_idx + i].shape[0], device=inputs_embeds.device))
            start = pos + img_seq_length
        if start < inputs_embeds.shape[1]:
            result.append(inputs_embeds[b, start:])
            result_attn.append(attention_mask[b, start:])

        img_idx += num_images_per_sample[b]
        text_chunk_embeds.append(torch.cat(result, dim=0))
        text_chunk_attns.append(torch.cat(result_attn, dim=0))

    inputs_embeds = torch.stack(text_chunk_embeds, dim=0)
    attention_mask = torch.stack(text_chunk_attns, dim=0)

    text_tokens_ = input_ids.clone()
    labels = text_tokens_.clone().contiguous()
    labels[:,:start_loc] = -100
    labels[:,start_loc+1:] = -100

    loss = []
    for i in range(len(labels)):
        input_embeds_i = inputs_embeds[i,:,:].unsqueeze(0)
        attention_mask_i = attention_mask[i,:].unsqueeze(0)
        labels_i = labels[i,:].unsqueeze(0)
        output_i = self.language_model(inputs_embeds=input_embeds_i, attention_mask=attention_mask_i, labels=labels_i,return_dict=True,output_attentions=self.config.output_attentions)
        loss_i = output_i.loss
        loss.append(loss_i.item())
    loss = 1.0 / loss[0]
    
    return loss


class TestMplugOwl:
    def __init__(self, device=None):
        model_path='MAGAer13/mplug-owl-llama-7b'
        self.model = MplugOwlForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float32)
        self.image_processor = MplugOwlImageProcessor.from_pretrained(model_path)
        # self.tokenizer = MplugOwlTokenizer.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = MplugOwlProcessor(self.image_processor, self.tokenizer)
        
        # from peft import LoraConfig, get_peft_model
        # peft_config = LoraConfig(
        #     target_modules=r'.*language_model.*\.(q_proj|v_proj)', 
        #     inference_mode=False, 
        #     r=8,
        #     lora_alpha=32, 
        #     lora_dropout=0.05
        # )
        # self.model = get_peft_model(self.model, peft_config)
        # self.model.print_trainable_parameters()
        # exit()

        self.model.forward_lm = MethodType(forward_lm, self.model)
        self.model.eval()
        if device is not None:
            self.move_to_device(device)
        
    def move_to_device(self, device=None):
        if device is not None and 'cuda' in device.type:
            self.device = device
            if torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        else:
            self.device = 'cpu'
            self.dtype = torch.float32
        self.model.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=256):
        prompts = [prompt_template.format(question)]
        image = get_image(image)
        inputs = self.processor(text=prompts, images=[image], return_tensors='pt')
        inputs = {k: v.to(self.device, dtype=self.dtype) if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generate_kwargs = {
            'do_sample': True,
            'top_k': 5,
            'max_length': max_new_tokens
        }

        with torch.no_grad():
            res = self.model.generate(**inputs, **generate_kwargs)
        generated_text = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)

        return generated_text

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=256):
        images = [get_image(image) for image in image_list]
        images = [self.image_processor(image, return_tensors='pt').pixel_values for image in images]
        images = torch.cat(images, dim=0).to(self.device, dtype=self.dtype)
        prompts = [prompt_template.format(question) for question in question_list]
        inputs = self.processor(text=prompts)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["pixel_values"] = images

        generate_kwargs = {
            'do_sample': True,
            'top_k': 5,
            'max_length': max_new_tokens
        }

        with torch.no_grad():
            res = self.model.generate(**inputs, **generate_kwargs)
        outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in res.tolist()]

        return outputs

    @torch.no_grad()
    def forward_lm(self, image, prompt, start_loc):
        prompts = [prompt_template.format(prompt)]
        image = get_image(image)
        inputs = self.processor(text=prompts, images=[image], return_tensors='pt')
        inputs = {k: v.to(self.device, dtype=self.dtype) if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        loss = self.model.forward_lm(**inputs, start_loc=start_loc)

        return loss
