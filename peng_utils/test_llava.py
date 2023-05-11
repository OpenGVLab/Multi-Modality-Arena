import torch
from transformers import AutoTokenizer, AutoConfig
from .llava import conv_templates, LlavaLlamaForCausalLM
from transformers import CLIPImageProcessor, StoppingCriteria

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def patch_config(config):
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024
    }

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


class TestLLaVA:
    def __init__(self, model_path="liuhaotian/LLaVA-Lightning-7B-delta-v1-1"):
        device, dtype = 'cpu', torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        patch_config(model_path)
        model = LlavaLlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
        image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=dtype)

        mm_use_im_start_end = False # getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_tower = model.model.vision_tower[0]
        vision_tower.to(device=device, dtype=dtype)
        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mm_use_im_start_end = mm_use_im_start_end
        self.image_token_len = image_token_len # 256
        self.conv_mode = 'simple'

    def generate(self, text_input, image=None, device=None):
        try:
            if device is not None and 'cuda' in device.type:
                self.model = self.model.to(device)
            else:
                device = 'cpu'

            qs = text_input
            cur_prompt = qs
            if self.mm_use_im_start_end:
                qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len + DEFAULT_IM_END_TOKEN
            else:
                qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len

            if self.conv_mode == 'simple_legacy':
                qs += '\n\n### Response:'
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            prompt = conv.get_prompt()
            inputs = self.tokenizer([prompt])
            input_ids = torch.as_tensor(inputs.input_ids).to(device)

            if image is not None:
                image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                images = image_tensor.unsqueeze(0).to(device)
            else:
                images = None
            
            keywords = ['###']
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=images,
                    do_sample=True,
                    temperature=0.7,
                    max_new_tokens=1024,
                    stopping_criteria=[stopping_criteria])

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

            print(f'Check outputs: {outputs}')

            if self.conv_mode == 'simple_legacy':
                while True:
                    cur_len = len(outputs)
                    outputs = outputs.strip()
                    for pattern in ['###', 'Assistant:', 'Response:']:
                        if outputs.startswith(pattern):
                            outputs = outputs[len(pattern):].strip()
                    if len(outputs) == cur_len:
                        break

            try:
                index = outputs.index(conv.sep)
            except ValueError:
                outputs += conv.sep
                index = outputs.index(conv.sep)

            outputs = outputs[:index].strip()

            return outputs
        except Exception as e:
            return getattr(e, 'message', str(e))
