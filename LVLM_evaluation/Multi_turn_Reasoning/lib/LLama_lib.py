from PIL import Image
import os
import cv2
import clip
import torch
import importlib
import numpy as np


models_mae_path = './lib/model_mae.py'
spec = importlib.util.spec_from_file_location('models_mae', models_mae_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
mae_vit_base_patch16 = module.mae_vit_base_patch16
model_ckpt_path = './LLaMA-Adapter-v2/llama_adapter_v2_0518.pth'


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class LLama():
    def __init__(self, model_type="llava", device="cuda"):
        _, img_transform = clip.load("ViT-L/14")
        generator = mae_vit_base_patch16()
        ckpt = torch.load(model_ckpt_path, map_location='cpu')
        ckpt_model = ckpt['model']
        msg = generator.load_state_dict(ckpt_model, strict=False)
        self.device = device
        self.dtype = torch.float16
        self.img_transform = img_transform
        self.generator = generator.to(self.device, dtype=self.dtype)
        self.model_type = model_type
        



    def ask(self, img_path, question, length_penalty=1.0, max_length=30, max_gen_len=64, temperature=0.1, top_p=0.75):
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)[:, :, ::-1]
        img = Image.fromarray(np.uint8(img))
        imgs = [img]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device, dtype=self.dtype)
      
        prompts = [PROMPT_DICT['prompt_no_input'].format_map({'instruction': question})]
        prompts = [self.generator.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        # with torch.cuda.amp.autocast():
        results = self.generator.generate(imgs, prompts, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
        result = results[0].strip()

        return result

    def caption(self, img_path, max_gen_len=64, temperature=0.1, top_p=0.75):
        # TODO: Multiple captions
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)[:, :, ::-1]
        img = Image.fromarray(np.uint8(img))
        imgs = [img]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device, dtype=self.dtype)
    
        question='a photo of'
        prompts = [PROMPT_DICT['prompt_no_input'].format_map({'instruction': question})]
        prompts = [self.generator.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        # with torch.cuda.amp.autocast():
        results = self.generator.generate(imgs, prompts, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
        result = results[0]
        result = result.replace('\n', ' ').strip()  # trim caption
        return result
