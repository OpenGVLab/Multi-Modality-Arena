import json
import os
import glob
import sys
import time
from pathlib import Path
from typing import Tuple
import clip
import cv2

from PIL import Image
import torch

from .llama_adapter_v2.models_mae import mae_vit_base_patch16

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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

class TestLLaMAAdapterv2:
    def __init__(self, model_path):
        _, self.img_transform = clip.load("ViT-L/14")
        self.model = mae_vit_base_patch16().cuda()     
        ckpt = torch.load(model_path, map_location='cpu') # llama-adapter-v2_7B_Lora
        ckpt_model = ckpt['model']
        msg = self.model.load_state_dict(ckpt_model, strict=False)
        # print(msg)
    
    def move_to_device(self, device=None):
        if device is not None and 'cuda' in device.type:
            dtype = torch.float16
        else:
            dtype = torch.float32
            device = 'cpu'
        
        return device, dtype

    def generate(self, text_input, image=None, device=None,
                # model,
                max_gen_len=256, temperature: float = 0.1, top_p: float = 0.75,
                ):
        try:
            device, _ = self.move_to_device(device)   
            imgs = [image]
            imgs = [self.img_transform(x) for x in imgs]
            imgs = torch.stack(imgs, dim=0).cuda().half()

            prompts = [PROMPT_DICT['prompt_no_input'].format_map({'instruction': text_input})]

            prompts = [self.model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
            # with torch.cuda.amp.autocast():
            results = self.generate(imgs, prompts, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
            result = results[0].strip()
            return result
        except Exception as e:
            return getattr(e, 'message', str(e))
        