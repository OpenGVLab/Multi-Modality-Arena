import os
import importlib
from gradio_client import Client

import clip
import torch

from . import get_BGR_image, DATA_DIR
from .llama_adapter_v2.models_mae import mae_vit_base_patch16


# # NOTE: please use customized clip and timm library

# models_mae_path = 'models/llama_adapter_v2/models_mae.py'
# spec = importlib.util.spec_from_file_location('models_mae', models_mae_path)
# module = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(module)
# mae_vit_base_patch16 = module.mae_vit_base_patch16
model_ckpt_path = f'{DATA_DIR}/llama_checkpoints/llama_adapter_v2_0518.pth'


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


class TestLLamaAdapterV2_web:
    def __init__(self, device=None) -> None:
        self.model = Client("http://106.14.127.192:8088/")
        self.max_length = 64
        self.temperature = 0.1
        self.top_p = 0.75

        if device is not None:
            self.move_to_device(device)
    
    def move_to_device(self, device):
        pass

    @torch.no_grad()
    def generate(self, image, question: str):
        image = get_BGR_image(image)
        image_name = '.llama_adapter_v2_inference.png'
        image.save(image_name)
        output = self.model.predict(image_name, question, self.max_length, self.temperature, self.top_p, fn_index=1)
        
        return output


class TestLLamaAdapterV2:
    def __init__(self, device=None) -> None:
        _, img_transform = clip.load("ViT-L/14")
        generator = mae_vit_base_patch16()
        ckpt = torch.load(model_ckpt_path, map_location='cpu')
        ckpt_model = ckpt['model']
        msg = generator.load_state_dict(ckpt_model, strict=False)

        self.img_transform = img_transform
        self.generator = generator

        if device is not None:
            self.move_to_device(device)

    def move_to_device(self, device):
        if type(device) is str and 'cuda' in device:
            self.dtype = torch.float16
            self.device = device
        elif type(device) is torch.device and 'cuda' in device.type:
            self.dtype = torch.float16
            self.device = device
        else:
            self.device = 'cpu'
            self.dtype = torch.float32
        self.generator = self.generator.to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def generate(self, image, question, max_gen_len=256, temperature=0.1, top_p=0.75):
        imgs = [get_BGR_image(image)]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device, dtype=self.dtype)

        prompts = [PROMPT_DICT['prompt_no_input'].format_map({'instruction': question})]
        prompts = [self.generator.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        # with torch.cuda.amp.autocast():
        results = self.generator.generate(imgs, prompts, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
        result = results[0].strip()

        return result
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_gen_len=256, temperature=0.1, top_p=0.75):
        imgs = [get_BGR_image(img) for img in image_list]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device, dtype=self.dtype)
        prompts = [PROMPT_DICT['prompt_no_input'].format_map({'instruction': question}) for question in question_list]
        results = self.generator.generate(imgs, prompts, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
        results = [result.strip() for result in results]

        return results

    