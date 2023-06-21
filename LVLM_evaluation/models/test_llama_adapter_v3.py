import torch
from . import get_image, DATA_DIR
from . import llama_adapter_v3 as llama

llama_dir = f'{DATA_DIR}/llama_checkpoints'
model_path = f'{DATA_DIR}/llama_checkpoints/llama-adapter-v3-400M-30token-pretrain100-finetune3/converted_checkpoint-3.pth'

ckpt_map = {
    '400M-p100-f0': 'llama-adapter-v3-400M-30token-pretrain100-finetune3/converted_checkpoint-0.pth',
    '400M-p100-f3': 'llama-adapter-v3-400M-30token-pretrain100-finetune3/converted_checkpoint-3.pth',
    '900M-p29-f3': 'llama-adapter-v3-900M-30token-pretrain29-finetune3/converted_checkpoint-3.pth'
}


class TestLLamaAdapterV3:
    def __init__(self, model_name, device=None) -> None:
        ckpt_name = model_name.split('_')[1]
        model_path = f"{DATA_DIR}/llama_checkpoints/{ckpt_map[ckpt_name]}"
        self.model, self.img_transform = llama.load(model_path, llama_dir, device='cpu', max_seq_len=256, max_batch_size=64)

        if device is not None:
            self.move_to_device(device)

    def move_to_device(self, device):
        if type(device) is str and 'cuda' in device:
            self.device = device
        elif type(device) is torch.device and 'cuda' in device.type:
            self.device = device
        else:
            self.device = 'cpu'
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def generate(self, image, question):
        imgs = [get_image(image)]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = [llama.format_prompt(question)]
        results = self.model.generate(imgs, prompts)
        result = results[0].strip()

        return result
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list):
        imgs = [get_image(img) for img in image_list]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = [llama.format_prompt(question) for question in question_list]
        results = self.model.generate(imgs, prompts)
        results = [result.strip() for result in results]

        return results