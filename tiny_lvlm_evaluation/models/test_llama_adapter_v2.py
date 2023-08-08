import torch
from . import get_BGR_image, DATA_DIR
from . import llama_adapter_v2 as llama

llama_dir = f'{DATA_DIR}/llama_checkpoints'
model_path = f'{DATA_DIR}/llama_checkpoints/llama_adapter_v2_LORA-BIAS-7B.pth' # llama_adapter_v2_BIAS-7B.pth, llama_adapter_v2_LORA-BIAS-7B.pth


class TestLLamaAdapterV2:
    def __init__(self, device=None) -> None:
        # choose from BIAS-7B, LORA-BIAS-7B
        model, preprocess = llama.load(model_path, llama_dir, device, max_seq_len=256, max_batch_size=16)
        model.eval()
        self.img_transform = preprocess
        self.model = model
        self.device = device

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=256):
        imgs = [get_BGR_image(image)]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = [llama.format_prompt(question)]
        results = self.model.generate(imgs, prompts, max_gen_len=max_new_tokens)
        result = results[0].strip()

        return result
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=256):
        imgs = [get_BGR_image(img) for img in image_list]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = [llama.format_prompt(question) for question in question_list]
        results = self.model.generate(imgs, prompts, max_gen_len=max_new_tokens)
        results = [result.strip() for result in results]

        return results
