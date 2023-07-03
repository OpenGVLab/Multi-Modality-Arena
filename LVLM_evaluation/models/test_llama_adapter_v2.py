import torch
from . import llama, get_BGR_image, DATA_DIR


class TestLLamaAdapterV2:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        llama_dir = f"{DATA_DIR}/llama_checkpoints"
        self.model, self.img_transform = llama.load("BIAS-7B", llama_dir, self.device, llama_dir)

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=128):
        img = self.img_transform(get_BGR_image(image)).unsqueeze(0).to(self.device)
        prompt = [llama.format_prompt(question)]
        results = self.model.generate(img, [prompt], max_gen_len=max_new_tokens)
        result = results[0].strip()

        return result
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=128):
        imgs = [get_BGR_image(img) for img in image_list]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = [llama.format_prompt(question) for question in question_list]
        results = self.model.generate(imgs, prompts, max_gen_len=max_new_tokens)
        results = [result.strip() for result in results]

        return results