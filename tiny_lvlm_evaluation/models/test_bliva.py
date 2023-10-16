import torch
from . import get_image
from .bliva.models import load_model_and_preprocess


class TestBLIVA:
    def __init__(self, device=None) -> None:
        device = 'cuda' if device is None else device
        model, vis_processors, _ = load_model_and_preprocess(name="bliva_vicuna", model_type="vicuna7b", is_eval=True, device='cpu')
        vis_processor = vis_processors["eval"]
        self.model = model.to(device, dtype=torch.float16)
        self.device = device
        self.vis_processor = vis_processor

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=30):
        imgs = [get_image(image)]
        imgs = [self.vis_processor(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = [question]
        results = self.model.generate({"image": imgs, "prompt": prompts}, max_length=max_new_tokens)
        result = results[0].strip()
        return result

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=30):
        imgs = [get_image(img) for img in image_list]
        imgs = [self.vis_processor(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = question_list
        with torch.cuda.amp.autocast():
            results = self.model.generate({"image": imgs, "prompt": prompts}, max_length=max_new_tokens)
        results = [result.strip() for result in results]
        return results