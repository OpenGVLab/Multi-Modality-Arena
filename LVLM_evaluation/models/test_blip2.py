import torch
import contextlib
from types import MethodType
from lavis.models import load_model_and_preprocess
from . import get_image


def new_maybe_autocast(self, dtype=None):
    if torch.cuda.is_bf16_supported() and dtype is torch.bfloat16:
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return contextlib.nullcontext()


class TestBlip2:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=self.device
        )
        self.model.maybe_autocast = MethodType(new_maybe_autocast, self.model)

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=30):
        image = get_image(image)
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        answer = self.model.generate({
            "image": image, "prompt": f"Question: {question} Answer:"
        }, max_length=max_new_tokens)

        return answer[0]
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=30):
        imgs = [get_image(img) for img in image_list]
        imgs = [self.vis_processors["eval"](x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = [f"Question: {question} Answer:" for question in question_list]
        output = self.model.generate({"image": imgs, "prompt": prompts}, max_length=max_new_tokens)

        return output