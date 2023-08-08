import torch
import contextlib
from types import MethodType
from lavis.models import load_model_and_preprocess
from . import get_image


def new_maybe_autocast(self, dtype=None):
    enable_autocast = self.device != torch.device("cpu")
    if not enable_autocast:
        return contextlib.nullcontext()
    elif dtype is torch.bfloat16 and torch.cuda.is_bf16_supported():
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return contextlib.nullcontext()


class TestBlip2:
    def __init__(self, device=None) -> None:
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device='cpu'
        )
        self.model.maybe_autocast = MethodType(new_maybe_autocast, self.model)

        if device is not None:
            self.move_to_device(device)

    def move_to_device(self, device):
        if device is not None and 'cuda' in device.type:
            self.dtype = torch.float32 # torch.float16
            self.device = device
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
            self.model.visual_encoder = self.model.visual_encoder.to(self.device, dtype=self.dtype)
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=30):
        image = get_image(image)
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device, dtype=self.dtype)
        answer = self.model.generate({
            "image": image, "prompt": f"Question: {question} Short answer:"
        }, max_length=max_new_tokens)

        return answer[0]
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=30):
        imgs = [get_image(img) for img in image_list]
        imgs = [self.vis_processors["eval"](x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device, dtype=self.dtype)
        prompts = [f"Question: {question} Short answer:" for question in question_list]
        output = self.model.generate({"image": imgs, "prompt": prompts}, max_length=max_new_tokens)

        return output
    