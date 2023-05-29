import torch
from transformers import CLIPImageProcessor
from .instruct_blip.models import load_model_and_preprocess
from .instruct_blip.models.eva_vit import convert_weights_to_fp16


class TestInstructBLIP:
    def __init__(self) -> None:
        self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device='cpu')

    def move_to_device(self, device=None):
        if device is not None and 'cuda' in device.type:
            dtype = torch.float16
            convert_weights_to_fp16(self.model.visual_encoder)
        else:
            dtype = torch.float32
            device = 'cpu'
            self.model.visual_encoder = self.model.visual_encoder.to(device, dtype=dtype)
        self.model = self.model.to(device, dtype=dtype)
        self.model.llm_model = self.model.llm_model.to(device, dtype=dtype)
        
        return device, dtype

    def generate(self, question, raw_image, device=None, keep_in_device=False):
        try:
            device, dtype = self.move_to_device(device)
            image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            output = self.model.generate({"image": image, "prompt": question})[0]

            if not keep_in_device:
                self.move_to_device()
            
            return output
        except Exception as e:
            return getattr(e, 'message', str(e))
    