import torch
from lavis.models import load_model_and_preprocess


class TestBlip2:
    def __init__(self) -> None:
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device='cpu'
        )

    def move_to_device(self, device):
        if device is not None and 'cuda' in device.type:
            dtype = torch.float16
            self.model = self.model.to(device, dtype=dtype)
            self.model.ln_vision = self.model.ln_vision.to(device, dtype=torch.float32)
        else:
            dtype = torch.float32
            self.model = self.model.to('cpu', dtype=dtype)

        return device, dtype

    def generate(self, question, raw_image, device=None, keep_in_device=False):
        try:
            device, dtype = self.move_to_device(device)
            image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.model.device, dtype=dtype)
            answer = self.model.generate({
                "image": image, "prompt": f"Question: {question} Answer:"
            })

            if not keep_in_device:
                self.model = self.model.to('cpu', dtype=torch.float32)
            
            return answer[0]
        except Exception as e:
            return getattr(e, 'message', str(e))
    