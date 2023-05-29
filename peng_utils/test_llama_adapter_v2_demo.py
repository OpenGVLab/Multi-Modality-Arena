import torch
from gradio_client import Client


class TestLLamaAdapterV2:
    def __init__(self) -> None:
        self.model = Client("http://106.14.127.192:8088/")
        self.max_length = 64
        self.temperature = 0.1
        self.top_p = 0.75

    def move_to_device(self, device=None):
        if device is not None and 'cuda' in device.type:
            dtype = torch.float16
        else:
            dtype = torch.float32
            device = 'cpu'
        
        return device, dtype

    def generate(self, question, raw_image, device=None, keep_in_device=False):
        try:
            device, dtype = self.move_to_device(device)
            image_name = '.llama_adapter_v2_inference.png'
            raw_image.save(image_name)
            output = self.model.predict(image_name, question, self.max_length, self.temperature, self.top_p, fn_index=1)
            
            return output
        except Exception as e:
            return getattr(e, 'message', str(e))
    