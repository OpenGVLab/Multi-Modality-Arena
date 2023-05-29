import torch
from gradio_client import Client


class TestMultiModelGPT:
    def __init__(self) -> None:
        self.model = Client('https://mmgpt.openmmlab.org.cn/')
        self.prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        self.ai_prefix = 'Response'
        self.user_prefix = 'Instruction'
        self.seperator = "\n\n### "
        self.history_buffer = -1
        self.max_new_token = 512
        self.num_beams = 3
        self.temperature = 1.0
        self.top_k = 20
        self.top_p = 1.0
        self.do_sample = True
        self.response_split = "### Response:"

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
            image_name = '.mmgpt_inference.png'
            raw_image.save(image_name)
            output = self.model.predict(question, image_name, self.prompt, self.ai_prefix, self.user_prefix, self.seperator,
                self.history_buffer, self.max_new_token, self.num_beams, self.temperature, self.top_k, self.top_p, self.do_sample, fn_index=1)[3]
            output = output.split(self.response_split)[-1].strip()
            
            return output
        except Exception as e:
            return getattr(e, 'message', str(e))