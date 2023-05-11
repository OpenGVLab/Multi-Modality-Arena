import torch
from .mplug_owl.interface import do_generate, get_model

TOKENIZER_PATH = '/nvme/data1/VLP_web_data/mPLUG-Owl-model/tokenizer.model'
CHECKPOINT_PATH = '/nvme/data1/VLP_web_data/mPLUG-Owl-model/instruction_tuned.pth' # pretrained.pth, instruction_tuned.pth


class TestMplugOwl:
    def __init__(self, checkpoint_path=CHECKPOINT_PATH, tokenizer_path=TOKENIZER_PATH):
        model, tokenizer, img_processor = get_model(
                checkpoint_path=checkpoint_path, tokenizer_path=tokenizer_path, device='cpu', dtype=torch.float32)
        self.model = model
        self.tokenizer = tokenizer
        self.img_processor = img_processor

    def move_to_device(self, device):
        if device is not None and 'cuda' in device.type:
            dtype = torch.float16
            self.model = self.model.to(device, dtype=dtype)
        else:
            dtype = torch.float32
            device = 'cpu'
            self.model = self.model.to('cpu', dtype=dtype)
        
        return device, dtype

    def generate(self, text_input, image, device=None, keep_in_device=False):
        # try:
        prompts = [f'''
            The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
            Human: <image>
            Human: {text_input}
        ''']

        generate_config = {
            "top_k": 5, 
            "max_length": 512,
            "do_sample":True
        }

        device, dtype = self.move_to_device(device)
        generated_text = do_generate(prompts, [image], self.model, self.tokenizer, self.img_processor, **generate_config, device=device, dtype=dtype)

        if not keep_in_device:
            self.model = self.model.to('cpu', dtype=torch.float32)

        return generated_text
        # except Exception as e:
        #     return getattr(e, 'message', str(e))