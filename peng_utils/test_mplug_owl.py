import torch
from .mplug_owl.processing_mplug_owl import MplugOwlProcessor, MplugOwlImageProcessor
from .mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from .mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer


class TestMplugOwl:
    def __init__(self, model_path='MAGAer13/mplug-owl-llama-7b'):
        self.model = MplugOwlForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float32)
        self.image_processor = MplugOwlImageProcessor.from_pretrained(model_path)
        self.tokenizer = MplugOwlTokenizer.from_pretrained(model_path)
        self.processor = MplugOwlProcessor(self.image_processor, self.tokenizer)
        self.model.eval()
        
    def move_to_device(self, device=None):
        if device is not None and 'cuda' in device.type:
            dtype = torch.bfloat16
            self.model.bfloat16()
        else:
            dtype = torch.float32
            device = 'cpu'
            self.model.float()
        self.model.to(device=device)
        
        return device, dtype

    def generate(self, text_input, image, device=None, keep_in_device=False):
        try:
            prompts = [f'''
                The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
                Human: <image>
                Human: {text_input}
                AI: 
            ''']

            generate_kwargs = {
                'do_sample': True,
                'top_k': 5,
                'max_length': 512
            }

            device, dtype = self.move_to_device(device)
            inputs = self.processor(text=prompts, images=[image], return_tensors='pt')
            inputs = {k: v.to(device, dtype=dtype) if v.dtype == torch.float else v for k, v in inputs.items()}
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                res = self.model.generate(**inputs, **generate_kwargs)
            generated_text = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)

            if not keep_in_device:
                self.move_to_device()

            return generated_text
        except Exception as e:
            return getattr(e, 'message', str(e))