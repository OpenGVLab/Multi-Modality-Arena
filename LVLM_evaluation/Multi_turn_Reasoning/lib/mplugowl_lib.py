import torch
from mplug_owl.processing_mplug_owl import MplugOwlProcessor, MplugOwlImageProcessor
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
import cv2
from PIL import Image

class MplugOwl():
    def __init__(self, model_type="llava", device="cuda"):
        model_path='MAGAer13/mplug-owl-llama-7b'
        self.model = MplugOwlForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float32)
        self.image_processor = MplugOwlImageProcessor.from_pretrained(model_path)
        self.tokenizer = MplugOwlTokenizer.from_pretrained(model_path)
        self.processor = MplugOwlProcessor(self.image_processor, self.tokenizer)
        self.model.eval()
        self.device = device        
        self.dtype = torch.float16
        self.model_type = model_type        
        self.model.to(device=self.device, dtype=self.dtype)


    def ask(self, image, question):
        prompts = [f'''
            The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
            Human: <image>
            Human: {question}
            AI: 
        ''']

        generate_kwargs = {
            'do_sample': True,
            'top_k': 5,
            'max_length': 512
        }

        image = Image.open(image).convert("RGB")
        inputs = self.processor(text=prompts, images=[image], return_tensors='pt')
        inputs = {k: v.to(self.device, dtype=self.dtype) if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            res = self.model.generate(**inputs, **generate_kwargs)
        generated_text = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
        
        return generated_text
    


    def caption(self, image):
        question='a photo of'
        prompts = [f'''
            The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
            Human: <image>
            Human: {question}
            AI: 
        ''']

        generate_kwargs = {
            'do_sample': True,
            'top_k': 5,
            'max_length': 512
        }
        image = Image.open(image).convert("RGB")
        inputs = self.processor(text=prompts, images=[image], return_tensors='pt')
        inputs = {k: v.to(self.device, dtype=self.dtype) if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            res = self.model.generate(**inputs, **generate_kwargs)
        generated_text = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
        generated_text = generated_text.replace('\n', ' ').strip() 

        return generated_text














