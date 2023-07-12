import torch
from transformers import CLIPImageProcessor
from instruct_blip.models import load_model_and_preprocess
from instruct_blip.models.eva_vit import convert_weights_to_fp16
import cv2
from PIL import Image


class InstructBLIP():
    def __init__(self, model_type="llava", device="cuda"):
        self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device='cpu')
        self.device = device
        self.model_type = model_type
        self.dtype = torch.float16    
        convert_weights_to_fp16(self.model.visual_encoder)            
        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.llm_model = self.model.llm_model.to(self.device, dtype=self.dtype)

    def ask(self, image, question):
        image = Image.open(image).convert("RGB") 
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        output = self.model.generate({"image": image, "prompt": question})[0]
        return output
    
    def caption(self, image):
        question='a photo of'
        image = Image.open(image).convert("RGB")  
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        output = self.model.generate({"image": image, "prompt": question})[0]
        output = output.replace('\n', ' ').strip() 
        return output
    