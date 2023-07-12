import torch
from transformers import CLIPImageProcessor
from otter.modeling_otter import OtterForConditionalGeneration
import cv2
from PIL import Image
CKPT_PATH='./VLP_web_data/otter-9b-hf'

class Otter():
    def __init__(self, model_type="llava", device="cuda"):
        model_path=CKPT_PATH
        self.model = OtterForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = self.model.text_tokenizer
        self.image_processor = CLIPImageProcessor()
        self.tokenizer.padding_side = "left"
        self.dtype = torch.float16
        self.device = device
        self.model_type = model_type               
        self.model = self.model.to(self.device, dtype=self.dtype)
        #self.model.vision_encoder = self.model.vision_encoder.to('cpu', dtype=torch.float32)


    def ask(self, image, question):
        #img = cv2.imread(image)
        #image = Image.fromarray(img)  
        image = Image.open(image).convert("RGB") 
        vision_x = (self.image_processor.preprocess([image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0))
        lang_x = self.model.text_tokenizer([f"<image> User: {question} GPT: <answer>"], return_tensors="pt")
        generated_text = self.model.generate(
            vision_x=vision_x.to(self.model.device, dtype=self.dtype),
            #vision_x=vision_x.to('cpu'),
            lang_x=lang_x["input_ids"].to(self.model.device),
            attention_mask=lang_x["attention_mask"].to(self.model.device, dtype=self.dtype),
            max_new_tokens=256,
            num_beams=3,
            no_repeat_ngram_size=3,
        )
        output = self.model.text_tokenizer.decode(generated_text[0])
        output = [x for x in output.split(' ') if not x.startswith('<')]
        out_label = output.index('GPT:')
        output = ' '.join(output[out_label + 1:])
        
        return output
    
    def caption(self, image):
        #img = cv2.imread(image)
        #image = Image.fromarray(img)  
        image = Image.open(image).convert("RGB") 
        question = ""
        vision_x = (self.image_processor.preprocess([image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0))
        lang_x = self.model.text_tokenizer([f"<image> User: {question} GPT: <answer>"], return_tensors="pt")
        generated_text = self.model.generate(
            vision_x=vision_x.to(self.model.device, dtype=self.dtype),
            #vision_x=vision_x.to('cpu'),
            lang_x=lang_x["input_ids"].to(self.model.device),
            attention_mask=lang_x["attention_mask"].to(self.model.device, dtype=self.dtype),
            max_new_tokens=256,
            num_beams=3,
            no_repeat_ngram_size=3,
        )
        output = self.model.text_tokenizer.decode(generated_text[0])
        output = [x for x in output.split(' ') if not x.startswith('<')]
        out_label = output.index('GPT:')
        output = ' '.join(output[out_label + 1:])
        output = output.replace('\n', ' ').strip() 
        return output
    