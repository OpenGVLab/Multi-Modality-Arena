import torch
from transformers import CLIPImageProcessor
from .otter.modeling_otter import OtterForConditionalGeneration
from .instruct_blip.models.eva_vit import convert_weights_to_fp16
from . import get_image, DATA_DIR

CKPT_PATH=f'{DATA_DIR}/otter-9b-hf'


class TestOtter:
    def __init__(self) -> None:
        model_path=CKPT_PATH
        self.model = OtterForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = self.model.text_tokenizer
        self.image_processor = CLIPImageProcessor()
        self.tokenizer.padding_side = "left"
        self.move_to_device()

    def move_to_device(self):
        if torch.cuda.is_available():
            self.dtype = torch.float16
            self.device = 'cuda'
            convert_weights_to_fp16(self.model.vision_encoder)
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
            self.model.vision_encoder = self.model.vision_encoder.to(self.device, dtype=self.dtype)
        self.model = self.model.to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=256):
        image = get_image(image)
        vision_x = (self.image_processor.preprocess([image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0))
        lang_x = self.model.text_tokenizer([f"<image> User: {question} GPT: <answer>"], return_tensors="pt")
        generated_text = self.model.generate(
            vision_x=vision_x.to(self.model.device, dtype=self.dtype),
            lang_x=lang_x["input_ids"].to(self.model.device),
            attention_mask=lang_x["attention_mask"].to(self.model.device, dtype=self.dtype),
            max_new_tokens=max_new_tokens,
            num_beams=3,
            no_repeat_ngram_size=3,
        )
        output = self.model.text_tokenizer.decode(generated_text[0])
        output = [x for x in output.split(' ') if not x.startswith('<')]
        out_label = output.index('GPT:')
        output = ' '.join(output[out_label + 1:])

        return output
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=256):
        imgs = [get_image(img) for img in image_list]
        imgs = [self.image_processor.preprocess([x], return_tensors="pt")["pixel_values"].unsqueeze(0) for x in imgs]
        vision_x = (torch.stack(imgs, dim=0))
        prompts = [f"<image> User: {question} GPT: <answer>" for question in question_list]
        lang_x = self.model.text_tokenizer(prompts, return_tensors="pt", padding=True)
        generated_text = self.model.generate(
            vision_x=vision_x.to(self.model.device, dtype=self.dtype),
            lang_x=lang_x["input_ids"].to(self.model.device),
            attention_mask=lang_x["attention_mask"].to(self.model.device, dtype=self.dtype),
            max_new_tokens=max_new_tokens,
            num_beams=3,
            no_repeat_ngram_size=3,
        )
        total_output = []
        for i in range(len(generated_text)):
            output = self.model.text_tokenizer.decode(generated_text[i])
            output = [x for x in output.split(' ') if not x.startswith('<')]
            out_label = output.index('GPT:')
            output = ' '.join(output[out_label + 1:])
            total_output.append(output)

        return total_output