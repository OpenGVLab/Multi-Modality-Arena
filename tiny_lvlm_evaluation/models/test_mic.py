import torch
from .mic.instructblip import InstructBlipConfig, InstructBlipForConditionalGeneration, InstructBlipProcessor
from . import get_image

DTYPE = torch.float16


class TestMIC:
    def __init__(self, device=None) -> None:
        model_type="instructblip"
        model_ckpt="BleachNick/MMICL-Instructblip-T5-xl"
        processor_ckpt = "Salesforce/instructblip-flan-t5-xl"
        config = InstructBlipConfig.from_pretrained(model_ckpt)

        if 'instructblip' in model_type:
            model = InstructBlipForConditionalGeneration.from_pretrained(model_ckpt, config=config).to('cuda',dtype=DTYPE)

        image_palceholder="图"
        sp = [image_palceholder]+[f"<image{i}>" for i in range(20)]
        processor = InstructBlipProcessor.from_pretrained(
            processor_ckpt
        )
        sp = sp + processor.tokenizer.additional_special_tokens[len(sp):]
        processor.tokenizer.add_special_tokens({'additional_special_tokens':sp})
        if model.qformer.embeddings.word_embeddings.weight.shape[0] != len(processor.qformer_tokenizer):
            model.qformer.resize_token_embeddings(len(processor.qformer_tokenizer))
        self.replace_token="".join(32*[image_palceholder])

        self.model = model
        self.processor = processor

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=64):
        image = [get_image(image)]
        prompt = f'Use the image 0: <image0>{self.replace_token} as a visual aid to help you answer the question. Question: {question} Answer:'

        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].to(DTYPE)
        inputs['img_mask'] = torch.tensor([[1]])
        inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)

        inputs = inputs.to('cuda:0')
        with torch.cuda.amp.autocast():
            outputs = self.model.generate(
                pixel_values = inputs['pixel_values'],
                input_ids = inputs['input_ids'],
                attention_mask = inputs['attention_mask'],
                img_mask = inputs['img_mask'],
                do_sample=False,
                max_length=max_new_tokens,
                min_length=1,
                set_min_padding_size =False,
            )
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return generated_text

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=64):
        output = [self.generate(image, question, max_new_tokens) for image, question in zip(image_list, question_list)]
        return output
