import torch
import transformers

from .otter_image.modeling_otter import OtterForConditionalGeneration
from . import get_image


def get_formatted_prompt(prompt: str, in_context_prompts: list = []) -> str:
    in_context_string = ""
    for in_context_prompt, in_context_answer in in_context_prompts:
        in_context_string += f"<image>User: {in_context_prompt} GPT:<answer> {in_context_answer}<|endofchunk|>"
    return f"{in_context_string}<image>User: {prompt} GPT:<answer>"


class TestOtterImage:
    def __init__(self, device=None) -> None:
        model = OtterForConditionalGeneration.from_pretrained("luodian/OTTER-9B-LA-InContext", device_map="auto")
        model.text_tokenizer.padding_side = "left"
        image_processor = transformers.CLIPImageProcessor()
        model.eval()
        self.model = model
        self.image_processor = image_processor

    def move_to_device(self, device):
        pass

    @torch.no_grad()
    def generate(self, raw_image, question, max_new_tokens=256):
        raw_image = get_image(raw_image)
        vision_x = self.image_processor.preprocess([raw_image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        lang_x = self.model.text_tokenizer(
            [
                get_formatted_prompt(question, []),
            ],
            return_tensors="pt",
        )
        bad_words_id = self.model.text_tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
        generated_text = self.model.generate(
            vision_x=vision_x.to(self.model.device),
            lang_x=lang_x["input_ids"].to(self.model.device),
            attention_mask=lang_x["attention_mask"].to(self.model.device),
            max_new_tokens=max_new_tokens,
            bad_words_ids=bad_words_id,
            do_sample=False,
            temperature=0,
            # num_beams=1,
            # no_repeat_ngram_size=3,
        )
        parsed_output = (
            self.model.text_tokenizer.decode(generated_text[0])
            .split("<answer>")[-1]
            .lstrip()
            .rstrip()
            .split("<|endofchunk|>")[0]
            .lstrip()
            .rstrip()
            .lstrip('"')
            .rstrip('"')
        )
        
        return parsed_output

    @torch.no_grad()
    def pure_generate(self, raw_image, question, max_new_tokens=256):
        raw_image = get_image(raw_image)
        vision_x = self.image_processor.preprocess([raw_image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        lang_x = self.model.text_tokenizer([question], return_tensors="pt")
        bad_words_id = self.model.text_tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
        generated_text = self.model.generate(
            vision_x=vision_x.to(self.model.device),
            lang_x=lang_x["input_ids"].to(self.model.device),
            attention_mask=lang_x["attention_mask"].to(self.model.device),
            max_new_tokens=max_new_tokens,
            bad_words_ids=bad_words_id,
            do_sample=False,
            temperature=0,
            # num_beams=1,
            # no_repeat_ngram_size=3,
        )
        parsed_output = (
            self.model.text_tokenizer.decode(generated_text[0])
            .split("<answer>")[-1]
            .lstrip()
            .rstrip()
            .split("<|endofchunk|>")[0]
            .lstrip()
            .rstrip()
            .lstrip('"')
            .rstrip('"')
        )
        
        return parsed_output

    def batch_generate(self, image_list, question_list, max_new_tokens=256):
        imgs = [get_image(img) for img in image_list]
        imgs = [self.image_processor.preprocess([x], return_tensors="pt")["pixel_values"].unsqueeze(0) for x in imgs]
        vision_x = (torch.stack(imgs, dim=0))
        prompts = [get_formatted_prompt(question, []) for question in question_list]
        lang_x = self.model.text_tokenizer(prompts, return_tensors="pt", padding=True)
        bad_words_id = self.model.text_tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
        generated_text = self.model.generate(
            vision_x=vision_x.to(self.model.device),
            lang_x=lang_x["input_ids"].to(self.model.device),
            attention_mask=lang_x["attention_mask"].to(self.model.device),
            max_new_tokens=max_new_tokens,
            bad_words_ids=bad_words_id,
            do_sample=False,
            temperature=0,
            # num_beams=1,
            # no_repeat_ngram_size=3,
        )
        total_output = []
        for i in range(len(generated_text)):
            parsed_output = (
                self.model.text_tokenizer.decode(generated_text[i])
                .split("<answer>")[-1]
                .lstrip()
                .rstrip()
                .split("<|endofchunk|>")[0]
                .lstrip()
                .rstrip()
                .lstrip('"')
                .rstrip('"')
            )
            total_output.append(parsed_output)

        return total_output
