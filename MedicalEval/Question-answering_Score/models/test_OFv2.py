import torch
from torch import nn
from huggingface_hub import hf_hub_download
from PIL import Image

from open_flamingo import create_model_and_transforms


class OFv2(nn.Module):
    def __init__(self, version: str='3BI',
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ) -> None:
        super().__init__()
        if version == '3BI':
            model, image_processor, tokenizer = create_model_and_transforms(
                clip_vision_encoder_path="ViT-L-14",
                clip_vision_encoder_pretrained="openai",
                lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",
                tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",
                cross_attn_every_n_layers=1
            )
            # grab model checkpoint from huggingface hub
            checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct", "checkpoint.pt")
        elif version == '4BI':
            model, image_processor, tokenizer = create_model_and_transforms(
                clip_vision_encoder_path="ViT-L-14",
                clip_vision_encoder_pretrained="openai",
                lang_encoder_path="togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
                tokenizer_path="togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
                cross_attn_every_n_layers=2
            )
            checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct", "checkpoint.pt")
        else:
            raise ValueError(f'OpenFlamingo v2 {version} NOT supported yet!')
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
        tokenizer.padding_side = "left" # For generation padding tokens should be on the left
        self.model = model.eval()
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        self.move_to_device(device)

    def move_to_device(self, device=None):
        if device is not None and 'cuda' in device.type:
            self.dtype = torch.float16
            self.device = device
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
        self.model = self.model.to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens: int=20, *args, **kwargs):
        if type(image_list[0]) is not str:
            images = [Image.fromarray(x) for x in image_list]
        else:
            images = [Image.open(img).convert('RGB') for img in image_list]
        vision_x = [self.image_processor(x).unsqueeze(0).unsqueeze(0).unsqueeze(0) for x in images]
        vision_x = torch.cat(vision_x, dim=0).to(self.device, dtype=self.dtype)
        prompt_template = kwargs.get('prompt_template', 'OFv2_vqa')
        if prompt_template == 'OFv2_vqa':
            prompts = [f"<image>Question: {x} Short answer:" for x in question_list]
        else:
            prompts = [f"<image>{x}" for x in question_list]
        lang_x = self.tokenizer(
            prompts,
            return_tensors="pt", padding=True,
        ).to(self.device)
        generated_text = self.model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"],
            attention_mask=lang_x["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=3,
        )
        outputs = self.tokenizer.batch_decode(generated_text, skip_special_tokens=True)
        results = [y[len(x)-len('<image>'):].strip() for x, y in zip(prompts, outputs)]
        return results
