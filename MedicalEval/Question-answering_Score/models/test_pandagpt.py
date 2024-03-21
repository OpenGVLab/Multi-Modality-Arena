import torch
from .pandagpt import OpenLLAMAPEFTModel
from . import DATA_DIR


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


class TestPandaGPT:
    def __init__(self, device=None):
        args = {
            'model': 'openllama_peft',
            'imagebind_ckpt_path': f'{DATA_DIR}/PandaGPT/imagebind_ckpt',
            'vicuna_ckpt_path': f'{DATA_DIR}/PandaGPT/vicuna_ckpt/7b_v0',
            'delta_ckpt_path': f'{DATA_DIR}/PandaGPT/pandagpt_ckpt/7b/pytorch_model.pt',
            'stage': 2,
            'max_tgt_len': 128,
            'lora_r': 32,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
        }
        model = OpenLLAMAPEFTModel(**args)
        delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
        model.load_state_dict(delta_ckpt, strict=False)
        model = model.eval().half().cuda()
        self.model = model

    def move_to_device(self, device):
        pass

    def generate(self, image, question, max_new_tokens=256):
        response = self.model.generate({
            'prompt': question,
            'image_paths': [image],
            'audio_paths': [],
            'video_paths': [],
            'thermal_paths': [],
            'top_p': 0.01,
            'temperature': 1.0,
            'max_tgt_len': max_new_tokens,
            'modality_embeds': []
        })
        return response
    
    def batch_generate(self, image_list, question_list, max_new_tokens=256):
        outputs = [self.generate(image, question, max_new_tokens) for image, question in zip(image_list, question_list)]
        return outputs