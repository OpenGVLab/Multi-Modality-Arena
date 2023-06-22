import os
import importlib
from types import MethodType
from gradio_client import Client

import clip
import torch

from . import get_BGR_image, DATA_DIR

MAX_SEQ_LEN, MAX_BATCH_SIZE = 256, 64
model_ckpt_path = f'{DATA_DIR}/llama_checkpoints/llama_adapter_v2_0518.pth'

# # NOTE: please use customized clip and timm library

# models_mae_path = 'models/llama_adapter_v2/models_mae.py'
# spec = importlib.util.spec_from_file_location('models_mae', models_mae_path)
# module = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(module)
# mae_vit_base_patch16 = module.mae_vit_base_patch16


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@torch.no_grad()
def forward_lm(
    self,
    imgs,
    prompt_prefix,
    start_loc,
    max_gen_len: int = 64,
    ):

    bsz = len(imgs)
    params = self.llma.params
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

    with torch.cuda.amp.autocast():
        visual_feats = self.encode_image(imgs)
    visual_feats = self.clip_proj_norm(self.clip_proj(visual_feats))
    query = self.visual_query.weight.unsqueeze(0).repeat(bsz, 1, 1)
    query = torch.cat([query, visual_feats], dim=1)
    for block in self.blocks:
        query = block(query)
    query = query[:, :10, :]
    query = self.prefix_projector(query)
    query = self.prefix_projector_norm(query)
    visual_tokens = query

    if isinstance(prompt_prefix[0], str):
        prompt_prefix = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompt_prefix]

    min_prompt_size = min([len(t) for t in prompt_prefix])
    max_prompt_size = max([len(t) for t in prompt_prefix])

    total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

    tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).long().to(visual_feats.device)
    for k, t in enumerate(prompt_prefix):
        tokens[k, : len(t)] = torch.tensor(t).long().to(visual_feats.device)

    start_pos = min_prompt_size
    prev_pos = 0
    logits = self.forward_inference(visual_tokens, tokens[:, prev_pos:start_pos], prev_pos)
    labels = tokens[:,0:start_pos]
    labels = labels.detach().cpu().numpy()
    labels[0, :start_loc] = -100
    labels[0, start_loc + 1:] = -100
    labels = torch.from_numpy(labels)
    labels = labels.to(visual_feats.device)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = loss_fct(shift_logits.view(-1,32000), shift_labels.view(-1))
    return loss


class TestLLamaAdapterV2_web:
    def __init__(self, device=None) -> None:
        self.model = Client("http://106.14.127.192:8088/")
        self.max_length = 64
        self.temperature = 0.1
        self.top_p = 0.75

        if device is not None:
            self.move_to_device(device)
    
    def move_to_device(self, device):
        pass

    @torch.no_grad()
    def generate(self, image, question: str):
        image = get_BGR_image(image)
        image_name = '.llama_adapter_v2_inference.png'
        image.save(image_name)
        output = self.model.predict(image_name, question, self.max_length, self.temperature, self.top_p, fn_index=1)
        
        return output


class TestLLamaAdapterV2:
    def __init__(self, device=None) -> None:
        _, img_transform = clip.load("ViT-L/14")
        from .llama_adapter_v2.models_mae import mae_vit_base_patch16
        generator = mae_vit_base_patch16()
        ckpt = torch.load(model_ckpt_path, map_location='cpu')
        ckpt_model = ckpt['model']
        msg = generator.load_state_dict(ckpt_model, strict=False)

        self.img_transform = img_transform
        self.generator = generator
        self.generator.forward_lm = MethodType(forward_lm, self.generator)

        if device is not None:
            self.move_to_device(device)

    def move_to_device(self, device):
        if type(device) is str and 'cuda' in device:
            self.dtype = torch.float16
            self.device = device
        elif type(device) is torch.device and 'cuda' in device.type:
            self.dtype = torch.float16
            self.device = device
        else:
            self.device = 'cpu'
            self.dtype = torch.float32
        self.generator = self.generator.to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=256, temperature=0.1, top_p=0.75):
        imgs = [get_BGR_image(image)]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device, dtype=self.dtype)

        prompts = [PROMPT_DICT['prompt_no_input'].format_map({'instruction': question})]
        prompts = [self.generator.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        # with torch.cuda.amp.autocast():
        results = self.generator.generate(imgs, prompts, max_gen_len=max_new_tokens, temperature=temperature, top_p=top_p)
        result = results[0].strip()

        return result
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=256, temperature=0.1, top_p=0.75):
        imgs = [get_BGR_image(img) for img in image_list]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device, dtype=self.dtype)
        prompts = [PROMPT_DICT['prompt_no_input'].format_map({'instruction': question}) for question in question_list]
        results = self.generator.generate(imgs, prompts, max_gen_len=max_new_tokens, temperature=temperature, top_p=top_p)
        results = [result.strip() for result in results]

        return results

    def forward_lm(self, image, prompt, start_loc):
        imgs = [get_BGR_image(image)]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device, dtype=self.dtype)
        prompts = [PROMPT_DICT['prompt_no_input'].format_map({'instruction': prompt})]
        prompts = [self.generator.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        loss = self.model.forward_lm(image, prompt, start_loc)
        return loss