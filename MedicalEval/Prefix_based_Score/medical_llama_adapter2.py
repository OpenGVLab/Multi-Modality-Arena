import os
import json
import pandas
import random
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.special import softmax
import requests
import os
import cv2
import importlib
import clip
import torch
from torchvision import transforms

from PIL import Image
from io import BytesIO
import pathlib

models_mae_path = './LLaMA-Adapter-v2/models_mae.py'
spec = importlib.util.spec_from_file_location('models_mae', models_mae_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
mae_vit_base_patch16 = module.mae_vit_base_patch16
model_ckpt_path = os.path.join(os.path.dirname(pathlib.Path(__file__).parent.absolute()), 'VLP_web_data', 'llama_checkpoints', 'llama_adapter_v2_0518.pth')
 
 

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
image_transform = transforms.Compose(
    [
        transforms.Resize(
            224, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)

class LLamaAdapterV2():
    def __init__(self, device="cuda"):
        generator = mae_vit_base_patch16()
        ckpt = torch.load(model_ckpt_path, map_location='cpu')
        ckpt_model = ckpt['model']
        msg = generator.load_state_dict(ckpt_model, strict=False)
        self.dtype = torch.float16
        self.device = device
        self.img_transform = image_transform
        self.generator = generator
        self.generator = self.generator.to(self.device, dtype=self.dtype)

    def generate(self, img, prompts,start_loc,lang_diff, max_gen_len=256, temperature=0.1, top_p=0.75):
        # img = Image.open(image_path).convert('RGB')
        img = np.array(img)[:, :, ::-1]
        img = Image.fromarray(np.uint8(img))
        imgs = [img]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        output = self.generator.forward_lm(imgs, prompts,start_loc,lang_diff,  max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
        return output


def load_candidates_medical(data):
    a,b,c,d = data.get('option_A'), data.get('option_B'), data.get('option_C'), data.get('option_D')
    answer_list = [a, b]
    if c is not None:
        answer_list.append(c)
    if d is not None:
        answer_list.append(d)
    return answer_list


def load_prompt(question, idx=4):
    prompts = ["{}".format(question),
               "{} Answer:".format(question),
               "{} The answer is".format(question),
               "Question: {} Answer:".format(question),
               "Question: {} The answer is".format(question)
               ]
    return prompts[idx]


def bytes2PIL(bytes_img):
    '''Transform bytes image to PIL.
    Args:
        bytes_img: Bytes image.
    '''
    pil_img = Image.open(BytesIO(bytes_img)).convert("RGB")
    return pil_img

     
def get_all_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    return file_list


def get_all_dirs(directory):
    all_files = get_all_files(directory)
    return list(set([os.path.dirname(f) for f in all_files ]))
 


@torch.no_grad()
def test(model, dataset=None, model_type='llama_adapter2', prompt_idx=4, save_path=''):
    with open(dataset) as f:
        data_all = json.load(f)
    cnt = 0
    correct = 0
    
    res = []
    for data in data_all:
        cnt += 1
        question = data['question']
        candidates = load_candidates_medical(data)
        answer = data['gt_answer']
        img_path = data['image_path']  
        
        
        prefix = load_prompt(question, prompt_idx)
        prefix_tokens = model.generator.tokenizer.encode(prefix,bos=True, eos=False)
        start_loc = len(prefix_tokens)
        
        candidate_scores = []  # pred scores of candidates
        raw_image = Image.open(img_path).convert("RGB")
        
        for candidate in candidates:
            prompts = [prefix + " {}.".format(candidate)]
            prompt_tokens = [model.generator.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
            
           
            lang_t =  len(model.generator.tokenizer.encode(prompts[0], bos=True, eos=False))
            lang_t1 = len(model.generator.tokenizer.encode(prefix, bos=True, eos=False))
            lang_diff = lang_t - lang_t1
            
            loss = model.generate(raw_image, prompt_tokens,start_loc, lang_diff)
            candidate_scores.append(loss.item())
        data['confidence'] =  str(candidate_scores)
        candidate_scores = softmax(np.reciprocal(candidate_scores))
        pred = candidates[np.argmax(candidate_scores)]
        print(candidates, candidate_scores)
        data['model_pred'] = pred
        
        data['is_correct'] = 'yes' if pred == answer else 'no'
        if pred == answer:
            correct += 1
        res.append(data)
        
    acc = correct / cnt
    print("Accuracy: ", acc)
    final_res = {'model_name': model_type, 'dataset_name': dataset, 'correct_precentage': acc, 'pred_dict': res}
    
    with open('{}/{}.json'.format(save_path, dataset.replace('/', '_')), 'w') as f:
        json.dump(final_res, f, indent=4, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    parser.add_argument("--dataset_path", type=str, default='/path/to/datset')
    parser.add_argument("--answer_path", type=str, default="output_res")
    args = parser.parse_args()
    return args

def run(args):
    model_type = 'llama_adapter2'
    vqa_model = LLamaAdapterV2(device=torch.device("cuda")) 
    
    answer_path = f'{args.answer_path}/{model_type}'
    os.makedirs(answer_path, exist_ok=True)
    sub_dataset = args.dataset_path
    test(vqa_model, dataset=sub_dataset, model_type=model_type, prompt_idx=4, save_path=answer_path)

if __name__ == "__main__":
    args = parse_args()
    run(args)

