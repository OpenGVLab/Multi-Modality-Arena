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
import torch
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION
from minigpt4.models import *
from minigpt4.processors import *

from PIL import Image
from io import BytesIO

CFG_PATH = './minigpt4/minigpt4_eval.yaml'


class MiniGPT4():
    def __init__(self, device=None):
        cfg = Config(CFG_PATH)
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cpu')
        vis_processor_cfg = cfg.preprocess_cfg.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.model, self.vis_processor = model, vis_processor
        self.model.llama_model = self.model.llama_model.float().to('cpu')
        self.chat = Chat(model, vis_processor, device='cpu')
        self.dtype = torch.float16
        self.device = device
        self.chat.device = device
        self.model = self.model.to(self.device, dtype=self.dtype)
        self.chat.move_stopping_criteria_device(self.device, dtype=self.dtype)

    
    def generate(self, image_path, question):
        chat_state = CONV_VISION.copy()
        img_list = []
        imgs = [Image.open(image_path).convert('RGB')]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)   
        self.chat.upload_img(imgs, chat_state, img_list)
        self.chat.ask(question, chat_state)
        llm_message = self.chat.answer(conv=chat_state, img_list=img_list)[0]

        return llm_message


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

 


@torch.no_grad()
def test(model, dataset=None, model_type='minigpt4', prompt_idx=4, save_path='/mnt/petrelfs/luquanfeng/Multi-Modality-Arena/tiny_lvlm_evaluation/final_answer_extra/minigpt4'):
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
        prefix_tokens = model.model.tokenizer(prefix, return_tensors="pt", truncation=True, max_length=512)
        start_loc = prefix_tokens.input_ids.size(1)
        
        candidate_scores = []  # pred scores of candidates
        raw_image = Image.open(img_path).convert("RGB")
        image = model.vis_processor(raw_image).unsqueeze(0).to(model.device)
        
        for candidate in candidates:
            prompt = prefix + " {}.".format(candidate)
            lang_t = model.model.tokenizer(prompt, return_tensors="pt")["input_ids"]
            lang_t1 = model.model.tokenizer(prefix, return_tensors="pt")["input_ids"]
            lang_diff = lang_t.shape[1] - lang_t1.shape[1]
            
            outputs = model.model.forward_lm({"image": image, "prompt": prompt, "start_loc": start_loc, "lang_diff":lang_diff})
            loss = outputs["loss"]
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
    model_type = 'minigpt4'
    vqa_model = MiniGPT4(device=torch.device("cuda")) 
    
    answer_path = f'{args.answer_path}/{model_type}'
    os.makedirs(answer_path, exist_ok=True)
    sub_dataset = args.dataset_path
    test(vqa_model, dataset=sub_dataset, model_type=model_type, prompt_idx=4, save_path=answer_path)

if __name__ == "__main__":
    args = parse_args()
    run(args)


