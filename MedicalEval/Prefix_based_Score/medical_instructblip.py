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
from transformers import CLIPImageProcessor
from instruct_blip.models import load_model_and_preprocess
from instruct_blip.models.eva_vit import convert_weights_to_fp16
import cv2

from PIL import Image
from io import BytesIO

    
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


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
def test(model, vis_processors, dataset=None, model_type='instructblip', prompt_idx=4, save_path=''):
    data_all = json.load(open(dataset))
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
        model.llm_tokenizer.padding_side = "right"
        model.llm_tokenizer.truncation_side = 'left'
        
        prefix_tokens = model.llm_tokenizer(prefix,return_tensors="pt",padding="longest",truncation=True,max_length=128)
        start_loc = prefix_tokens['input_ids'].size(1) #llm_tokens['input_ids']
        
        candidate_scores = []  # pred scores of candidates
        raw_image = Image.open(img_path).convert("RGB")
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        
        for candidate in candidates:
            prompt = prefix + " {}.".format(candidate)
            
            lang_t = model.llm_tokenizer(prompt,return_tensors="pt",padding="longest",truncation=True,max_length=128)["input_ids"].size(1)
            lang_t1 = model.llm_tokenizer(prefix,return_tensors="pt",padding="longest",truncation=True,max_length=128)["input_ids"].size(1)
            lang_diff = lang_t - lang_t1
            
            
            outputs = model.forward_lm({"image": image, "prompt": prompt, "start_loc": start_loc, "lang_diff":lang_diff})
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
    model_type = 'instructblip'
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)   

    answer_path = f'{args.answer_path}/{model_type}'
    os.makedirs(answer_path, exist_ok=True)
    sub_dataset = args.dataset_path
    test(model, vis_processors, dataset=sub_dataset, model_type=model_type, prompt_idx=4, save_path=answer_path)

if __name__ == "__main__":
    args = parse_args()
    run(args)

