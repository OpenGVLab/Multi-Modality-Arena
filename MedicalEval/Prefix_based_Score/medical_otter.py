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
from otter.modeling_otter import OtterForConditionalGeneration
import cv2
from PIL import Image
from torch.nn import CrossEntropyLoss
import pathlib


CKPT_PATH=os.path.join(os.path.dirname(pathlib.Path(__file__).parent.absolute()), 'VLP_web_data', 'otter-9b-hf')

from io import BytesIO

class Otter():
    def __init__(self,  device="cuda"):
        model_path=CKPT_PATH
        self.model = OtterForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = self.model.text_tokenizer
        self.image_processor = CLIPImageProcessor()
        self.tokenizer.padding_side = "left"
        self.dtype = torch.float16
        self.device = device             
        self.model = self.model.to(self.device, dtype=self.dtype)
        #self.model.vision_encoder = self.model.vision_encoder.to('cpu', dtype=torch.float32)


    def generate(self, image, question):
        img = cv2.imread(image)
        image = Image.fromarray(img) 
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
def test(model, dataset=None, model_type='otter', prompt_idx=4, save_path=''):
    with open(dataset) as f:
        data_all = json.load(f)
    cnt = 0
    correct = 0
    
    res = []
    random.seed(2022)
    for data in data_all:
        cnt += 1
        question = data['question']
        candidates = load_candidates_medical(data)
        answer = data['gt_answer']
        img_path = data['image_path']  
        prefix = load_prompt(question, prompt_idx)
        prefix_tokens = model.tokenizer(prefix)
        start_loc = len(prefix_tokens.input_ids)
        candidate_scores = []  # pred scores of candidates
        raw_image = Image.open(img_path).convert("RGB")
        image = (model.image_processor.preprocess([raw_image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0))
    
        for candidate in candidates:
            prompt = prefix + " {}.".format(candidate)
            prompt_tokens = model.tokenizer(prompt, return_tensors="pt")  # prefix(不带选项的) tokens 的长度
            lang_t = prompt_tokens["input_ids"]
            
            prefix_tokens = model.tokenizer(prefix, return_tensors="pt")  
            lang_t1 = prefix_tokens["input_ids"]
            lang_diff = lang_t.shape[1] - lang_t1.shape[1]
            
            
            attenion_mask = prompt_tokens["attention_mask"]                     
            outputs = model.model.forward(vision_x = image.to(model.model.device, dtype=model.dtype), lang_x = lang_t.to(model.model.device), attention_mask = attenion_mask.to(model.model.device, dtype=model.dtype))
            logits = outputs.logits                    
            targets = prompt_tokens["input_ids"].to(model.model.device)
            targets[0, :start_loc] = -100
            targets[0, start_loc + lang_diff:] = -100

            l_s = logits.size(1)
            t_s = targets.size(1)
            dif = t_s - l_s + 1
            shift_labels = targets[..., dif:].contiguous()
            shift_logits = logits[...,:-1,:].contiguous()
            loss_fct = CrossEntropyLoss(reduction="mean")                
            loss = loss_fct(shift_logits.view(-1, 32004), shift_labels.view(-1))                
            loss = loss.cpu()
        
            
            candidate_scores.append(loss.float())
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
    model_type = 'otter'
    vqa_model = Otter(device=torch.device("cuda")) 
    
    answer_path = f'{args.answer_path}/{model_type}'
    os.makedirs(answer_path, exist_ok=True)
    sub_dataset = args.dataset_path
    test(vqa_model, dataset=sub_dataset, model_type=model_type, prompt_idx=4, save_path=answer_path)

if __name__ == "__main__":
    args = parse_args()
    run(args)

