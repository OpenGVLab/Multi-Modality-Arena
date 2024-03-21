import os
import json
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.special import softmax

import torch
from mplug_owl.processing_mplug_owl import MplugOwlProcessor, MplugOwlImageProcessor
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
from transformers import AutoTokenizer
from PIL import Image

from io import BytesIO

prompt_template = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <image>\nHuman: {}\nAI:"

generate_kwargs = {
    'do_sample': True,
    'top_k': 5,
    'max_length': 512
}

class MplugOwl():
    def __init__(self, device=None):
        model_path='MAGAer13/mplug-owl-llama-7b'
        self.model = MplugOwlForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float32)
        self.image_processor = MplugOwlImageProcessor.from_pretrained(model_path)
        # self.tokenizer = MplugOwlTokenizer.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = MplugOwlProcessor(self.image_processor, self.tokenizer)
        self.model.eval()  
        self.device = device
        self.dtype = torch.float16
        self.model.to(device=self.device, dtype=self.dtype)
    @torch.no_grad()
    def generate(self, image, question):
        prompts = [prompt_template.format(question)]
        image = Image.open(image).convert('RGB')
        inputs = self.processor(text=prompts, images=[image], return_tensors='pt')
        inputs = {k: v.to(self.device, dtype=self.dtype) if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            res = self.model.generate(**inputs, **generate_kwargs)
        generated_text = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)

        return generated_text        


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



def get_idx2name_dict():
    names = {}
    with open("./files/ImageNet_mapping.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            idx_name = line.split(' ')[0]
            cate_name = ' '.join(line.split(' ')[1:]).split(',')[0]
            names[idx_name] = cate_name
    return names


def get_name2idx_dict():
    dic = get_idx2name_dict()
    dic_new = dict([val, key] for key, val in dic.items())
    return dic_new


@torch.no_grad()
def test(model, dataset=None, model_type='owl', prompt_idx=4, save_path=''):
    with open(dataset) as f:
        data_all = json.load(f)
    cnt = 0
    correct = 0
    
    if not  os.path.exists(save_path):
        os.makedirs(save_path)
    
    res = []
    random.seed(2022)
    for data in data_all:
        cnt += 1
        question = data['question']
        candidates = load_candidates_medical(data)
        answer = data['gt_answer']
        img_path = data['image_path']  
        prefix = load_prompt(question, prompt_idx)
        prefix_tokens = model.processor(text = prefix)
        start_loc = prefix_tokens.input_ids.size(1)    
        
        
        
        candidate_scores = []  # pred scores of candidates
        raw_image = Image.open(img_path).convert("RGB")
        # image = (model.image_processor.preprocess([raw_image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0))
        
        
        images = [raw_image]
        images = [model.image_processor(image, return_tensors='pt').pixel_values for image in images]
        images = torch.cat(images,dim=0).to(model.device, dtype=model.dtype) 
        image_size = images.size(0)
        
        for candidate in candidates:
            prompt = prefix + " {}.".format(candidate)
            prompts = [prompt for _ in range(image_size)]
            inputs = model.processor(text = prompts)
            inputs = {k:v.to(model.device) for k,v in inputs.items()}
            inputs["pixel_values"] = images
            lang_t = model.tokenizer(prompt, return_tensors="pt")["input_ids"].size(1)    
            lang_t1 = model.tokenizer(prefix, return_tensors="pt")["input_ids"].size(1)    
            lang_diff = lang_t - lang_t1
            start_loc = lang_t1
            #import pdb; pdb.set_trace()
            # TODO 这个start_loc 的计算可能有问题

            outputs = model.model.forward_lm(**inputs, start_loc=start_loc, lang_diff=lang_diff)
            candidate_scores.append(outputs)
        
        data['confidence'] =  str(candidate_scores)
        print('ori:', candidate_scores)
        candidate_scores = softmax(candidate_scores)
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
    model_type = 'mplugowl'
    vqa_model = MplugOwl(device=torch.device("cuda")) 
    
    answer_path = f'{args.answer_path}/{model_type}'
    os.makedirs(answer_path, exist_ok=True)
    sub_dataset = args.dataset_path
    test(vqa_model, dataset=sub_dataset, model_type=model_type, prompt_idx=4, save_path=answer_path)

if __name__ == "__main__":
    args = parse_args()
    run(args)