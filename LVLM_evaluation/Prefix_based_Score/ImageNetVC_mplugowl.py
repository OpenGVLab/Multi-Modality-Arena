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
from mplug_owl.processing_mplug_owl import MplugOwlProcessor, MplugOwlImageProcessor
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
from transformers import AutoTokenizer
import cv2
from PIL import Image

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



def write_json(results, subset='color', model_type="rank", path='./results', prompt_idx=0):
    type_path = os.path.join(path, model_type)
    if not os.path.exists(type_path):
        os.mkdir(type_path)
    subset_path = os.path.join(type_path, subset)
    if not os.path.exists(subset_path):
        os.mkdir(subset_path)
    save_path = os.path.join(subset_path, "prompt{}".format(prompt_idx) + '.json')
    with open(save_path, "w") as f:
        json.dump(results, f)


def load_candidates(subset='color'):
    if subset == 'color':
        candidates = ['brown', 'black', 'white', 'yellow', 'green', 'gray', 'red', 'orange', 'blue', 'silver', 'pink']
    elif subset == 'shape':
        candidates = ['round', 'rectangle', 'triangle', 'square', 'oval', 'curved', 'cylinder', 'straight',
                      'cone', 'curly', 'heart', 'star']
    elif subset == 'material':
        candidates = ['metal', 'wood', 'plastic', 'cotton', 'glass', 'fabric', 'stone', 'rubber', 'ceramic',
                      'cloth', 'leather', 'flour', 'paper', 'clay', 'wax', 'concrete']
    elif subset == 'component':
        candidates = ['yes', 'no']
    elif subset == 'others_yes':
        candidates = ['yes', 'no']
    elif subset == 'others_number':
        candidates = ['2', '4', '6', '1', '8', '3', '5']
    elif subset == 'others_other':
        candidates = ['long', 'small', 'short', 'large', 'forest', 'water', 'ocean', 'big', 'tree', 'ground', 'tall',
                      'wild', 'outside', 'thin', 'head', 'thick', 'circle', 'brown', 'soft', 'land', 'neck', 'rough',
                      'chest', 'smooth', 'fur', 'hard', 'top', 'plants', 'black', 'metal', 'books', 'vertical', 'lake',
                      'grass', 'road', 'sky', 'front', 'kitchen', 'feathers', 'stripes', 'baby', 'hair', 'feet',
                      'mouth', 'female', 'table']
    else:
        print("Subset does not exist!")
        candidates = []
    return candidates



def load_prompt(question, idx=0):
    prompts = ["{}".format(question),
               "{} Answer:".format(question),
               "{} The answer is".format(question),
               "Question: {} Answer:".format(question),
               "Question: {} The answer is".format(question),
               ]
    return prompts[idx]


def get_score(subset='color', save_path='./results/opt-2.7b-search/'):
    correct = 0
    df = pandas.read_csv('./files/dataset/{}.csv'.format(subset), header=0)
    candidates = load_candidates(subset)
    path = save_path + "{}.p".format(subset)
    id2score = pickle.load(open(path, "rb"))
    for idx in tqdm(range(len(df))):
        answer = df['answer'][idx]
        score = id2score[idx + 1]
        pred = candidates[np.argmax(score)]
        if pred == answer:
            correct += 1
    print(correct / len(df))


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
def test(model, subset='color', model_type='synthesis', prompt_idx=0):
    image_type = "rank"    
    cnt = 0
    correct = 0
    df = pandas.read_csv('./files/dataset/{}.csv'.format(subset), header=0)
    ImageNet_path = './datasets/{}/'.format(image_type)  # the path that contains the images
    name2idx = get_name2idx_dict()
    results = []
    id2scores = {}
    for i in tqdm(range(len(df))):
        cnt += 1
        category = df['category'][i]
        question = df['question'][i]
        answer = str(df['answer'][i]).lower()
        if subset == 'others':
            if answer in ['yes', 'no']:
                sub_subset = 'others_yes'
            elif answer in ['2', '4', '6', '1', '8', '3', '5']:
                sub_subset = 'others_number'
            else:
                sub_subset = 'others_other'
            candidates = load_candidates(sub_subset)
        else:
            candidates = load_candidates(subset)
        prefix = load_prompt(question, prompt_idx)
        prefix_tokens = model.processor(text = prefix)
        start_loc = prefix_tokens.input_ids.size(1)        
        img_dir = ImageNet_path + name2idx[category]
        img_list = os.listdir(img_dir)
        candidate_scores = []

        images = [Image.open(os.path.join(img_dir,img_path)).convert('RGB') for img_path in img_list]
        images = [model.image_processor(image, return_tensors='pt').pixel_values for image in images]
        images = torch.cat(images,dim=0).to(model.device, dtype=model.dtype) 
        image_size = images.size(0)

        for candidate in candidates:
            prompt = prefix + " {}.".format(candidate)
            prompts = [prompt for i in range(image_size)]
            inputs = model.processor(text = prompts)
            inputs = {k:v.to(model.device) for k,v in inputs.items()}
            inputs["pixel_values"] = images
            outputs = model.model.forward_lm(**inputs, start_loc=start_loc)
            candidate_scores.append(outputs)
        candidate_scores = np.concatenate(candidate_scores, axis=1)
        img_scores = softmax(candidate_scores,axis=1) 
        mean_scores = np.mean(img_scores, axis=0)
        pred = candidates[np.argmax(mean_scores)]
        result_line = {"question_id": cnt, "answer": pred}
        id2scores[cnt] = mean_scores
        results.append(result_line)
        if pred == answer:
            correct += 1
    dir_path = "./results/{}/{}".format(model_type, subset)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)                  
    write_json(results, subset, model_type, prompt_idx=prompt_idx)
    pickle.dump(id2scores, open("./results/{}/{}/prompt{}.p".format(model_type, subset, prompt_idx), "wb"))
    print("Accuracy: ", correct / cnt)
    return correct / cnt


def run():
    # rank, search, synthesis, train
    device_id = 0
    model_type = 'mplugowl'    
    subset_list = ['color', 'shape', 'material', 'component', 'others']       
    model = MplugOwl(device=torch.device("cuda:{}".format(device_id)))                                                        
    for subset in subset_list:
        print("Tested on the {} subset...".format(subset))
        results = []
        for idx in range(0, 5):  # prompt idx
            acc = test(model, subset=subset, model_type=model_type, prompt_idx=idx)
            results.append(acc)
        with open("./results/{}/{}/results.txt".format(model_type, subset), "w") as f:
            for idx, acc in enumerate(results):
                f.write("Accuracy for prompt{}: {} \n".format(idx, acc))
            avg = np.mean(results)
            std = np.std(results, ddof=1)
            f.write("Mean result: {}, Std result: {}".format(100 * avg, 100 * std))


if __name__ == "__main__":
    run()


