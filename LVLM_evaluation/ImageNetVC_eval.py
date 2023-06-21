import os
import json
import pandas
import random
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from scipy.special import softmax

import torch
from .models import get_model


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


def get_idx2name_dict():
    names = {}
    with open(f"{args.dataset_path}/ImageNet_mapping.txt", "r") as f:
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
def test(model, subset='color', model_name='synthesis', prompt_idx=0, answer_path='./answers'):
    cnt = 0
    correct = 0
    df = pandas.read_csv(f'{args.dataset_path}/{subset}.csv', header=0)
    ImageNet_path = f'{args.dataset_path}/images/'
    name2idx = get_name2idx_dict()
    results = []
    id2scores = {}
    for i in tqdm(range(len(df))):
        if i > 0:
            break
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
        prefix_tokens = model.t5_tokenizer(prefix, return_tensors="pt", truncation=True, max_length=512)
        start_loc = prefix_tokens.input_ids.size(1)
        img_dir = ImageNet_path + name2idx[category]
        img_list = os.listdir(img_dir)
        for img in img_list:
            if len(img) < 15:
                img_list.remove(img)
        random.seed(2022)
        selected_img_names = random.sample(img_list, 10)
        img_scores = []
        for img_name in selected_img_names:
            candidate_scores = []  # pred scores of candidates
            img_path = os.path.join(img_dir, img_name)
            for candidate in candidates:
                prompt = prefix + " {}.".format(candidate)
                loss = model.forward_lm(img_path, prompt, start_loc)
                candidate_scores.append(loss.item())
            candidate_scores = softmax(np.reciprocal(candidate_scores))
            img_scores.append(candidate_scores)
        mean_scores = np.mean(img_scores, axis=0)
        pred = candidates[np.argmax(mean_scores)]
        result_line = {"question_id": cnt, "answer": pred}
        id2scores[cnt] = mean_scores
        results.append(result_line)
        if pred == answer:
            correct += 1
    
    write_json(results, subset, model_name, prompt_idx=prompt_idx, path=answer_path)
    pickle.dump(id2scores, open(f"{answer_path}/{model_name}/{subset}/prompt{prompt_idx}.pkl", "wb"))
    print("Accuracy: ", correct / cnt)
    return correct / cnt


def get_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model_name", type=str, default="LLaMA-Adapter-v2")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--dataset_path", type=str, default="./datasets/ImageNetVC")
    parser.add_argument("--answer_path", type=str, default="./answers")
    args = parser.parse_args()
    return args


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    model = get_model(args.model_name, device=torch.device('cuda'))
    subset_list = ['color', 'shape', 'material', 'component', 'others']
    for subset in subset_list:
        print("Tested on the {} subset...".format(subset))
        results = []
        for idx in range(0, 1):  # prompt idx
            acc = test(model, subset=subset, model_name=args.model_name, prompt_idx=idx, answer_path=args.answer_path)
            results.append(acc)
        with open(f"{args.answer_path}/{args.model_name}/{subset}/results.txt", "w") as f:
            for idx, acc in enumerate(results):
                f.write("Accuracy for prompt{}: {} \n".format(idx, acc))
            avg = np.mean(results)
            std = np.std(results, ddof=1)
            f.write("Mean result: {}, Std result: {}".format(100 * avg, 100 * std))


if __name__ == "__main__":
    args = get_args()
    main(args)
