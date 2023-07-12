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
from lavis.models import load_model_and_preprocess
import torch

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"


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
def test(model, vis_processors, subset='color', model_type='synthesis', prompt_idx=0):
    cnt = 0
    correct = 0
    image_type = "rank"
    df = pandas.read_csv('./files/dataset/{}.csv'.format(subset), header=0)
    ImageNet_path = './datasets/{}/'.format(image_type)  # the path that contains the images
    name2idx = get_name2idx_dict()
    results = []
    id2scores = {}
    for i in tqdm(range(len(df))):
        if i>0:
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
            raw_image = Image.open(img_path).convert('RGB')
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            for candidate in candidates:
                prompt = prefix + " {}.".format(candidate)
                outputs = model.forward_lm({"image": image, "prompt": prompt, "start_loc": start_loc})
                loss = outputs["loss"]
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
    dir_path = "./results/{}/{}".format(model_type, subset)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)                
    write_json(results, subset, model_type, prompt_idx=prompt_idx)
    pickle.dump(id2scores, open("./results/{}/{}/prompt{}.p".format(model_type, subset, prompt_idx), "wb"))
    print("Accuracy: ", correct / cnt)
    return correct / cnt


def run():
    # rank, search, synthesis, train
    model_type = 'blip2'
    subset_list = ['color', 'shape', 'material', 'component', 'others']
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl",
                                                         is_eval=True, device=device)
    for subset in subset_list:
        print("Tested on the {} subset...".format(subset))
        results = []
        for idx in range(0, 5):  # prompt idx
            acc = test(model, vis_processors, subset=subset, model_type=model_type, prompt_idx=idx)
            results.append(acc)
        with open("./results/{}/{}/results.txt".format(model_type, subset), "w") as f:
            for idx, acc in enumerate(results):
                f.write("Accuracy for prompt{}: {} \n".format(idx, acc))
            avg = np.mean(results)
            std = np.std(results, ddof=1)
            f.write("Mean result: {}, Std result: {}".format(100 * avg, 100 * std))


if __name__ == "__main__":
    run()

