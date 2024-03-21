import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from copy import deepcopy
import difflib
from PIL import Image
from io import BytesIO

def bytes2PIL(bytes_img):
    '''Transform bytes image to PIL.
    Args:
        bytes_img: Bytes image.
    '''
    pil_img = Image.open(BytesIO(bytes_img)).convert("RGB")
    return pil_img

def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()
 
def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    # Initialize variables to keep track of the most similar string and its index
    most_similar_str = None
    most_similar_index = None
    highest_similarity = 0
    
    # Iterate through each string in the list
    for i, str in enumerate(str_list):
        # Calculate the similarity between the current string and the target string
        similarity = str_similarity(str, target_str)
        
        # If the current string is more similar than the previous most similar string, update the variables
        if similarity > highest_similarity:
            most_similar_str = str
            most_similar_index = i
            highest_similarity = similarity
    
    # Return the index of the most similar string
    return most_similar_index


def MedicalEval(pred_dict: list) -> tuple:
    tot = len(pred_dict)
    succ = 0
    for data in pred_dict:
        try:
            a,b,c,d = data.get('option_A'), data.get('option_B'), data.get('option_C'), data.get('option_D')
            answer_list = [a, b]
            if c is not None:
                answer_list.append(c)
            if d is not None:
                answer_list.append(d)
            
            if answer_list[find_most_similar_index(answer_list, data['model_pred'])] == data['gt_answer']:
                succ += 1
                data['is_correct'] = 'yes'
            else:
                data['is_correct'] = 'no'
        except:
            continue
        
    return pred_dict, succ/tot


def evaluate_medical_QA(
    model,
    dataset,
    model_name,
    dataset_name,
    time,
    batch_size=1,
    answer_path='./answers'
):
    predictions=[]
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
    for batch in tqdm(dataloader, desc="Running inference"):
        try:
            image_paths = batch['image_path']
            imgs = []
            for img_path in image_paths:
                imgs.append(Image.open(img_path).convert("RGB"))
            outputs = model.batch_generate(imgs, batch['question'])
            print(outputs)
            for entity, output in zip(batch['entity'], outputs):
                answer_dict = deepcopy(entity)
                answer_dict['model_pred'] = output
                predictions.append(answer_dict)
        except:
            continue

    answer_dir = os.path.join(answer_path, time)
    os.makedirs(answer_dir, exist_ok=True)
        
    final_dict, correct_precentage = MedicalEval(predictions)
    save_dict = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "correct_precentage" :correct_precentage,
        "pred_dict" : final_dict
    }
    
    
    answer_path = os.path.join(answer_dir, f"{dataset_name}.json")
    with open(answer_path, "w") as f:
        f.write(json.dumps(save_dict, indent=4))
        
    print(f'{dataset_name}:{correct_precentage}')
    return correct_precentage

def pred_medical_QA(
    model,
    dataset,
    model_name,
    dataset_name,
    time,
    batch_size=1,
    answer_path='./answers'
):
    predictions=[]
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
    for batch in tqdm(dataloader, desc="Running inference"):
        try:
            print(batch['question'])
            image_paths = batch['image_path']
            imgs = []
            for img_path in image_paths:
                imgs.append(Image.open(img_path).convert("RGB"))
            outputs = model.batch_generate(imgs, batch['question'])
            print(outputs)
            for entity, output in zip(batch['entity'], outputs):
                answer_dict = deepcopy(entity)
                answer_dict['model_pred'] = output
                predictions.append(answer_dict)
        except:
            print(f'error: {batch}')
            continue

    answer_dir = os.path.join(answer_path, time)
    os.makedirs(answer_dir, exist_ok=True)
    
    save_dict = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "pred_dict" : predictions
    }

    answer_path = os.path.join(answer_dir, f"{model_name}_{dataset_name}.json")
    with open(answer_path, "w") as f:
        f.write(json.dumps(save_dict, indent=4))
        
    return save_dict
