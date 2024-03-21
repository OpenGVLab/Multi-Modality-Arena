import os, json
from torch.utils.data import Dataset
import random

class MedicalDataset(Dataset):
    def __init__(self, data_path, num=3):
        self.data = json.load(open(data_path))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entity = self.data[idx]
        a,b,c,d = entity.get('option_A'), entity.get('option_B'), entity.get('option_C'), entity.get('option_D')
        answer_list = [a, b]
        if c is not None:
            answer_list.append(c)
        if d is not None:
            answer_list.append(d)
        question = entity['question'] + f'Here are {len(answer_list)} candidate answers:' + str(answer_list)+' Only return what you think is the correct answer from the candidate answers, do not return any other irrelevant text!'
        answers = entity['gt_answer']
        img_path = entity['image_path']
        
        return {
                "image_path": img_path,
                "question": question,
                "entity": entity
            }
    
    
class MedicalQADataset(Dataset):
    def __init__(self, data_path, num=64):
        self.data = json.load(open(data_path))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entity = self.data[idx]
        question = entity['question']
        img_path = entity['image_path']
        
        return {
                "image_path": img_path,
                "question": question,
                "entity": entity
            }
    