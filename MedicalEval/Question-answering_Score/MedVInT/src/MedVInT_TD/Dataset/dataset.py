import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
import PIL
import numpy as np
import torch.nn.functional as F
import transformers
import pandas as pd
import random
import copy
from .randaugment import RandomAugment    
from PIL import Image
import tqdm
import cv2
import json


class Slake_Dataset(Dataset):
    def __init__(self,  json_path, tokenizer_path, img_dir = './Data/Slake1.0/imgs/',img_tokens = 32, seq_length = 512,voc_size = 32000, mode = 'Test',text_type = 'blank'):
        self.img_root = img_dir
        self.tokenizer = transformers.LlamaTokenizer.from_pretrained('path_to/llama/tokenizer.model')
        self.tokenizer.pad_token_id=0
        self.tokenizer.eos_token_id=1
        self.mode = mode
        self.img_padding = [-100 for i in range(img_tokens)]
        self.attn_padding = [1 for i in range(img_tokens)]
        self.H = 512
        self.W = 512
        self.C = 3
        self.text_type = text_type
        
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop((self.H,self.W),scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                #transforms.RandomHorizontalFlip(),
                RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',]),     
                transforms.ToTensor(),
                normalize,
            ]) 
        if self.mode == 'Test':
            self.transform = transforms.Compose([                        
                    transforms.Resize((self.H,self.W), interpolation=Image.BICUBIC),
                    transforms.ToTensor(),
                    normalize,
                ])
            
        self.mode = mode
        self.seq_length = seq_length
        self.voc_size = voc_size

        print('Load dataset from {}'.format(json_path))
        path_pool = [json_path]
        # pdb.set_trace()
        images, questions, choice_A, choice_B, choice_C, choice_D, answers, info_all = [], [], [], [], [], [], [], []
        for meta_path in path_pool:
            with open(meta_path, 'r', encoding="utf-8") as f:
                data = json.load(f)
                images_this_meta, questions_this_meta, choice_A_this_meta, choice_B_this_meta, choice_C_this_meta, choice_D_this_meta, answers_this_meta, info_all_this_meta = [], [], [], [], [], [], [], []
                for i in range(len(data)):
                    data_now = data[i]
                    keys_now = data_now.keys()
                    images_this_meta.append(data_now['image_path'])
                    questions_this_meta.append(data_now['question'])
                    choice_A_this_meta.append(str(data_now['option_A']).strip())
                    choice_B_this_meta.append(str(data_now['option_B']).strip())
                    answers_this_meta.append(str(data_now['gt_answer']))
                    if 'option_C' in keys_now:
                        choice_C_this_meta.append(str(data_now['option_C']).strip())
                    else:
                        choice_C_this_meta.append('None_Option')
                    if 'option_D' in keys_now:
                        choice_D_this_meta.append(str(data_now['option_D']).strip())
                    else:
                        choice_D_this_meta.append('None_Option')
                    info_all_this_meta.append(data_now)
                images.extend(images_this_meta)
                questions.extend(questions_this_meta)
                choice_A.extend(choice_A_this_meta)
                choice_B.extend(choice_B_this_meta)
                choice_C.extend(choice_C_this_meta)
                choice_D.extend(choice_D_this_meta)
                answers.extend(answers_this_meta)
                info_all.extend(info_all_this_meta)

        self.data_list = []
        for x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8 in zip(images, questions, choice_A, choice_B, choice_C, choice_D, answers, info_all):
            image_path = x_1
            self.data_list.append({'url': image_path, 'question': x_2, 'choice_A': x_3, 'choice_B': x_4, 'choice_C': x_5, 'choice_D': x_6, 'answer': x_7, 'info_all': x_8})
        print(f"total length: {len(self)}")
    
    def random_answer(self, Question,Answer):
        Answer = str(Answer)
        pre_text = 'Question: '+ Question +'The Answer is:'
        final_o = 'Question: '+ Question +'The Answer is:' + Answer
        return pre_text,final_o
     
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        sample = self.data_list[idx]
        image_path, question, choice_A, choice_B, choice_C, choice_D, answer, info_all = sample['url'], sample['question'], sample['choice_A'], sample['choice_B'], sample['choice_C'], sample['choice_D'], sample['answer'].strip(), sample['info_all']
        question = ' ' + str(question).strip()
        answer = str(answer).strip()
        choice_list = [choice_A, choice_B, choice_C, choice_D]
        choice_label_list = ['A', 'B', 'C', 'D']
        choice_sentence = ' '
        for i in range(len(choice_list)):
            if i < 2:
                choice_sentence = choice_sentence + choice_label_list[i] + ':' + choice_list[i] + ' '
            else:
                if choice_list[i] != 'None_Option':
                    choice_sentence = choice_sentence + choice_label_list[i] + ':' + choice_list[i] + ' '
        question_sentence = 'Question: ' + question + 'Choices:' + choice_sentence +'The Answer is:'
        gt_label = choice_label_list[choice_list.index(answer)]
        gt_content = gt_label + ':' + answer
        choice_list_final = [item for item in choice_list if item != 'None_Option']
        image = np.array(Image.open(image_path)).astype(np.float32)

        if len(image.shape) < 3:
            image = np.expand_dims(image,axis=2).repeat(3,axis=2)
        image = Image.fromarray(image)

        image = self.transform(image)
        answer = gt_content
        
        if self.mode == 'Train':
            pre_text,final_o = self.random_answer(question, answer)
            
            final_o = self.tokenizer(final_o)
            input_ids = final_o['input_ids']
            input_ids.append(self.tokenizer.eos_token_id)
            input_ids = np.array(input_ids)
            
            if len(input_ids) < self.seq_length:
                input_ids = np.pad(input_ids, (0, self.seq_length - len(input_ids)), 'constant', constant_values=0)
            else:
                input_ids = input_ids[:self.seq_length]
                
            label = copy.deepcopy(input_ids)
            label[label==0] = -100
            if pre_text != '':
                pre_text = self.tokenizer(pre_text)
                if len(pre_text['input_ids'])<len(label):
                    label[:len(pre_text['input_ids'])] = -100
            label = label.tolist()
            label = np.array(self.img_padding + label)
            
            item = {
                'input_ids': input_ids,       
                'images': image,
                'labels': label,
            }
                
        if self.mode == 'Test':
            item = {
                'input_ids': question_sentence,
                'img_path': image_path,       
                'images': image,
                'labels': answer,
                'Answer_label': gt_label,
                'Info_all': info_all,
                'Choice_list': choice_list_final,
            }
        return item
        

