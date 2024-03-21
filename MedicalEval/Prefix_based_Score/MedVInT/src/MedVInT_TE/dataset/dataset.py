import csv
import json
import logging
import os
import re
import difflib
import sys
import torch
import random
from abc import abstractmethod
from itertools import islice
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Mapping
from torch.utils.data import DataLoader
import PIL
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from collections import defaultdict
from PIL import Image
from dataset.randaugment import RandomAugment
from transformers import AutoModel,BertConfig,AutoTokenizer,AutoProcessor,LlamaTokenizer
import cv2
import json


class Binary_OUR_VQA_Dataset(Dataset):
    def __init__(self,json_path,image_res,is_blank,is_train=True):
        self.is_blank = is_blank
        self.is_train = is_train
        self.tokenizer = LlamaTokenizer.from_pretrained('/mnt/petrelfs/huyutao/pretrain_model/llama/tokenizer.model')
        special_tokens_dict = {'mask_token': "</s>",
                               'eos_token': "</s>",
                               'bos_token': "<s>",
                               'unk_token': "<unk>"}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.tokenizer.pad_token_id=0
        
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.train_transform = transforms.Compose([                        
                transforms.RandomResizedCrop([image_res,image_res],scale=(0.8, 1.0),ratio=(0.75, 1.3333), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),    
                transforms.ToTensor(),
                normalize,
            ])
        self.test_transform = transforms.Compose([                        
                transforms.RandomResizedCrop([image_res,image_res]),   
                transforms.ToTensor(),
                normalize,
            ])



        print('Load dataset from {}'.format(json_path))
        path_pool = [json_path]
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

        if self.is_train:
            self.transform = self.train_transform
        else:
            self.transform = self.test_transform
        
    def encode_mlm(self,question_text,question_text_with_answer,mask_token= '</s>', pad_token='<unk>', eos_token = '</s>'):
        def measure_word_len(word):
            token_ids = self.tokenizer.encode(word)
            return len(token_ids) - 1
        
        question_text_with_answer_tokens = question_text_with_answer.split()
        question_text_tokens = question_text.split()
        bert_input_tokens = []
        output_mask = []
        bert_label_tokens = []  # 被 mask 的保留原词, 否则用 [PAD] 代替
        
        for i, token in enumerate(question_text_with_answer_tokens):
            if i < len(question_text_tokens):
                word_len = measure_word_len(token)
                bert_input_tokens += [token]
                bert_label_tokens += [pad_token]*word_len
                output_mask += [0]*word_len
            else:
                word_len = measure_word_len(token)
                bert_input_tokens += [mask_token]*word_len
                bert_label_tokens += [token]
                output_mask += [1]*word_len
        bert_input_tokens += [eos_token]
        bert_label_tokens += [eos_token]
        bert_input = ' '.join(bert_input_tokens)
        bert_label = ' '.join(bert_label_tokens)
        return bert_input,bert_label
    
    def __len__(self):
        return len(self.data_list)
    
    
    def __getitem__(self, index):

        sample = self.data_list[index]
        image_path, question, choice_A, choice_B, choice_C, choice_D, answer, info_all = sample['url'], sample['question'], sample['choice_A'], sample['choice_B'], sample['choice_C'], sample['choice_D'], sample['answer'].strip(), sample['info_all']
        question = ' ' + str(question).strip()
        answer = str(answer).strip()
        choice_list = [choice_A, choice_B, choice_C, choice_D]
        choice_label_list = ['A', 'B', 'C', 'D']
        choice_sentence = ''
        for i in range(len(choice_list)):
            if i < 2:
                choice_sentence = choice_sentence + choice_label_list[i] + ':' + choice_list[i] + ' '
            else:
                if choice_list[i] != 'None_Option':
                    choice_sentence = choice_sentence + choice_label_list[i] + ':' + choice_list[i] + ' '
        question_sentence = question + choice_sentence
        gt_label = choice_label_list[choice_list.index(answer)]
        gt_content = gt_label + ':' + answer
        choice_list_final = [item for item in choice_list if item != 'None_Option']

        image = np.array(Image.open(image_path)).astype(np.float32)

        if len(image.shape) < 3:
            image = np.expand_dims(image,axis=2).repeat(3,axis=2)
        image = Image.fromarray(image)

        image = self.transform(image)
        answer = gt_content



        encoded_input_ids_pool = []
        encoded_attention_mask_pool = []
        label_pool = []
        for j in range(len(choice_list_final)):
            answer_now = choice_list_final[j]
            answer_label_now = choice_label_list[j]

            if self.is_blank:
                question_text = 'Question: '+ question +'The Answer is:'
                question_text_with_answer = 'Question: '+ question +'The Answer is:' + answer_now
            else:
                question_text = 'Question: '+ question + 'The choices are: '  + choice_sentence + 'The Answer is: '
                question_text_with_answer = 'Question: '+ question + 'The choices are: '  + choice_sentence + 'The Answer is: ' + answer_label_now

        
            bert_input, bert_label = self.encode_mlm(question_text,question_text_with_answer)

            if self.is_train:
                encoded_input = self.tokenizer(bert_input, add_special_tokens=True, padding='max_length', truncation=True, max_length= 256)
                encoded_label = self.tokenizer(bert_label, add_special_tokens=True, padding='max_length', truncation=True, max_length= 256)
            else:
                encoded_input = self.tokenizer(bert_input, add_special_tokens=True, padding='max_length', truncation=True, max_length= 256,return_tensors="pt")
                encoded_label = self.tokenizer(bert_label, add_special_tokens=True, padding='max_length', truncation=True, max_length= 256,return_tensors="pt")
            
            encoded_input_ids_pool.append(encoded_input['input_ids'])
            encoded_attention_mask_pool.append(encoded_input['attention_mask'])
            label_pool.append(encoded_label['input_ids'])


        return {
            "image": image,
            "encoded_input_ids": encoded_input_ids_pool,
            "encoded_attention_mask": encoded_attention_mask_pool,
            "label": label_pool,
            "image_path": image_path,
            'Choice_list': choice_list_final,
            'Answer': answer,
            'Answer_label': gt_label,
            'Info_all': info_all,
            }


