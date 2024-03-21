import tqdm.auto as tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from torchvision import transforms
from PIL import Image   
from petrel_client.client import Client
import cv2
import json
from torch.utils.data import Dataset
import numpy as np
from einops import repeat



def get_tokenizer(tokenizer_path, max_img_size = 100, image_num = 32):
    '''
    Initialize the image special tokens
    max_img_size denotes the max image put length and image_num denotes how many patch embeddings the image will be encoded to 
    '''
    if isinstance(tokenizer_path,str):
        image_padding_tokens = []
        text_tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_path,
        )
        special_token = {"additional_special_tokens": ["<image>","</image>"]}
        for i in range(max_img_size):
            image_padding_token = ""
            
            for j in range(image_num):
                image_token = "<image"+str(i*image_num+j)+">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append("<image"+str(i*image_num+j)+">")
            image_padding_tokens.append(image_padding_token)
            text_tokenizer.add_special_tokens(
                special_token
            )
            ## make sure the bos eos pad tokens are correct for LLaMA-like models
            text_tokenizer.pad_token_id = 0
            text_tokenizer.bos_token_id = 1
            text_tokenizer.eos_token_id = 2    
    
    return  text_tokenizer,image_padding_tokens



class DemoTest(Dataset):
    def __init__(self,json_path, processor):
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

        self.processor = processor



    def __len__(self):
        return len(self.data_list)
    
    
    def __getitem__(self, index):


        sample = self.data_list[index]
        image_path, question, choice_A, choice_B, choice_C, choice_D, answer, info_all = sample['url'], sample['question'], sample['choice_A'], sample['choice_B'], sample['choice_C'], sample['choice_D'], sample['answer'].strip(), sample['info_all']
        question = ' ' + str(question).strip()
        answer = str(answer).strip()
        choice_list = [choice_A, choice_B, choice_C, choice_D]
        choice_list_long = ['A:' + choice_A, 'B:' + choice_B, 'C:' + choice_C, 'D:' + choice_D]
        choice_label_list = ['A', 'B', 'C', 'D']
        choice_sentence = ''
        for i in range(len(choice_list)):
            if i < 2:
                choice_sentence = choice_sentence + choice_label_list[i] + ':' + choice_list[i] + '.'
            else:
                if choice_list[i] != 'None_Option':
                    choice_sentence = choice_sentence + choice_label_list[i] + ':' + choice_list[i] + '.'
        question_prompt = 'This is a medical Question with several Options, and there is only one correct answer among these options. Please select the correct answer for the question. Remember, you can only select one option. <image>Question:'
        option_prompt = '### The candidate Options are: '
        question_sentence = question_prompt + question + option_prompt + choice_sentence
        final_prompt = question_sentence + '### Answer:'
        gt_label = choice_label_list[choice_list.index(answer)]
        gt_content = gt_label + ':' + answer
        choice_list_final = [item for item in choice_list if item != 'None_Option']


        image = np.array(Image.open(image_path)).astype(np.float32)
        if len(image.shape) < 3:
            image = np.expand_dims(image,axis=2).repeat(3,axis=2)
        image = Image.fromarray(image)
        demo_images = [image]


        pixels = self.processor.preprocess_images(demo_images)
        pixels = repeat(pixels, 'N c h w -> N T c h w', T=1)
        tokenized_data = self.processor.encode_text(final_prompt)
        lang_x=tokenized_data["input_ids"]
        attention_mask=tokenized_data["attention_mask"]



        return {
            "vision_x": pixels,
            "gt_content": gt_content,
            "choice_list":choice_list,
            "choice_list_long":choice_list_long,
            "lang_x": lang_x,
            "attention_mask": attention_mask,
            "info_all": info_all,
            }







