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
import copy



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

def combine_and_preprocess(question,image_list,image_padding_tokens, client):
    
    transform = transforms.Compose([                        
                transforms.Resize([512,512],interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])
    images  = []
    new_qestions = [_ for _ in question]
    padding_index = 0
    for img in image_list:
        img_path = img['img_path']
        position = img['position']


        # new image process
        image = np.array(Image.open(image_path)).astype(np.float32)
        if len(image.shape) < 3:
            image = np.expand_dims(image,axis=2).repeat(3,axis=2)
        image = Image.fromarray(image)
        image = transform(image)
        image = image.unsqueeze(0).unsqueeze(-1) # c,w,h,d

        
        ## pre-process the img first
        target_H = 512 
        target_W = 512 
        target_D = 4 
        # This can be different for 3D and 2D images. For demonstration we here set this as the default sizes for 2D images. 
        images.append(torch.nn.functional.interpolate(image, size = (target_H,target_W,target_D)))
        
        ## add img placeholder to text
        new_qestions[position] = "<image>"+ image_padding_tokens[padding_index] +"</image>" + new_qestions[position]
        padding_index +=1
    
    vision_x = torch.cat(images,dim = 1) #cat tensors
    text = ''.join(new_qestions) 
    return text, vision_x, 



class DemoTest(Dataset):
    def __init__(self,json_path, processor):
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


        self.processor = processor
        self.max_words = 128



    def __len__(self):
        return len(self.data_list)
    
    
    def __getitem__(self, index):


        sample = self.data_list[index]
        image_path, question, choice_A, choice_B, choice_C, choice_D, answer, info_all = sample['url'], sample['question'], sample['choice_A'], sample['choice_B'], sample['choice_C'], sample['choice_D'], sample['answer'].strip(), sample['info_all']
        # if isinstance(caption, list):
        #     caption = random.choice(caption)
        question = ' ' + str(question).strip()
        answer = str(answer).strip()
        choice_list = [choice_A, choice_B, choice_C, choice_D]
        choice_label_list = ['A', 'B', 'C', 'D']
        choice_sentence = ''
        for i in range(len(choice_list)):
            if i < 2:
                choice_sentence = choice_sentence + choice_label_list[i] + ':' + choice_list[i] + '.'
            else:
                if choice_list[i] != 'None_Option':
                    choice_sentence = choice_sentence + choice_label_list[i] + ':' + choice_list[i] + '.'
        question_prompt = 'You are a helpful medical assistant. You are being provided an image with a question. Please answer the question. <image>Question:'
        final_prompt = question_prompt + question + ' Answer:'
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


        #####Generate label######
        input1_pool = []
        input2_pool = []
        attention_mask_pool = []
        labels_pool = []
        for j in range(len(choice_list_final)):
            answer_now = choice_list_final[j]
            input1 = final_prompt
            input2 = final_prompt + answer_now

            # ######## new way to make label #######
            tokenized_data1 = self.processor.encode_text(input1)
            tokenized_data2 = self.processor.encode_text(input2)
            input_ids1 = tokenized_data1["input_ids"][0]
            attention_mask1 = tokenized_data1["attention_mask"][0]
            input_ids2 = tokenized_data2["input_ids"][0]
            attention_mask2 = tokenized_data2["attention_mask"][0]
            length1 = input_ids1.shape[0]
            length2 = input_ids2.shape[0]
            labels = copy.deepcopy(input_ids2)
            labels[:length1] = 0
            labels = labels
            input1_pool.append(input_ids1)
            input2_pool.append(input_ids2)
            attention_mask_pool.append(attention_mask2)
            labels_pool.append(labels)



        return {
            "vision_x": pixels,
            "gt_content": answer,
            "choice_list":choice_list_final,
            "lang_x1":input1_pool,
            "lang_x": input2_pool,
            "attention_mask": attention_mask_pool,
            "label": labels_pool,
            "info_all": info_all,
            }
