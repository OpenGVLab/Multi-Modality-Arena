import tqdm.auto as tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import transformers
from dataclasses import dataclass, field
from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from torchvision import transforms
from PIL import Image
import cv2
import json
from torch.utils.data import Dataset
import numpy as np
import spacy
from spacy.tokens import Span
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker


class umls_extractor:
    def __init__(self):
        nlp = spacy.load("en_core_sci_lg")
        nlp.add_pipe("abbreviation_detector")
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        self.nlp = nlp
        
    def extract(self,text):
        doc = self.nlp(text)
        ent_set = doc.ents
        return ent_set

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

def stack_images(images):
    
    target_H = 512
    target_W = 512
    target_D = 4
    if len(images) == 0:
        return torch.zeros((1,3,target_H,target_W,target_D))
    MAX_D = 4
    D_list = list(range(4,65,4))
    
    for ii in images:
        try:
            D = ii.shape[3]
            if D > MAX_D:
                MAX_D = D
        except:
            continue
    for temp_D in D_list:
        if abs(temp_D - MAX_D)< abs(target_D - MAX_D):
            target_D = temp_D
    
    stack_images = []
    for s in images:
        if len(s.shape) == 3:
        #print(s.shape)
            stack_images.append(torch.nn.functional.interpolate(s.unsqueeze(0).unsqueeze(-1), size = (target_H,target_W,target_D)))
        else:
            stack_images.append(torch.nn.functional.interpolate(s.unsqueeze(0), size = (target_H,target_W,target_D)))
    images = torch.cat(stack_images, dim=0)
    return images

def find_position(label, key_embeddings):
    loss_reweight = torch.ones(label.shape)
    for i in range(len(label)):
        if label[i] == -100:
            loss_reweight[i] = 0
        else:
            for key_embedding in key_embeddings:
                if torch.equal(label[i:i+len(key_embedding)], key_embedding):
                    loss_reweight[i:i+len(key_embedding)] = 3
    return loss_reweight

class QuickTest(Dataset):
    def __init__(self,json_path):

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

        
        self.transform_test = transforms.Compose([                        
            transforms.Resize([512,512],interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

        self.max_seq = 100
        self.voc_size = 32000
        self.words_extract = umls_extractor()
        self.text_tokenizer, self.image_padding_tokens = get_tokenizer('./Language_files')



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
        gt_label = choice_label_list[choice_list.index(answer)]
        gt_content = gt_label + ':' + answer
        choice_list_final = [item for item in choice_list if item != 'None_Option']


        # load image
        image = np.array(Image.open(image_path)).astype(np.float32)
        if len(image.shape) < 3:
            image = np.expand_dims(image,axis=2).repeat(3,axis=2)
        image = Image.fromarray(image)
        image = self.transform_test(image)
        image = image.unsqueeze(-1)

        answer = str(answer)
        question = str(question)

        image =[
                {
                    'image': image,
                    'position': {"question": 0}, #indicate where to put the images in the text string, range from [0,len(question)-1]
                }, # can add abitrary number of imgs
            ]
        

        lang_x_pool = []
        attention_mask_pool = []
        labels_pool = []
        reweight_tensor_pool = []
        emphasize_words_pool = []

        ### get vision_x #####
        images_now, question_now, answer_now  = self.text_add_image(image, question, choice_list_final[0])
        # print(question,answer)
        ### make vision_x
        try:
            vision_x = stack_images(images_now)
        except:
            print(self.data_whole[index].items())



        for j in range(len(choice_list_final)):
            images_now, question_now, answer_now  = self.text_add_image(image, question, choice_list_final[j])


            ### make lang_x ###
            self.text_tokenizer.padding_side = "right"
            text_tensor = self.text_tokenizer(
                question_now + ' ' + answer_now, max_length=self.max_seq, truncation=True, padding="max_length", return_tensors="pt"
            )
            lang_x = text_tensor["input_ids"][0]
            attention_mask = text_tensor["attention_mask"][0]
            try:
                lang_x[torch.sum(attention_mask)] = self.text_tokenizer.eos_token_id
            except:
                pass

            ### make label ###
            
            emphasize_words = []
            emphasize_words =  [str(_) for _ in self.words_extract.extract(answer_now)]
            
            if emphasize_words != []:
                emphasize_words_tensor  = self.text_tokenizer(
                    emphasize_words , max_length=self.max_seq
                )
                key_embeddings = [torch.tensor(_[1:]) for _ in emphasize_words_tensor['input_ids']]
            else:
                key_embeddings = []          


            question_tensor = self.text_tokenizer(
                question_now, max_length=self.max_seq, truncation=True, padding="max_length", return_tensors="pt"
            )
            question_length  = torch.sum(question_tensor["attention_mask"][0])
            labels = lang_x.clone()
            labels[labels == self.text_tokenizer.pad_token_id] = -100
            labels[labels >= self.voc_size] = -100
            labels[:question_length] = -100
            
            reweight_tensor = find_position(labels, key_embeddings)


            # set up list
            lang_x_pool.append(lang_x)
            attention_mask_pool.append(attention_mask)
            labels_pool.append(labels)
            reweight_tensor_pool.append(reweight_tensor)
            emphasize_words_pool.append(emphasize_words)


        return {'vision_x': vision_x, 'gt_label': gt_label, 'lang_x':lang_x_pool, 'attention_mask': attention_mask_pool, 'labels':labels_pool, 'loss_reweight': reweight_tensor_pool, 'key_words_query': emphasize_words_pool, 'info_all': info_all}



    def text_add_image(self,images,question,answer):
        ref_image = []
        question = str(question)
        answer = str(answer)
        question_list = [[] for _ in range(len(str(question)))]
        answer_list = [[] for _ in range(len(str(answer)))]
        for index, image in enumerate(images):
            ref_image.append(image["image"])
            position = image["position"]
            position = list(position.items())[0]
            if position[0] == 'question':
                insert_loc = position[1] -1
                if insert_loc < 0:
                    insert_loc = 0
                question_list[insert_loc].append(index)
            if position[0] == 'answer':
                insert_loc = position[1] -1
                if insert_loc < 0:
                    insert_loc = 0
                answer_list[insert_loc].append(index)
        new_question = ''
        new_answer = ''
        question = str(question)
        for char_i in range(len(question)):
            if question_list[char_i] == []:
                new_question = new_question + question[char_i]
            if question_list[char_i] != []:
                for img_index in question_list[char_i]:
                    try:
                        new_question = new_question + '<image>' + self.image_padding_tokens[img_index] + '</image>'
                    except:
                        print("Error: out of max image input size")
                new_question = new_question + question[char_i]
        answer = str(answer)
        for char_i in range(len(str(answer))):
            if answer_list[char_i] == []:
                new_answer = new_answer + answer[char_i]
            if answer_list[char_i] != []:
                for img_index in answer_list[char_i]:
                    try:
                        new_answer = new_answer + '<image>' + self.image_padding_tokens[img_index] + '</image>'
                    except:
                        print("Error: out of max image input size")
                new_answer = new_answer + answer[char_i]
        new_answer = new_answer.replace('•','')

        return ref_image,new_question,new_answer