import os, json
import argparse
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
import difflib
import requests

from dataset_quicktest_losstest import QuickTest

import numpy as np
import tqdm.auto as tqdm
from torch.utils.data import DataLoader  
import torch
import pdb



def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()

def append_to_json(save_path, old_data, pred_now, correct_flag):

    base_name = old_data['question_id'][0] + '.json'
    filename = os.path.join(save_path, base_name)

    old_data['pred'] = pred_now
    old_data['is_correct'] = correct_flag
    json_data = old_data

    with open(filename, 'w') as f:
        f.write(json.dumps(json_data, indent=4))

def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    # Initialize variables to keep track of the most similar string and its index
    most_similar_str = 'Error Bug'
    most_similar_index = 'Error Bug'
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
    return most_similar_str

def get_args_parser():
    parser = argparse.ArgumentParser('radfm test', add_help=False)
    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--json_path', default='json_path',
                        help='a json file that contains the path of all the evaluated json files')

    return parser

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
    
    
def main():

    args = get_args_parser()
    args = args.parse_args()
    save_root_path = os.path.join(args.output_dir, 'saved_dir')
    index_list = ['A','B','C','D']
    
    print("Setup Model")
    model = MultiLLaMAForCausalLM(
        lang_model_path='./Language_files', ### Build up model based on LLaMa-13B config
    )
    ckpt = torch.load('PATH_TO_RADFM_PARAMETER',map_location ='cpu') # Please dowloud our checkpoint from huggingface and Decompress the original zip file first
    model.load_state_dict(ckpt)
    print("Finish loading model")


    print("Setup tokenizer")
    text_tokenizer,image_padding_tokens = get_tokenizer('./Language_files')
    print("Finish loading tokenizer")

    json_pool_path = args.json_path
    with open(json_pool_path, 'r') as json_pool_file:
        json_pool = json.load(json_pool_file)
    dataset_position = os.path.join(args.output_dir, 'dataset_position.npy')
    if os.path.exists(dataset_position):
        begin_dataset = int(np.load(dataset_position)[0])
    else:
        begin_dataset = 0

    for json_iter in range(begin_dataset, len(json_pool)):
        np.save(dataset_position, [json_iter])
        json_path_now = json_pool[json_iter]

        modality_now = json_path_now.split('/')[-3]
        dataset_now = json_path_now.split('/')[-2]
        save_json_path = os.path.join(save_root_path, modality_now, dataset_now)
        if not os.path.exists(save_json_path):
            os.makedirs(save_json_path)


        # setup dataset
        print("Setup Data")
        Test_dataset = QuickTest(json_path_now)
        Test_dataloader = DataLoader(
                Test_dataset,
                batch_size=1,
                num_workers=1,
                pin_memory=True,
                sampler=None,
                shuffle=False,
                collate_fn=None,
                drop_last=False,
        )


        print("Start Testing")
        
        model = model.to('cuda')
        model.eval()

        # setup test record
        test_position = os.path.join(args.output_dir, 'test_position{}.npy'.format(json_iter))
        save_path = os.path.join(args.output_dir, 'test_log{}.txt'.format(json_iter))
        begin_point = os.path.exists(test_position)
        if begin_point:
            test_record = open(save_path, 'a')
            start_point = int(np.load(test_position)[0])
            ACC = int(np.load(test_position)[1])
        else:
            test_record = open(save_path, 'w')
            start_point = 0
            ACC = 0

        print('modality_now: {}, dataset_now: {} \n'.format(modality_now, dataset_now))
        test_record.write('modality_now: {}, dataset_now: {} \n'.format(modality_now, dataset_now))



        for i, sample in enumerate(Test_dataloader):
            if i < start_point or i == start_point:
                continue
            else:
                with torch.no_grad():
                    vision_x = sample['vision_x'].to('cuda')

                    sample_length = len(sample['lang_x'])

                    loss_pool = []
                    acc_pool = []

                    for m in range(sample_length):
                        lang_x = sample['lang_x'][m].to('cuda')
                        attention_mask = sample['attention_mask'][m].to('cuda')
                        labels = sample['labels'][m].to('cuda')
                        loss_reweight = sample['loss_reweight'][m].to('cuda')
                        key_words_query = sample['key_words_query'][m]

                        output_container = model(lang_x, vision_x, attention_mask, labels, loss_reweight,key_words_query)
                        loss = output_container['loss'].cpu().numpy().item()
                        acc = output_container['logits'].cpu().numpy().item()
                        loss_pool.append(loss)
                        acc_pool.append(acc)

                    correct = 0
                    gt_label = sample['gt_label'][0].strip().lower()
                    pred_label = index_list[np.argmin(loss_pool)].strip().lower()
                    if gt_label == pred_label:
                        ACC = ACC +1
                        correct = 1 
                    append_to_json(save_json_path, sample['info_all'], loss_pool, correct)
                    np.save(test_position, [i, ACC])

                    if i % 100 == 0:
                        accuracy = ACC / i
                        print('The accuracy at {} is {} \n'.format(i, accuracy))
                        test_record.write('The accuracy at {} is {} \n'.format(i, accuracy))
                        test_record.flush()
                     

        final_accuracy = ACC/i
        print('The test is end, we test {} samples, the final accuracy is {} \n'.format(i, final_accuracy))
        test_record.write('The test is end, we test {} samples, the final accuracy is {} \n'.format(i, final_accuracy))
        test_record.flush()

    
if __name__ == "__main__":
    main()
       
