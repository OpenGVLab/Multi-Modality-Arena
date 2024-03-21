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

from dataset_quicktest import QuickTest

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

def combine_and_preprocess(question,image_list,image_padding_tokens):
    
    transform = transforms.Compose([                        
                transforms.RandomResizedCrop([512,512],scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])
    images  = []
    new_qestions = [_ for _ in question]
    padding_index = 0
    for img in image_list:
        img_path = img['img_path']
        position = img['position']
        
        
        
        image = Image.open(img_path).convert('RGB')   
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
    
    vision_x = torch.cat(images,dim = 1).unsqueeze(0) #cat tensors and expand the batch_size dim
    text = ''.join(new_qestions) 
    return text, vision_x, 
    
    
def main():

    args = get_args_parser()
    args = args.parse_args()
    save_root_path = os.path.join(args.output_dir, 'saved_dir')
    
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
                    # pdb.set_trace()
                    text = sample['text'][0]
                    vision_x = sample['vision_x'].to('cuda')

                    lang_x = text_tokenizer(
                            text, max_length=2048, truncation=True, return_tensors="pt"
                    )['input_ids'].to('cuda')

                    generation = model.generate(lang_x,vision_x)
                    generated_texts = text_tokenizer.batch_decode(generation, skip_special_tokens=True) 
                
                
                    gt_content = sample['gt_content'][0]
                    Choice_list = [item[0] for item in sample['choice_list']]
                    matched_pred = find_most_similar_index(Choice_list, generated_texts[0])
                    matched_gt  = find_most_similar_index(Choice_list, gt_content)
                    corret = 0
                    if matched_pred == matched_gt:
                        ACC = ACC +1
                        corret = 1 
                    append_to_json(save_json_path, sample['info_all'], generated_texts, corret)
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
       
