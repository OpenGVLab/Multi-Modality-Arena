import argparse
import os
import csv
import json
import math
import numpy as np
import tqdm.auto as tqdm
from typing import Optional
import difflib 
import transformers
from transformers import Trainer
from dataclasses import dataclass, field
from torch import nn
from torch.utils.data import DataLoader  
import torch
from tensorboardX import SummaryWriter
from torch.nn import functional as F
import pdb

from dataset import Binary_OUR_VQA_Dataset
from models.llama.vqa_model import Binary_VQA_Model


def append_to_json(save_path, old_data, pred_now, correct_flag):

    base_name = old_data['question_id'][0] + '.json'
    filename = os.path.join(save_path, base_name)

    old_data['pred'] = pred_now
    old_data['is_correct'] = correct_flag
    json_data = old_data

    with open(filename, 'w') as f:
        f.write(json.dumps(json_data, indent=4))


@dataclass
class ModelArguments:
    embed_dim: Optional[int] = field(default=768)
    pretrained_tokenizer:  Optional[str] = field(default="path_to/llama/tokenizer.model")
    pretrained_model: Optional[str] = field(default="path_to/llama/7B_hf")
    image_encoder: Optional[str] = field(default="PMC_CLIP")
    pmcclip_pretrained: Optional[str] = field(default="path_to/pmc_clip/standard_res50/checkpoint.pt")
    clip_pretrained: Optional[str] = field(default="openai/clip-vit-base-patch32")
    ckp: Optional[str] = field(default="./Results/VQA_lora_noclip/vqa/checkpoint-6500")
    

@dataclass
class DataArguments:
    is_blank: Optional[bool] = field(default=True)
    image_res: Optional[int] = field(default=512)
    json_path: str = field(default='json_path')

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default="./Results")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    logging_dir: Optional[str] = field(default="./logs")
    logging_steps: Optional[int] = field(default=50)
    

def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()
 
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
    return most_similar_index
  
def get_generated_texts(label,outputs,tokenizer):
    #1,256
    outputs = outputs[label!=0][1:-1]
    generated_text = tokenizer.decode(outputs)
    return generated_text

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    save_root_path = os.path.join(training_args.output_dir, 'saved_dir')



    index_list = ['A','B','C','D']
    print("Setup Model")
    ckp = model_args.ckp + '/pytorch_model.bin'
    print(ckp)
    print(model_args.image_encoder)
    model = Binary_VQA_Model(model_args)
    ckpt = torch.load(ckp, map_location='cpu')
    for name in list(ckpt.keys()):
        if 'lora_A' in name:
            new_name = name.replace('lora_A', 'lora_A.default')
            ckpt[new_name] = ckpt.pop(name)
        if 'lora_B' in name:
            new_name = name.replace('lora_B', 'lora_B.default')
            ckpt[new_name] = ckpt.pop(name)
    model.load_state_dict(ckpt)
 


    json_pool_path = data_args.json_path
    with open(json_pool_path, 'r') as json_pool_file:
        json_pool = json.load(json_pool_file)
    dataset_position = os.path.join(training_args.output_dir, 'dataset_position.npy')
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

        print("Setup Data")
        Test_dataset = Binary_OUR_VQA_Dataset(json_path_now, data_args.image_res,is_blank=data_args.is_blank,is_train=False)
        # batch size should be 1
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


        # selected_parameter = 3000
        test_position = os.path.join(training_args.output_dir, 'test_position{}.npy'.format(json_iter))
        save_path = os.path.join(training_args.output_dir, 'test_log{}.txt'.format(json_iter))
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

        with open(os.path.join(training_args.output_dir,'result.csv'), mode='w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['Figure_path','Pred','Label','Correct'])
            np.save(dataset_position, [json_iter])
            for i, sample in enumerate(Test_dataloader):
                if i < start_point or i == start_point:
                    continue
                else:
                    loss_pool = []
                    image = sample['image'].to('cuda')
                    for m in range(len(sample['label'])):
                        img_path = sample['image_path']
                        label = sample['label'][m].to('cuda')[:,0,:]
                        question_inputids = sample['encoded_input_ids'][m].to('cuda')[:,0,:]
                        question_attenmask = sample['encoded_attention_mask'][m].to('cuda')[:,0,:]
                        with torch.no_grad():
                            outputs = model(image,question_inputids,question_attenmask)# 
                        loss = F.nll_loss(outputs.transpose(1, 2), label, ignore_index=0)
                        loss_item = loss.cpu().numpy().item()
                        loss_pool.append(loss_item)

                    Answer_label = sample['Answer_label'][0].strip().lower()
                    pred_label = index_list[np.argmin(loss_pool)].strip().lower()
                    corret = 0
                    if Answer_label == pred_label:
                        ACC = ACC +1
                        corret = 1 
                    append_to_json(save_json_path, sample['Info_all'], loss_pool, corret)
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
            writer.writerow([ACC/i])


if __name__ == "__main__":
    main()
    
