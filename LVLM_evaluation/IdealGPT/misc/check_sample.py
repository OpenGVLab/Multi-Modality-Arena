from tqdm import tqdm
import os
import yaml
import pdb
import argparse
import re
import pdb
import random
import jsonlines

def parse():
    parser = argparse.ArgumentParser(description='IdealGPT in test datasets.')
    parser.add_argument('--result', type=str, default='/home/haoxuan/code/SubGPT/exp_result/vcr_val_random500_t5xl_captionv2_twoagent_promptv1a_multitry/result', 
                        help='VCR predicted result path')
    parser.add_argument("--data_subset", type=str, default='/home/haoxuan/code/SubGPT/exp_data/vcr_val_random500_annoid.yaml', help="specify the subset of the dataset.")
    args = parser.parse_args()
    return args


args = parse()
blip_gpt_folder = args.result
random.seed(3)

with open(args.data_subset, "r") as file:
    id_partitions = yaml.safe_load(file)

ids_insubset = []
dataset_anno_dir = os.path.join('/home/haoxuan/data/vcr1/', 'val.jsonl')
with jsonlines.open(dataset_anno_dir) as reader:
    for cur_ann in tqdm(reader):
        annot_id = cur_ann['annot_id']
        if annot_id in id_partitions:
            ids_insubset.append(annot_id)

result_ids = os.listdir(blip_gpt_folder)
result_ids = [i.split('.yaml')[0] for i in result_ids]

for i, value in enumerate(tqdm(ids_insubset)):
    if value not in result_ids:
        print(f'Miss {i}-th data, with id:{value}')
