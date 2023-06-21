import argparse
import torch
import yaml
import json
from tqdm import tqdm
import jsonlines
import os
import sys
# Get the absolute path to the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
from lib.blip2_lib import Blip2Lavis


parser = argparse.ArgumentParser(description='IdealGPT in test datasets.')
parser.add_argument('--device_id', type=int, default=0, 
                    help='Which GPU to use.')
parser.add_argument("--dataset", type=str, default='vcr_val', help="specify the dataset to generate caption.")
parser.add_argument("--data_subset", type=str, default=None, help="specify the subset of the dataset.")
parser.add_argument("--blip_model", type=str, default='pretrain_flant5xl', help="specify the BLIP2 model.")
parser.add_argument("--data_partition", type=str, default=None, help="id_id, specify the partition of the dataset.")
parser.add_argument('--prompt_setting', type=str,  default='v1', 
                    help='Prompt Setting Version, currently support v1/v2')
args = parser.parse_args()

def load_vcr_data(vcr_data_path):
    img_paths = {}
    vcr_anno_path = os.path.join(vcr_data_path, 'val.jsonl')

    # Obtain subset annot_ids
    id_partitions = None
    if args.data_subset is not None:
        with open(args.data_subset, "r") as file:
            id_partitions = yaml.safe_load(file)
            
    with jsonlines.open(vcr_anno_path) as reader:
        for cur_ann in tqdm(reader):
            annot_id = cur_ann['annot_id']
            # Only keep the input partition.
            if id_partitions is not None:
                if annot_id not in id_partitions:
                    continue
            img_path = os.path.join(vcr_data_path, 'vcr1images', cur_ann['img_fn'])
            img_paths[annot_id] = img_path

    # Only keep samples in partitions.
    if args.data_partition is not None:
        start_data_id, end_data_id = args.data_partition.split('_')
        _ids = list(img_paths)[int(start_data_id):int(end_data_id)+1]
        img_paths = {key:img_paths[key] for key in _ids}

    return img_paths

def load_ve_data(ve_data_path, image_path):
    ve_anno_path = os.path.join(ve_data_path, 'snli_ve_dev.jsonl')
    img_paths = {}

    # Obtain subset annot_ids
    id_partitions = None
    if args.data_subset is not None:
        with open(args.data_subset, "r") as file:
            id_partitions = yaml.safe_load(file)

    with jsonlines.open(ve_anno_path) as jsonl_file:
        for line in jsonl_file:
            pair_id = line['pairID']
            image_name = line['Flikr30kID']
            if id_partitions is not None:
                if pair_id not in id_partitions:
                    continue
            img_path = os.path.join(image_path, image_name)
            img_paths[pair_id] = img_path

    # Only keep samples in partitions.
    if args.data_partition is not None:
        start_data_id, end_data_id = args.data_partition.split('_')
        _ids = list(img_paths)[int(start_data_id):int(end_data_id)+1]
        img_paths = {key:img_paths[key] for key in _ids}

    return img_paths

# ========================================
#             Data Loading
# ========================================
if 'vcr' in args.dataset:
    vcr_data_path = '/home/haoxuan/data/vcr1/'
    img_paths = load_vcr_data(vcr_data_path)
elif 've' in args.dataset:
    ve_data_path = '/dvmm-filer3a/users/rui/multi-task/datasets/visual_entailment/'
    ve_image_path = '/home/tangtangwzc/flickr_data/flickr30k_images/flickr30k_images/'
    img_paths = load_ve_data(ve_data_path, ve_image_path)
else:
    raise NotImplementedError('Not support other datasets yet.')
print('{} Samples Loading Finished'.format(len(img_paths)))

if args.data_subset is not None:
    subset_name = args.data_subset.split('/')[-1].split('.yaml')[0]
else:
    subset_name = 'fullset'

if 't5xl' in args.blip_model:
    vqa_model = Blip2Lavis(name="blip2_t5", model_type="pretrain_flant5xl", device=torch.device("cuda:{}".format(args.device_id)))
elif 't5xxl' in args.blip_model:
    vqa_model = Blip2Lavis(name="blip2_t5", model_type="pretrain_flant5xxl", device=torch.device("cuda:{}".format(args.device_id)))
else:
    raise NotImplementedError(f'Not Support {args.blip_model} yet.')

if args.prompt_setting == 'v1':
    caption_prompt = 'a photo of'
    result_folder = os.path.join('./exp_data/caption', args.dataset, f'blip_{args.blip_model}_{subset_name}_prompt_aphotoof')
elif args.prompt_setting == 'v2':
    caption_prompt = ''
    result_folder = os.path.join('./exp_data/caption', args.dataset, f'blip_{args.blip_model}_{subset_name}_prompt_v2_none')

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# ========================================
#             Generating Captions
# ========================================
result = {}
for key, img_path in tqdm(img_paths.items(), desc='Generating Captions'):
    caption = vqa_model.caption(img_path, caption_prompt)
    result[key] = {
        'img_fn':img_path,
        'caption':caption,
        'annot_id':key}

    with open(os.path.join(result_folder, '{}.yaml'.format(key)), 'w') as f:
        yaml.dump(result[key], f)


with open(os.path.join(result_folder, 'all_{}.yaml'.format(args.data_partition)), 'w') as f:
    yaml.dump(result, f)

