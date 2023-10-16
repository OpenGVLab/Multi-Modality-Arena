import os
import json
import shutil
import argparse

import torch
import numpy as np
from PIL import Image

from task_datasets import dataset_class_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    # datasets
    parser.add_argument("--dataset-names", type=str, default=None)
    parser.add_argument("--sample-num", type=int, default=50)
    parser.add_argument("--sample-seed", type=int, default=20230719)
    parser.add_argument("--saved-dir", type=str, default='tiny_lvlm_datasets')

    args = parser.parse_args()
    return args


def sample_dataset(dataset, dataset_name, max_sample_num=50, seed=0):
    sampled_indices = 'bard_RandomSeed20230719_50samples_per_dataset.jsonl'
    with open(sampled_indices, 'r') as f:
        dataset_indices = [json.loads(x) for x in f.readlines()]
    new_dataset_indices = {x['dataset']: x['indices'] for x in dataset_indices}
    
    np.random.seed(seed)
    if dataset_name in new_dataset_indices:
        selected_indices = new_dataset_indices[dataset_name]
        selected_indices = selected_indices
        selected_indices = [x for x in range(max(selected_indices) + 1, max(selected_indices) + 1 + 100)]
        max_sample_num = max(selected_indices) + 1
        random_indices = np.random.choice(
            len(dataset), max_sample_num, replace=False
        )
        random_indices = random_indices[selected_indices]
    elif dataset_name in ["MetaWorld", "FrankaKitchen", "Minecraft", "VirtualHome", "MinecraftPolicy"]:
        random_indices = [x for x in range(len(dataset))]
    else:
        random_indices = np.random.choice(
            len(dataset), max_sample_num, replace=False
        )
    
    print(f'The length of {dataset_name} is {len(random_indices)}')
    dataset = torch.utils.data.Subset(dataset, random_indices)
    return dataset


def save_sampled_dataset(dataset, dataset_name, saved_dir):
    sampled_dataset = []
    sampled_dataset_path = f'{saved_dir}/{dataset_name}'
    os.makedirs(sampled_dataset_path, exist_ok=True)

    for i in range(len(dataset)):
        sample = dataset[i]
        old_image_path = sample['image_path']
        if type(old_image_path) is str:
            if dataset_name == 'IconQA':
                image_name = f"{i:02d}.png"
            else:
                image_name = old_image_path.split('/')[-1]
            shutil.copy(old_image_path, f"{sampled_dataset_path}/{image_name}")
        else:
            image = Image.fromarray(old_image_path)
            image_name = f"{i:02d}.png"
            image.save(f"{sampled_dataset_path}/{image_name}")
        sample['image_path'] = image_name
        sampled_dataset.append(sample)
    
    with open(f"{sampled_dataset_path}/dataset.json", 'w') as f:
        f.write(json.dumps(sampled_dataset, indent=4))

    return dataset


def main(args):
    dataset_names = args.dataset_names.split(',') if args.dataset_names is not None else list(dataset_class_dict.keys())
    for dataset_name in dataset_names:
        dataset = dataset_class_dict[dataset_name]()
        dataset = sample_dataset(dataset, dataset_name, args.sample_num, args.sample_seed)
        save_sampled_dataset(dataset, dataset_name, args.saved_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)