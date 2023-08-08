import os
import json
import argparse
import datetime

import torch
import numpy as np

from models import get_model
from utils import dataset_task_dict
from task_datasets import dataset_class_dict, GeneralDataset
from sample_dataset import sample_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    # models
    parser.add_argument("--model-name", type=str, default="LLaMA-Adapter-v2")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)

    # datasets
    parser.add_argument("--dataset-names", type=str, default=None)
    parser.add_argument("--sample-num", type=int, default=50)
    parser.add_argument("--sample-seed", type=int, default=20230719)
    parser.add_argument("--use-sampled", action='store_false')
    parser.add_argument("--sampled-root", type=str, default='tiny_lvlm_datasets')

    # result_path
    parser.add_argument("--answer_path", type=str, default="./tiny_answers")

    args = parser.parse_args()
    return args


def eval_sample_dataset(dataset, dataset_name, max_sample_num=50, seed=0):
    if max_sample_num == -1:
        return dataset
    return sample_dataset(dataset, dataset_name, max_sample_num, seed)


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    model = get_model(args.model_name, device=torch.device('cuda'))
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    answer_path = f"{args.answer_path}/{args.model_name}"

    result = {}
    dataset_names = args.dataset_names.split(',') if args.dataset_names is not None else list(dataset_class_dict.keys())
    for dataset_name in dataset_names:
        eval_function, task_type = dataset_task_dict[dataset_name]
        if args.use_sampled:
            dataset = GeneralDataset(dataset_name, root=args.sampled_root)
        else:
            dataset = dataset_class_dict[dataset_name]()
            dataset = eval_sample_dataset(dataset, dataset_name, args.sample_num, args.sample_seed)
        metrics = eval_function(model, dataset, args.model_name, dataset_name, task_type, time, args.batch_size, answer_path=answer_path)
        result[dataset_name] = metrics

    result_path = os.path.join(os.path.join(answer_path, time), 'result.json')
    with open(result_path, "w") as f:
        f.write(json.dumps(result, indent=4))


if __name__ == "__main__":
    args = parse_args()
    main(args)