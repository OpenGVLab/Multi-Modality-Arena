import os
import json
import argparse
import datetime
from functools import partial

import torch
import numpy as np

from utils import evaluate_OCR, evaluate_VQA, evaluate_Caption, evaluate_KIE, evaluate_MRR, evaluate_embodied, evaluate_zero_shot_image_classification
from task_datasets import ocrDataset, dataset_class_dict
from models import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    # models
    parser.add_argument("--model_name", type=str, default="LLaMA-Adapter-v2")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=-1)

    # datasets
    parser.add_argument("--ocr_dataset_name", type=str, default="IIIT5K SVT IC13 IC15 SVTP CUTE80 COCO-Text Total-Text WordArt CTW HOST WOST")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--sample_num", type=int, default=-1)
    parser.add_argument("--sample_seed", type=int, default=0)

    # result_path
    parser.add_argument("--answer_path", type=str, default="./answers")

    # eval choices
    parser.add_argument("--eval_ocr", action="store_true", help="Whether to evaluate on ocr.")
    parser.add_argument("--eval_vqa", action="store_true", help="Whether to evaluate on vqa.")
    parser.add_argument("--eval_caption", action="store_true", help="Whether to evaluate on caption.")
    parser.add_argument("--eval_kie", action="store_true", default=False, help="Whether to evaluate on kie.")
    parser.add_argument("--eval_mrr", action="store_true", default=False, help="Whether to evaluate on mrr.")
    parser.add_argument("--eval_embod", action="store_true", default=False, help="Whether to evaluate on embodied.")
    parser.add_argument("--eval_cls", action="store_true", default=False, help="Whether to evaluate on zero-shot classification.")

    args = parser.parse_args()
    return args


def sample_dataset(dataset, max_sample_num=5000, seed=0):
    if max_sample_num == -1:
        return dataset

    if len(dataset) > max_sample_num:
        np.random.seed(seed)
        random_indices = np.random.choice(
            len(dataset), max_sample_num, replace=False
        )
        dataset = torch.utils.data.Subset(dataset, random_indices)
    return dataset


def get_eval_function(args):
    if args.eval_vqa:
        eval_func = evaluate_VQA
    elif args.eval_caption:
        eval_func = evaluate_Caption
    elif args.eval_kie:
        eval_func = evaluate_KIE
    elif args.eval_mrr:
        eval_func = evaluate_MRR
    elif args.eval_embod:
        eval_func = evaluate_embodied
    elif args.eval_cls:
        eval_func = evaluate_zero_shot_image_classification
    else:
        raise NotImplementedError("Invalid choice of evaluation function")

    if args.max_new_tokens == -1:
        return eval_func
    else:
        return partial(eval_func, max_new_tokens=args.max_new_tokens)


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    model = get_model(args.model_name, device=torch.device('cuda'))
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    answer_path = f"{args.answer_path}/{args.model_name}"

    result = {}
    if args.eval_ocr:
        ocr_dataset_name = args.ocr_dataset_name.split()
        for i in range(len(ocr_dataset_name)):
            dataset = ocrDataset(ocr_dataset_name[i])
            dataset = sample_dataset(dataset, args.sample_num, args.sample_seed)
            metrics = evaluate_OCR(model, dataset, args.model_name, ocr_dataset_name[i], time, args.batch_size, answer_path)
            result[ocr_dataset_name[i]] = metrics

    eval_function = get_eval_function(args)
    if eval_function is not None:
        dataset = dataset_class_dict[args.dataset_name]()
        dataset = sample_dataset(dataset, args.sample_num, args.sample_seed)
        metrics = eval_function(model, dataset, args.model_name, args.dataset_name, time, args.batch_size, answer_path=answer_path)
        result[args.dataset_name] = metrics
    
    result_path = os.path.join(os.path.join(answer_path, time), 'result.json')
    with open(result_path, "w") as f:
        f.write(json.dumps(result, indent=4))


if __name__ == "__main__":
    args = parse_args()
    main(args)