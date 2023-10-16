import os
import json
import argparse
import datetime

import torch
from models import get_model
from utils import evaluate_VQA
from task_datasets import GeneralDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model-name", type=str, default="LLaMA-Adapter-v2")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sampled-root", type=str, default='updated_datasets')
    parser.add_argument("--answer-path", type=str, default="./tiny_answers")

    args = parser.parse_args()
    return args


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    model = get_model(args.model_name, device=torch.device('cuda'))
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    answer_path = f"{args.answer_path}/{args.model_name}"

    result = {}
    overall_score = 0
    dataset_names = ['Visual_Reasoning', 'Visual_Perception', 'Visual_Knowledge_Acquisition', 'Visual_Commonsense', 'Object_Hallucination']
    # Visual_Commonsense, Object_Hallucination
    for dataset_name in dataset_names:
        dataset = GeneralDataset(dataset_name, root=args.sampled_root)
        metrics = evaluate_VQA(model, dataset, args.model_name, dataset_name, 'VQA', time, args.batch_size, answer_path=answer_path)
        result[dataset_name] = metrics
        overall_score += metrics
    result['Overall_Score'] = overall_score
    print(f"Overall Score: {overall_score}")

    result_path = os.path.join(os.path.join(answer_path, time), 'result.json')
    with open(result_path, "w") as f:
        f.write(json.dumps(result, indent=4))


if __name__ == "__main__":
    args = parse_args()
    main(args)