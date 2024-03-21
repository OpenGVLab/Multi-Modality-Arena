import os
import json
import argparse
import datetime

import torch
import numpy as np
import time as ttttt

from models import get_model
from utils.medicalqa import evaluate_medical_QA
from task_datasets.medical_datasets import MedicalDataset
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    # models
    parser.add_argument("--model_name", type=str, default="LLaMA-Adapter-v2")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)

    # datasets
    parser.add_argument("--sample-num", type=int, default=50)
    parser.add_argument("--sample-seed", type=int, default=20230719)
    parser.add_argument("--dataset_path", type=str, default='all')

    # result_path
    parser.add_argument("--answer_path", type=str, default="./final")

    args = parser.parse_args()
    return args


def main(args):
    sssss = ttttt.time()
    model = get_model(args.model_name, device=torch.device('cuda'))
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    answer_path = f"{args.answer_path}/{args.model_name}"

    
    dataset = MedicalDataset(args.dataset_path)
    metrics = evaluate_medical_QA(model=model, dataset=dataset, model_name=args.model_name, dataset_name=os.path.dirname(args.dataset_path).replace('/','_'), time=time, batch_size=args.batch_size, answer_path=answer_path)

    eeeee = ttttt.time()
    print(f"finish: {eeeee-sssss} s")


if __name__ == "__main__":
    args = parse_args()
    main(args)