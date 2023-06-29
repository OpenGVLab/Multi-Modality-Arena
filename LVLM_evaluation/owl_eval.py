import os
import json
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from models import get_model


class OwlDataset(Dataset):
    data_root = '/home/xupeng/workplace/OwlEval'

    def __init__(self):
        self.data = []
        with open(f"{self.data_root}/questions.jsonl", 'r') as f:
            for line in f.readlines():
                sample = json.loads(line)
                sample['image'] = f"{self.data_root}/cases/{sample['image']}"
                self.data.append(sample)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def name(self):
        return "OwlEval"


def evaluate_Owl(
    model,
    model_name,
    batch_size=1,
    answer_path='./answers'
):
    dataset = OwlDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
    predictions=[]
    for batch in tqdm(dataloader, desc="Running inference"):
        outputs = model.batch_generate(batch['image'], batch['question'])
        for i in range(len(outputs)):
            image = batch['image'][i].replace(f"{dataset.data_root}/cases/", '')
            answer_dict = {
                'image': image, 'question_id': batch['question_id'][i], 'question': batch['question'][i],
                'answer': outputs[i], 'model_id': model_name
            }
            predictions.append(answer_dict)
    os.makedirs(answer_path, exist_ok=True)
    answer_path = os.path.join(answer_path, f"{dataset.name}.json")
    with open(answer_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    
    # models
    parser.add_argument("--model_name", type=str, default="LLaMA-Adapter-v2")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    
    # result_path
    parser.add_argument("--answer_path", type=str, default="./answers")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    answer_path = f"{args.answer_path}/{args.model_name}"
    model = get_model(args.model_name, device=torch.device('cpu' if args.device == -1 else f"cuda:{args.device}"))
    evaluate_Owl(model, args.model_name, args.batch_size, answer_path=answer_path)
