import os
import json
from torch.utils.data import Dataset

from . import DATA_DIR


class EmbodiedDataset(Dataset):
    data_root = f"{DATA_DIR}/Embodied_Datasets" # '/home/huangsiyuan/holistic_evaluation/EmbodiedEvaluation'
    dataset_list = ["MetaWorld", "FrankaKitchen", "Minecraft", "VirtualHome", "MinecraftPolicy"]
    
    def __init__(self, dataset_name):
        assert dataset_name in self.dataset_list, f"{dataset_name} not in the list"
        self.dataset_root = os.path.join(self.data_root, dataset_name)
        self.dataset = self.prepare_dataset()

    def prepare_dataset(self):
        dataset_file = os.path.join(self.dataset_root, 'val_dataset.json')
        if os.path.exists(dataset_file):
            dataset = json.load(open(dataset_file, 'r'))
        else:
            # for quick implementation, we only use the pre-created val dataset
            raise RuntimeError(f'Dataset {dataset_file} not found')
        
        prefix_prompt = None
        prefix_prompt_file = os.path.join(self.dataset_root, 'prefix_prompt.txt')
        if os.path.exists(prefix_prompt_file):
            # load prompt and merge into one string
            prefix_prompt = open(prefix_prompt_file, 'r').read().strip()

        if prefix_prompt is not None:
            if "INSERT HERE" in prefix_prompt:
                # replace INSERT HERE with the question
                for i in range(len(dataset)):
                    dataset[i]['question'] = prefix_prompt.replace("{INSERT HERE}", dataset[i]['question'])
            else:
                for i in range(len(dataset)):
                    dataset[i]['question'] = prefix_prompt + ' ' + dataset[i]['question']

        return dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img_path = self.dataset[idx]['image_path']
        full_img_path = os.path.join(self.dataset_root, img_path)
        question = self.dataset[idx]['question']
        answers = "BLANK"
        return {
            "image_path": full_img_path,
            "question": question,
            "gt_answers": answers}