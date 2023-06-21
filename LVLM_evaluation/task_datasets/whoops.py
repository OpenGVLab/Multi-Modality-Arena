import os
import json
from torch.utils.data import Dataset


class WHOOPSCaptionDataset(Dataset):
    def __init__(
        self,
        root: str='datasets/whoops',
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = f'{root}/whoops_images'
        self.anno_path = f'{root}/whoops_captions.json'
        self.annotation = json.load(open(self.anno_path, "r"))

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index: int):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        answers = ann['caption']

        return {
            "image_path": image_path,
            "gt_answers": answers,
        }


class WHOOPSVQADataset(Dataset):
    def __init__(
        self,
        root: str='datasets/whoops',
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = f'{root}/whoops_images'
        self.anno_path = f'{root}/whoops_vqa_pairs.json'
        self.annotation = json.load(open(self.anno_path, "r"))

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["image"])
        answers = ann['reference']
        question = ann['question']

        return {
            "image_path": image_path,
            "question": question,
            "gt_answers": answers}


class WHOOPSWeirdDataset(Dataset):
    def __init__(
        self,
        root: str='datasets/whoops',
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = f'{root}/whoops_images'
        self.anno_path = f'{root}/whoops_explanation_of_violation.json'
        self.annotation = json.load(open(self.anno_path, "r"))

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index: int):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        # answers = {
        #     'designer_explanation': ann['designer_explanation'],
        #     'crowd_explanations': ann['crowd_explanations'],
        # }
        answers = [ann['designer_explanation']] + ann['crowd_explanations']

        return {
            "image_path": image_path,
            "gt_answers": answers,
        }