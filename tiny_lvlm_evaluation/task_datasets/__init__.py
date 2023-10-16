DATA_DIR = '/nvme/share/xupeng/datasets' # '.', '/nvme/share/leimeng/datasets'

import json
from functools import partial
from torch.utils.data import Dataset

from .ocr_datasets import ocrDataset
from .embod_datasets import EmbodiedDataset
from .kie_datasets import SROIEDataset, FUNSDDataset
from .cls_datasets import ImageNetDataset, CIFAR10Dataset, OxfordIIITPet, Flowers102
from .vqa_datasets import (
    VCR_OCDataset, VCR_MCIDataset, MSCOCO_OCDataset, MSCOCO_MCIDataset,
    DocVQADataset, TextVQADataset, STVQADataset, OCRVQADataset,
    OKVQADataset, GQADataset, IconQADataset, VSRDataset,
    WHOOPSDataset, ScienceQADataset, VizWizDataset, ImageNetVC,
    MSCOCO_POPEDataset_random, MSCOCO_POPEDataset_popular, MSCOCO_POPEDataset_adversarial
)


class GeneralDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        root='tiny_lvlm_datasets'
    ):
        self.root = root
        self.dataset_name = dataset_name
        with open(f"{root}/{dataset_name}/dataset.json", 'r') as f:
            self.dataset = json.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample['image_path'] = f"{self.root}/{self.dataset_name}/{sample['image_path']}"
        return sample


dataset_class_dict = {
    # NOTE: Visual Perception (400)
    # classification
    'ImageNet': ImageNetDataset,
    'CIFAR10': CIFAR10Dataset,
    'OxfordIIITPet': OxfordIIITPet,
    'Flowers102': Flowers102,
    # OC, MCI
    'VCR1_OC': VCR_OCDataset,
    'VCR1_MCI': VCR_MCIDataset,
    'MSCOCO_OC': MSCOCO_OCDataset,
    'MSCOCO_MCI': MSCOCO_MCIDataset,

    # NOTE: Visual Knowledge Acquisition (700)
    # OCR
    "IIIT5K": partial(ocrDataset, dataset_name="IIIT5K"),
    "IC13": partial(ocrDataset, dataset_name="IC13"),
    "IC15": partial(ocrDataset, dataset_name="IC15"),
    "Total-Text": partial(ocrDataset, dataset_name="Total-Text"),
    "CUTE80": partial(ocrDataset, dataset_name="CUTE80"),
    "SVT": partial(ocrDataset, dataset_name="SVT"),
    "SVTP": partial(ocrDataset, dataset_name="SVTP"),
    "COCO-Text": partial(ocrDataset, dataset_name="COCO-Text"),
    "WordArt": partial(ocrDataset, dataset_name="WordArt"),
    "CTW": partial(ocrDataset, dataset_name="CTW"),
    "HOST": partial(ocrDataset, dataset_name="HOST"),
    "WOST": partial(ocrDataset, dataset_name="WOST"),
    # KIE Datasets
    'SROIE': SROIEDataset,
    'FUNSD': FUNSDDataset,

    # NOTE: Visual Reasoning (550)
    # VQA Datasets
    'DocVQA': DocVQADataset,
    'TextVQA': TextVQADataset,
    'STVQA': STVQADataset,
    'OCRVQA': OCRVQADataset,
    'OKVQA': OKVQADataset,
    'GQA': GQADataset,
    'IconQA': IconQADataset,
    'VSR': VSRDataset,
    'WHOOPS': WHOOPSDataset,
    'ScienceQA': ScienceQADataset,
    'VizWiz': VizWizDataset,

    # NOTE: Visual Commonsense (250) -> 500
    # ImageNetVC
    'ImageNetVC_color': partial(ImageNetVC, task='color'),
    'ImageNetVC_shape': partial(ImageNetVC, task='shape'),
    'ImageNetVC_material': partial(ImageNetVC, task='material'),
    'ImageNetVC_component': partial(ImageNetVC, task='component'),
    'ImageNetVC_others': partial(ImageNetVC, task='others'),

    # NOTE: Object Hallucination (150) -> 300
    # Object Hallucination
    'MSCOCO_pope_random': MSCOCO_POPEDataset_random,
    'MSCOCO_pope_popular': MSCOCO_POPEDataset_popular,
    'MSCOCO_pope_adversarial': MSCOCO_POPEDataset_adversarial,

    # Embodied Datasets
    "MetaWorld": partial(EmbodiedDataset, dataset_name="MetaWorld"),
    "FrankaKitchen": partial(EmbodiedDataset, dataset_name="FrankaKitchen"),
    "Minecraft": partial(EmbodiedDataset, dataset_name="Minecraft"),
    "VirtualHome": partial(EmbodiedDataset, dataset_name="VirtualHome"),
    "MinecraftPolicy": partial(EmbodiedDataset, dataset_name="MinecraftPolicy"),
}
