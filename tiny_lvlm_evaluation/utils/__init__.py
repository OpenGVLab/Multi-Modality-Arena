from .vqa import evaluate_VQA


dataset_task_dict = {
    # classification
    'ImageNet': (evaluate_VQA, 'VQA'),
    'CIFAR10': (evaluate_VQA, 'VQA'),
    'OxfordIIITPet': (evaluate_VQA, 'VQA'),
    'Flowers102': (evaluate_VQA, 'VQA'),
    # OC, MCI
    'VCR1_OC': (evaluate_VQA, 'VQA'),
    'VCR1_MCI': (evaluate_VQA, 'VQA'),
    'MSCOCO_OC': (evaluate_VQA, 'VQA'),
    'MSCOCO_MCI': (evaluate_VQA, 'VQA'),
    # OCR
    "IIIT5K": (evaluate_VQA, 'VQA'),
    "IC13": (evaluate_VQA, 'VQA'),
    "IC15": (evaluate_VQA, 'VQA'),
    "Total-Text": (evaluate_VQA, 'VQA'),
    "CUTE80": (evaluate_VQA, 'VQA'),
    "SVT": (evaluate_VQA, 'VQA'),
    "SVTP": (evaluate_VQA, 'VQA'),
    "COCO-Text": (evaluate_VQA, 'VQA'),
    "WordArt": (evaluate_VQA, 'VQA'),
    "CTW": (evaluate_VQA, 'VQA'),
    "HOST": (evaluate_VQA, 'VQA'),
    "WOST": (evaluate_VQA, 'VQA'),
    # KIE Datasets
    'SROIE': (evaluate_VQA, 'VQA'),
    'FUNSD': (evaluate_VQA, 'VQA'),
    # VQA Datasets
    'DocVQA': (evaluate_VQA, 'VQA'),
    'TextVQA': (evaluate_VQA, 'VQA'),
    'STVQA': (evaluate_VQA, 'VQA'),
    'OCRVQA': (evaluate_VQA, 'VQA'),
    'OKVQA': (evaluate_VQA, 'VQA'),
    'GQA': (evaluate_VQA, 'VQA'),
    'IconQA': (evaluate_VQA, 'VQA'),
    'VSR': (evaluate_VQA, 'VQA'),
    'WHOOPS': (evaluate_VQA, 'VQA'),
    'ScienceQA': (evaluate_VQA, 'VQA'),
    'VizWiz': (evaluate_VQA, 'VQA'),
    # ImageNetVC
    'ImageNetVC_color': (evaluate_VQA, 'VQA'),
    'ImageNetVC_shape': (evaluate_VQA, 'VQA'),
    'ImageNetVC_material': (evaluate_VQA, 'VQA'),
    'ImageNetVC_component': (evaluate_VQA, 'VQA'),
    'ImageNetVC_others': (evaluate_VQA, 'VQA'),
    # Object Hallucination
    'MSCOCO_pope_random': (evaluate_VQA, 'VQA'),
    'MSCOCO_pope_popular': (evaluate_VQA, 'VQA'),
    'MSCOCO_pope_adversarial': (evaluate_VQA, 'VQA'),
    # Embodied Datasets
    "MetaWorld": (evaluate_VQA, 'Embodied'),
    "FrankaKitchen": (evaluate_VQA, 'Embodied'),
    "Minecraft": (evaluate_VQA, 'Embodied'),
    "VirtualHome": (evaluate_VQA, 'Embodied'),
    "MinecraftPolicy": (evaluate_VQA, 'Embodied'),
}