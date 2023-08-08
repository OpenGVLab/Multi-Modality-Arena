import os
import json
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

from . import DATA_DIR


class VCR_OCDataset(Dataset):
    def __init__(
        self,
        image_dir_path=f"{DATA_DIR}/VCR/vcr1images",
        ann_path= "utils_data/GVT/vcr1_oc.json"
    ):
        self.data = json.load(open(ann_path, "r"))
        self.image_dir_path = image_dir_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['text_in']
        answers = self.data[idx]['text_out']
        img_path = os.path.join(self.image_dir_path, str(self.data[idx]['image_id']))
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}


class VCR_MCIDataset(Dataset):
    def __init__(
        self,
        image_dir_path=f"{DATA_DIR}/VCR/vcr1images",
        ann_path= "utils_data/GVT/vcr1_mci.json"
    ):
        self.data = json.load(open(ann_path, "r"))
        self.image_dir_path = image_dir_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['text_in']
        answers = self.data[idx]['text_out']
        question = (
            f'Question: {question}\n\n'
            'Choose the single most likely answer from the following choices <choice>:\n- Yes\n- No\n\n'
            'The output format follows exactly as below:\nAnswer: <choice>'
        )

        img_path = os.path.join(self.image_dir_path,str(self.data[idx]['image_id']))
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}


class MSCOCO_OCDataset(Dataset):
    def __init__(
        self,
        image_dir_path=f"{DATA_DIR}/MSCOCO/val2014",
        ann_path="utils_data/GVT/coco_oc.json"
    ):
        self.data = json.load(open(ann_path, "r"))
        self.image_dir_path = image_dir_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['text_in']
        answers = self.data[idx]['text_out']
        name = 'COCO_val2014_' + str(self.data[idx]['image_id']).zfill(len('000000007991')) + '.jpg'
        img_path = os.path.join(self.image_dir_path, name)
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}


class MSCOCO_MCIDataset(Dataset):
    def __init__(
        self,
        image_dir_path=f"{DATA_DIR}/MSCOCO/val2014",
        ann_path= "utils_data/GVT/coco_mci.json"
    ):
        self.data = json.load(open(ann_path, "r"))
        self.image_dir_path = image_dir_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['text_in']
        answers = self.data[idx]['text_out']
        question = (
            f'Question: {question}\n\n'
            'Choose the single most likely answer from the following choices <choice>:\n- Yes\n- No\n\n'
            'The output format follows exactly as below:\nAnswer: <choice>'
        )
        name = 'COCO_val2014_' + str(self.data[idx]['image_id']).zfill(len('000000007991')) + '.jpg'
        img_path = os.path.join(self.image_dir_path,name)
        if os.path.isfile(img_path):
            return {
                "image_path": img_path,
                "question": question,
                "gt_answers": answers}
        else:
            print(img_path, 'not exist!!!')
            return self.__getitem__((idx + 1) % len(self))


class DocVQADataset(Dataset):
    data_root = f'{DATA_DIR}/VQA_Datasets/DocVQA/val'

    def __init__(self):
        ann_path = f"{self.data_root}/val_v1.0.json"
        self.data = json.load(open(ann_path, "r"))["data"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['question']
        answers = self.data[idx]['answers']
        img_path = os.path.join(self.data_root, self.data[idx]['image'])
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}


class TextVQADataset(Dataset):
    data_root = f"{DATA_DIR}/VQA_Datasets/TextVQA"

    def __init__(self):
        self.data = json.load(open(f"{self.data_root}/TextVQA_0.5.1_val.json", "r"))["data"]
        self.image_dir_path = self.data_root + '/train_images'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['question']
        answers = self.data[idx]['answers']
        img_path = os.path.join(self.image_dir_path, f"{self.data[idx]['image_id']}.jpg")
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}


class STVQADataset(Dataset):
    data_root = f"{DATA_DIR}/VQA_Datasets/STVQA"

    def __init__(self):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        data = json.load(open(f"{self.data_root}/train_task_3.json", "r"))['data']
        for i in range(len(data)):
            image_path = self.data_root + '/train_imgs/' + data[i]['dataset'] + '/' + data[i]['file_name']
            self.image_list.append(image_path)
            self.answer_list.append(data[i]['answers'])
            self.question_list.append(data[i]['question'])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}


class OCRVQADataset(Dataset):
    data_root = f'{DATA_DIR}/VQA_Datasets/OCRVQA'

    def __init__(self):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        dataset = json.load(open(f'{self.data_root}/dataset.json', "r"))
        for idx, data in enumerate(dataset):
            if dataset[data]['split'] != 2:
                continue
            questions =  dataset[data]['questions']
            for index, question in enumerate(questions):
                img_name = dataset[data]['imageURL'].split('/')[-1]
                image_file = os.path.join(self.data_root, 'images', img_name)
                gt_answers = dataset[data]['answers'][index]
                self.image_list.append(image_file)
                self.answer_list.append(gt_answers)
                self.question_list.append(question)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}


class OKVQADataset(Dataset):
    data_root = f"{DATA_DIR}/VQA_Datasets/OKVQA"

    def __init__(self):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        questions = json.load(open(f"{self.data_root}/OpenEnded_mscoco_val2014_questions.json", "r"))['questions']
        question_dict = {x['question_id']: x['question'] for x in questions}
        annotations = json.load(open(f"{self.data_root}/mscoco_val2014_annotations.json", "r"))['annotations']
        for i in range(len(annotations)):
            question = question_dict[annotations[i]['question_id']]
            answers = [x['answer'] for x in annotations[i]['answers']]
            image_path = f"{self.data_root}/val2014/COCO_val2014_000000{annotations[i]['image_id']:06d}.jpg"
            self.answer_list.append(answers)
            self.image_list.append(image_path)
            self.question_list.append(question)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}


class GQADataset(Dataset):
    data_root = f"{DATA_DIR}/VQA_Datasets/GQA"

    def __init__(self):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        annotations = json.load(open(f"{self.data_root}/questions/testdev_balanced_questions.json", "r"))
        for sample in annotations:
            sample = annotations[sample]
            image_path = f"{self.data_root}/images/{sample['imageId']}.jpg"
            self.image_list.append(image_path)
            self.answer_list.append(sample['answer'])
            self.question_list.append(sample['question'])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}


class IconQADataset(Dataset):
    split='test'
    data_root = f'{DATA_DIR}/VQA_Datasets/IconQA'

    def __init__(self):
        self.image_list = []
        self.question_list = []
        self.answer_list = []

        dataset_dir = f"{self.data_root}/dataset/{self.split}/choose_txt"
        for sample in os.listdir(dataset_dir):
            image_path = f"{dataset_dir}/{sample}/image.png"
            self.image_list.append(image_path)
            data = json.load(open(f"{dataset_dir}/{sample}/data.json", 'r'))
            question = f"Question: {data['question']}\n" \
                       f"Options: {' '.join(data['choices'])}\n"
            self.question_list.append(question)
            self.answer_list.append(data['choices'][data['answer']])

    def __len__(self):
        return len(self.question_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}


class VSRDataset(Dataset):
    data_root = f"{DATA_DIR}/VQA_Datasets/VSR"
    choices = ['No', 'Yes']

    def __init__(self):
        self.image_list = []
        self.question_list = []
        self.answer_list = []

        data = []
        with open(f"{self.data_root}/all_vsr_validated_data.jsonl", "r") as f:
            for line in f.readlines():
                data.append(json.loads(line))
        for sample in data:
            image_path = f"{self.data_root}/images/{sample['image']}"
            caption = sample['caption']
            question = (
                f'Question: Is the caption "{caption}" correctly describing the image?\n\n'
                'Choose the single most likely answer from the following choices <choice>:\n- Yes\n- No\n\n'
                'The output format follows exactly as below:\nAnswer: <choice>'
            )
            answer = self.choices[sample['label']]
            self.answer_list.append(answer)
            self.image_list.append(image_path)
            self.question_list.append(question)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}


class WHOOPSDataset(Dataset):
    def __init__(
        self,
        root: str=f'{DATA_DIR}/WHOOPS',
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


class ScienceQADataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data_root = f'{DATA_DIR}/VQA_Datasets/ScienceQA'
        self.dataset = []
        jsonl_path = f'{self.data_root}/annos_test_image_2017samples.jsonl'
        with open(jsonl_path, 'r') as f:
            self.dataset = [json.loads(x) for x in f.readlines()]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> dict:
        sample = self.dataset[index]
        image_path = os.path.join(self.data_root, sample['image_path'])
        return {
            "image_path": image_path,
            "question": sample['question'],
            "gt_answers": sample['answer'],
        }


class VizWizDataset(Dataset):
    data_root = f"{DATA_DIR}/VQA_Datasets/VizWiz"

    def __init__(self):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        self.load_data(split='val')
        # self.load_data(split='train')

    def load_data(self, split='val'):
        annotations = json.load(open(f"{self.data_root}/{split}_grounding.json", "r"))
        for image_name in annotations:
            sample = annotations[image_name]
            image_path = f"{self.data_root}/{split}/{image_name}"
            self.image_list.append(image_path)
            self.answer_list.append(sample['answers'])
            self.question_list.append(sample['question'])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}


class ImageNetVC(Dataset):
    def __init__(
        self, task: str='shape', root: str=f'{DATA_DIR}/ImageNetVC'
    ) -> None:
        super().__init__()
        csv_path = os.path.join(root, f'{task}.csv')
        wid2label_path = os.path.join(root, 'ImageNet_mapping.txt')
        label2wid = {}
        with open(wid2label_path, 'r') as f:
            for line in f.readlines():
                wid = line[:9]
                for x in line[9:].split(','):
                    label2wid[x.strip()] = wid
        # category,question,answer
        annos = pd.read_csv(csv_path)
        img_dir = os.path.join(root, 'images')
        self.data = []
        for i, row in annos.iterrows():
            label = row['category']
            question = row['question']
            answer = row['answer']
            wid = label2wid[label]
            for image_path in sorted(list(Path(os.path.join(img_dir, wid)).glob('*'))):
                sample = {
                    'image_path': str(image_path),
                    'question': question,
                    'answer': answer
                }
                self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> dict:
        sample = self.data[index]
        return {
            'image_path': sample['image_path'],
            'question': sample['question'],
            'gt_answers': sample['answer'],
        }


class MSCOCO_POPEDataset_random(Dataset):
    def __init__(
        self,
        image_dir_path=f"{DATA_DIR}/MSCOCO/val2014",
        ann_path= "utils_data/MSCOCO_POPE/coco_pope_random1.json"
    ):
        self.data = json.load(open(ann_path, "r"))
        self.image_dir_path = image_dir_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['text']
        answers = self.data[idx]['label']
        name = str(self.data[idx]['image'])
        img_path = os.path.join(self.image_dir_path,name)
        if os.path.isfile(img_path):
            return {
                "image_path": img_path,
                "question": question,
                "gt_answers": answers}
        else:
            print(img_path, 'not exist!!!')
            return self.__getitem__((idx + 1) % len(self))


class MSCOCO_POPEDataset_popular(Dataset):
    def __init__(
        self,
        image_dir_path=f"{DATA_DIR}/MSCOCO/val2014",
        ann_path= "utils_data/MSCOCO_POPE/coco_pope_popular1.json"
    ):
        self.data = json.load(open(ann_path, "r"))
        self.image_dir_path = image_dir_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['text']
        answers = self.data[idx]['label']
        name = str(self.data[idx]['image'])
        img_path = os.path.join(self.image_dir_path,name)
        if os.path.isfile(img_path):
            return {
                "image_path": img_path,
                "question": question,
                "gt_answers": answers}
        else:
            print(img_path, 'not exist!!!')
            return self.__getitem__((idx + 1) % len(self))


class MSCOCO_POPEDataset_adversarial(Dataset):
    def __init__(
        self,
        image_dir_path=f"{DATA_DIR}/MSCOCO/val2014",
        ann_path= "utils_data/MSCOCO_POPE/coco_pope_adversarial1.json"
    ):
        self.data = json.load(open(ann_path, "r"))
        self.image_dir_path = image_dir_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['text']
        answers = self.data[idx]['label']
        name = str(self.data[idx]['image'])
        img_path = os.path.join(self.image_dir_path,name)
        if os.path.isfile(img_path):
            return {
                "image_path": img_path,
                "question": question,
                "gt_answers": answers}
        else:
            print(img_path, 'not exist!!!')
            return self.__getitem__((idx + 1) % len(self))
