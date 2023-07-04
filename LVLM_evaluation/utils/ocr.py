import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

from .tools import has_word, remove_special_chars


def evaluate_OCR(
    model,
    dataset,
    model_name,
    dataset_name,
    time,
    batch_size=1,
    answer_path='./answers',
    question='what is written in the image?',
    max_new_tokens=256
):
    predictions=[]
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
    for batch in tqdm(dataloader, desc="Running inference"):
        outputs = model.batch_generate(batch['image_path'], [question for _ in range(len(batch['image_path']))], max_new_tokens=max_new_tokens)
        for image_path, gt_answer, output in zip(batch['image_path'], batch['gt_answers'], outputs):
            answer_dict={'question': question, 'answer': output,
            'gt_answers': gt_answer, 'image_path': image_path,
            'model_name': model_name}
            predictions.append(answer_dict)
    answer_dir = os.path.join(answer_path, time)
    os.makedirs(answer_dir, exist_ok=True)
    answer_path = os.path.join(answer_dir, f"{dataset_name}.json")
    with open(answer_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))
    correct = 0
    num = 0
    with open(answer_path, 'r') as f:
        dict = json.load(f)
        for i in range(len(dict)):
            gt_answers = dict[i]['gt_answers']
            answer = dict[i]['answer']
            gt_answers = remove_special_chars(gt_answers).lower()
            answer = remove_special_chars(answer).lower()
            if has_word(answer, gt_answers):
                correct+=1
            num+=1
    print(f'{dataset_name}:{float(correct)/num}')
    return float(correct)/num