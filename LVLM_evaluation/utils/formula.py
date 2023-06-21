import more_itertools
from tqdm import tqdm
import os
import json
import re


def evaluate_Formula(
    model,
    dataset,
    model_name,
    dataset_name,
    time,
    question='Please write out the expression of the formula in the image using LaTeX format.',
    batch_size=1,
    answer_path='./answers'
):
    #Please write out the expression of the formula in the image using LaTeX format.
    predictions=[]
    for batch in more_itertools.chunked(
        tqdm(dataset, desc="Running inference"), batch_size
    ):
        batch = batch[0]
        output = model.generate(image=batch['image_path'], question=question)
        answer_dict={'question':question, 'answer':output, 
        'gt_answers':batch['gt_answers'], 'image_path':batch['image_path'],
        'model_name':model_name}
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
            gt_answers = re.sub(r'\s+', '', dict[i]['gt_answers'])
            answer = re.sub(r'\s+', '', dict[i]['answer'])
            if gt_answers in answer:
                correct+=1
            num+=1
    print(f'{dataset_name}:{float(correct)/num}')
    return float(correct)/num