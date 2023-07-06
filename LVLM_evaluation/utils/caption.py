import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

from .cider import CiderScorer

"""
NOTE: caption prompt candidates
1. what is described in the image?
2. Generate caption of this image:
"""


def evaluate_Caption(
    model,
    dataset,
    model_name,
    dataset_name,
    time,
    batch_size=1,
    answer_path='./answers',
    question='what is described in the image?',
    max_new_tokens=16,
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
    
    with open(answer_path, 'r') as f:
        dict = json.load(f)
        cider_scorer = CiderScorer(n=4, sigma=6.0)
        for i in range(len(dict)):
            gt_answers = dict[i]['gt_answers']
            answer = dict[i]['answer']
            cider_scorer += (answer, gt_answers)
        (score, scores) = cider_scorer.compute_score()
    for i, sample_score in zip(range(len(dict)), scores):
        dict[i]['cider_score'] = sample_score
    with open(answer_path, "w") as f:
        f.write(json.dumps(dict, indent=4))
    
    print(f'{dataset_name}: {score}')
    return score