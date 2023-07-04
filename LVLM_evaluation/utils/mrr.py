import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

from .tools import VQAEval


def evaluate_MRR(
    model,
    dataset,
    model_name,
    dataset_name,
    time,
    batch_size=1,
    answer_path='./answers',
    max_new_tokens=256
):
    predictions=[]
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
    for batch in tqdm(dataloader, desc="Running inference"):
        outputs = model.batch_generate(batch['image_path'], batch['question'], max_new_tokens=max_new_tokens)
        for image_path, question, gt_answer, output in zip(batch['image_path'], batch['question'], batch['gt_answers'], outputs):
            answer_dict={'question': question, 'answer': output,
            'gt_answers': gt_answer, 'image_path': image_path,
            'model_name': model_name}
            predictions.append(answer_dict)
    
    eval = VQAEval()
    mrr_list = []
    for i in range(len(predictions)):
        gt_answers = predictions[i]['gt_answers']
        answer = predictions[i]['answer']
        mrr = eval.evaluate_MRR(answer, gt_answers)
        predictions[i]['MRR'] = mrr
        mrr_list.append(mrr)
    
    answer_dir = os.path.join(answer_path, time)
    os.makedirs(answer_dir, exist_ok=True)
    answer_path = os.path.join(answer_dir, f"{dataset_name}.json")
    with open(answer_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))

    dataset_mrr = sum(mrr_list) / len(mrr_list)
    print(f'{dataset_name}: MRR-{dataset_mrr}')
    return dataset_mrr