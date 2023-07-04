import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader


# the eval is more similar to VQA
def evaluate_embodied(
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
        outputs = model.generate(batch['image_path'][-1], batch['question'][-1], max_new_tokens=max_new_tokens)
        for image_path, gt_answer, question, output in zip(batch['image_path'], batch['gt_answers'], batch["question"], [outputs]):
            answer_dict={'question': question, 'answer': output,
            'gt_answers': gt_answer, 'image_path': image_path,
            'model_name': model_name}
            predictions.append(answer_dict)

    answer_dir = os.path.join(answer_path, time)
    os.makedirs(answer_dir, exist_ok=True)
    answer_path = os.path.join(answer_dir, f"{dataset_name}.json")
    with open(answer_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))

    # currently no quantitative evaluation, just return 1.0
    return 1.0
