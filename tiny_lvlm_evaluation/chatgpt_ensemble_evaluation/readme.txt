1.First, orginize the evaluation from different LVLM as follow:

-/path_to/original_item
----LLaMA-Adapter-v2
--------dataset1.json
--------dataset2.json
--------....
----InstructBLIP
--------dataset1.json
--------dataset2.json
--------....
----....

Each 'dataset.json' should be organized as follow:
[
    {
        "question": "The photo of the",
        "answer": "The image features a small white and brown dog sitting on a tiled floor.",
        "gt_answers": "dog",
        "image_path": "./tiny_lvlm_datasets/CIFAR10/00.png",
        "model_name": "LLaMA-Adapter-v2",
        "task_type": "VQA",
        "question_id": "000000"
    },
    {
        "question": "The photo of the",
        "answer": "The image features a blue sports car with a sleek design, parked on",
        "gt_answers": "automobile",
        "image_path": "./tiny_lvlm_datasets/CIFAR10/01.png",
        "model_name": "LLaMA-Adapter-v2",
        "task_type": "VQA",
        "question_id": "000001"
    },
    ....

]

2.Run chatgpt_evaluate.py to obtain the evaluation result for each LVLM. The results will be saved to '/path_to/save_dir' as you set in the code.

3.Run cal_ensemble_accuracy.py to obtain the final ensemble result. The root_path in the code should be set as the same as '/path_to/save_dir' in step 2.
