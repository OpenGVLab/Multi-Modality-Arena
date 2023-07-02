# holistic_evaluation

Here are some details about environment, model checkpoints and the how to perform evaluation.

## Environments

In env_info folder, there are two yaml files: llama_adapter.yaml is the environment used for running the LLaMA-Adapter-v2 while VLP_web.yaml is used for running the rest models. Specifically, the CLIP and timm library shown in llama_adapter.yaml are customized, which are also be provided in env_info folder.

## Model checkpoints

The model checkpoints used in BLIP2, InstructBLIP, LLaVA, mPLUG-Owl and Otter will be downloaded automatically when initialing the corresponding testers. However, some files should be downloaded previously when running MiniGPT-4, VPGTrans and LLaMA-Adapter-v2. For these files, we recommand you to put them in the same directory and replace variable `DATA_DIR` in the `models/__init__.py` with it.

For MiniGPT-4, please download its [7B-ckpt](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing) and put it in `DATA_DIR/MiniGPT-4/pretrained_minigpt4_7b.pth`. Then for VPGTrans, you need download its [Vicuna model](https://drive.google.com/drive/folders/1YpBaEBNL-2a5DrU3h2mMtvqkkeBQaRWp?usp=sharing) and put int in `DATA_DIR/VPGTrans_Vicuna`. Finally, for LLaMA-Adapter-v2, we will update our code for using its huggingface version as soon as possible.

## Evaluation

```bash
python eval.py \
--model_name $MODEL
--device $CUDA_DEVICE_INDEX \
--batch_size $EVAL_BATCH_SIZE \
--dataset_name $DATASET \
--eval_(ocr/vqa/caption/kie/mrr/embod/cls) \
--answer_path $SAVE_DIR
# please check the name of models/datasets in (models/task_datasets)/__init__.py
```
