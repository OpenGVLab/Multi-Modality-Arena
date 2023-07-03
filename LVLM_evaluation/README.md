# holistic_evaluation

## Environments

* Setup a new conda env for LLaMA-Adapter-v2
```bash
cd env_info
conda create -n llama_adapter_v2 python=3.8 -y
pip install -r llama_requirements.txt
```

* Setup a new conda env for the rest models
```bash
cd env_info
conda env create -f VLP_web.yaml
```


## Model checkpoints
Most weights and checkpoint files will be downloaded automatically when initialing the corresponding testers. However, there are some files you should download personally and put in a directory. Then please replace the variable `DATA_DIR` in the `models/__init__.py` with the directory you save these files. Please note that the downloaded files should be organized as follows:

```bash
/path/to/DATA_DIR
├── llama_checkpoints
│   ├── 7B
│   │   ├── checklist.chk
│   │   ├── consolidated.00.pth
│   │   └── params.json
│   └── tokenizer.model
├── MiniGPT-4
│   ├── alignment.txt
│   └── pretrained_minigpt4_7b.pth
├── VPGTrans_Vicuna
└── otter-9b-hf
```

* For LLaMA-Adapter-v2, please obtain the LLaMA backbone weights using [this form](https://forms.gle/jk851eBVbX1m5TAv5).

* For MiniGPT-4, please download [alignment.txt](https://github.com/Vision-CAIR/MiniGPT-4/blob/22d8888ca2cf0aac862f537e7d22ef5830036808/prompts/alignment.txt#L3) and [pretrained_minigpt4_7b.pth](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing).

* For VPGTrans, please download [VPGTrans_Vicuna](https://drive.google.com/drive/folders/1YpBaEBNL-2a5DrU3h2mMtvqkkeBQaRWp?usp=sharing).


## Datasets



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
