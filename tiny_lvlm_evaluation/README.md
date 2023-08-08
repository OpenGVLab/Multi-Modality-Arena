# Tiny LVLM Evaluation

## Environments

```bash
conda create -n lvlm_eval python=3.8 -y
pip install -r requirements.txt
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
├── otter-9b-hf
└── PandaGPT
    ├── imagebind_ckpt
    ├── vicuna_ckpt
    └── pandagpt_ckpt
```

* For LLaMA-Adapter-v2, please obtain the LLaMA backbone weights using [this form](https://forms.gle/jk851eBVbX1m5TAv5).

* For MiniGPT-4, please download [alignment.txt](https://github.com/Vision-CAIR/MiniGPT-4/blob/22d8888ca2cf0aac862f537e7d22ef5830036808/prompts/alignment.txt#L3) and [pretrained_minigpt4_7b.pth](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing).

* For VPGTrans, please download [VPGTrans_Vicuna](https://drive.google.com/drive/folders/1YpBaEBNL-2a5DrU3h2mMtvqkkeBQaRWp?usp=sharing).

* For Otter, you can download the version we used in our evaluation from [this repo](https://huggingface.co/BellXP/otter-9b-hf). However, please note that the authors of Otter have updated their model, which is better than the version we used in evaluations, please check their [github repo](https://github.com/Luodian/Otter/tree/main) for the newest version.

* For PandaGPT, please follow the instructions in [here](https://github.com/yxuansu/PandaGPT/tree/main#environment) to prepare the weights of imagebind, vicuna and pandagpt.

## Datasets and Evaluation
```bash
python eval_tiny.py \
--model_name $MODEL
--device $CUDA_DEVICE_INDEX \
--batch-size $EVAL_BATCH_SIZE \
--dataset-names $DATASET_NAMES \
--sampled-root $SAMPLED_DATASET_DIR
--answer_path $SAVE_DIR
--use-sampled # use it when you have prepared the all datasets in LVLM evaluation and do not download the sampled data
# please check the name of models/datasets in (models/task_datasets)/__init__.py
```

The datasets used in Tiny LVLM Evaluation are the subset of the datasets used in LVLM Evaluation. Therefore, you can download the sampled subset in [here (password: 9svo)](https://pan.baidu.com/s/1JJXpS2XylwvpE9kKXnilSQ?pwd=9svo) and use it directly. The script sample_dataset.py is used to sample the subsets used in Tiny LVLM Evaluation and save it.
