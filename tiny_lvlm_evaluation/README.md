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

The datasets used in Tiny LVLM Evaluation are the subset of the datasets used in LVLM Evaluation. Therefore, you can download the sampled subset in [here](https://drive.google.com/file/d/1K7vLSBH9qi-OHQSBxxKKQkD3JseCae8y/view?usp=sharing) and use it directly. The script sample_dataset.py is used to sample the subsets used in Tiny LVLM Evaluation and save it.

Besides, the inference results on 42 datasets of all 12 multimodal models studied in Tiny LVLM-eHub, including Bard, are downloadable from [Google Drive](https://drive.google.com/file/d/1yhtKbIRcnLO3RaUl6UoRSlSS2bsKmBkW/view?usp=sharing).

## Prompt Engineering

The table below shows prompts used for each dataset and across all multimodal models under study.

| Prompt | Dataset |
|---|---|
| Classify the main object in the image. | ImageNet1K, CIFAR10 |
| What breed is the flower in the image? | Flowers102 |
| What breed is the pet in the image? | OxfordIIITPet |
| What is written in the image? | All 12 OCR datasets |
| Question: {question}\nChoose the best answer from the following choices:\n- option#1\n- option#2\n- option#3\n | IconQA |
| Context:\n{context}\n\nQuestion: {question}\nChoose the best answer from the following choices:\n- option#1\n- option#2\n- option#3 | ScienceQA |
| Question: {question}\n\nChoose the single most likely answer from the following choices \<choice>:\n- Yes\n- No\n\nThe output format follows exactly as below:\nAnswer: \<choice> | MSCOCO_MCI, VCR_MCI |
| Question: Is the caption "{caption}" correctly describing the image?\n\nChoose the single most likely answer from the following choices \<choice>:\n- Yes\n- No\n\nThe output format follows exactly as below:\nAnswer: \<choice> | VSR |
| use original questions | other datasets |

