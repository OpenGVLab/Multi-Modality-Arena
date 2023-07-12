## Contents

+ [Installation](#installation)
+ [Dataset](#dataset)
+ [Model](#model)
+ [Run](#run)
+ [Evaluation](#evaluation)
+ [Cite](#cite)

## Installation

Clone our repository and create a new python environment via the follwing command
```
conda env create -f environment.yml
conda activate idealgpt
```

If you would like to use [LLaVA](https://github.com/haotian-liu/LLaVA) and [MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4) to solve sub-questions, please install them as mentioned in their repository.

## Dataset

In our paper, we conduct experiments on [SNLI-VE](https://github.com/necla-ml/SNLI-VE) and [VCR](https://visualcommonsense.com/). Please refer to their website to see how to download the data.


## Model
Most weights and checkpoint files will be downloaded automatically when initialing the corresponding testers. However, there are some files you should download personally and put in a directory. Then please replace the variable `DATA_DIR` in the `models/__init__.py` with the directory you save these files. Please note that the downloaded files should be organized as follows:

```bash
/path/to/VLP_web_data
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

## Run

> NOTE: 1. If you would like to run our code, please replace the filepath with yours. 2. You need to configure an OpenAI key to use OpenAI API. More details can be found at [OpenAI platform](https://platform.openai.com/)

In order to save money and running time, we have randomly select 500 samples from the val/dev split of VCR and SNLI-VE. 
The sampled indexes are saved in vcr_val_random500_annoid.yaml and ve_dev_random500_pairid.yaml in misc/exp_data split.
You can also randomly select more or less samples from the val/dev split of VCR and SNLI-VE by running sample_data.py. (dataset can be vcr_val or ve_dev)

```
cd misc
python sample_data.py --dataset=vcr_val
cd ..
```

Then, you can use IdealGPT to do inference. Here is an example of zero-shot VCR.
## VCR

```Shell
BLIP2
python blip_gpt_main.py  \
    --data_root=./datasets/vcr \
    --exp_tag=vcr_blip2 \
    --dataset=vcr_val \
    --device_id=0 \
    --prompt_setting=v1a \
    --data_partition=0_499 \
    --vqa_model=blip2_t5_xl  \
    --temp_gpt=0.0  \
    --data_subset=./misc/exp_data/vcr_val_random500_annoid.yaml  \
    --openai_key=<your_openai_key>

```


## SNLI-VE

```Shell
BLIP2
python blip_gpt_main.py  \
    --data_root=./datasets//SNLI-VE/SNLI-VE/data/Flickr30K/snli_1.0 \
    --exp_tag=ve_blip2 \
    --dataset=ve_dev \
    --device_id=0 \
    --prompt_setting=v1a \
    --data_partition=0_499 \
    --vqa_model=blip2_t5_xl  \
    --temp_gpt=0.0  \
    --data_subset=./misc/exp_data/ve_dev_random500_pairid.yaml  \
    --openai_key=<your_openai_key>
```


## Evaluation
We employ accuracy to evaluate zero-shot performance on VCR and SNLI-VE.

1. VCR
```
python vcr_eval.py --result=[path of saved VCR result folder, named by exp_tag in run]
```

2. SNLI-VE
```
python ve_eval.py --result=[path of saved VCR result folder, named by exp_tag in run]
```


## Note
This code is based on IdealGPT [(https://github.com/Hxyou/IdealGPT)](https://github.com/Hxyou/IdealGPT).
