## Requirements

- Python >= 3.7
- Pytorch >= 1.8.0

## Installation

Please follow the original guideline in [BLIP-2](https://github.com/salesforce/LAVIS) to settle down the environment.

```
conda create -n blip2_ivc python=3.7
pip install -r requirements.txt
```

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

## Image Sources


Top-10 ranked images: [Baidu Pan Link](https://pan.baidu.com/s/1HlMMXuM1h3OARJY1JzfGwA?pwd=cgs2) (password: cgs2)


## Evaluation

```
python ./ImageNetVC_blip2.py
```

## Note

This code is based on ImageNetVC [(https://github.com/hemingkx/ImageNetVC/tree/main/VaLM/BLIP-2)](https://github.com/hemingkx/ImageNetVC/tree/main/VaLM/BLIP-2).

