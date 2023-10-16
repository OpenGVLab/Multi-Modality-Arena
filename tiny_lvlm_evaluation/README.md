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

## Ability-level Benchmark
Beyond the inclusion of a partial dataset from the Tiny LVLM Evaluation, we present an enhanced dataset segmentation. This novel division systematically categorizes the datasets featured in the Tiny LVLM Evaluation according to their specific targeted abilities. Subsequently, we curate a subset of datasets that align with the evaluative criteria of the LVLM model and aggregate these subsets into an ability-level subset, excluding those related to embodied intelligence. Furthermore, this benchmark includes recently released models to bolster its comprehensiveness.

You can download the ability-level subset from [here](https://drive.google.com/file/d/1PuFC612XzOmKwzRldtBb1CFZnIjiR7we/view?usp=sharing), and the inference results of all 20 multimodal models included in our benchmark can be found in [here](https://drive.google.com/file/d/12pEJZSIUf-v4EDKmtiqgJK2j6nCBVY_W/view?usp=sharing).

Here is an example command for using this benchmark:
```bash
python updated_eval_tiny.py
--model-name $MODEL
--device $CUDA_DEVICE_INDEX
--batch-size $EVAL_BATCH_SIZE
--sampled-root $ROOT_DIR_OF_SAMPLED_SUBSETS
--answer_path $SAVE_DIR
```

For detailed performance metrics, please refer to following tables.
### Visual Reasoning
| Rank | Model | Version | Score |
| :--: | :--: | :--: | :--: |
| 🏅️ | **[Bard](https://bard.google.com/)** | Bard | **64.18** |
| 🥈 | **[Qwen-VL-Chat](https://github.com/QwenLM/Qwen-VL)** | Qwen-VL-Chat | **62.36** |
| 🥉 | **[Lynx](https://github.com/bytedance/lynx-llm)** | Vicuna-7B | **52.18** |
| 4 | [InternLM-XComposer](https://github.com/InternLM/InternLM-XComposer) | InternLM-XComposer-7B | 48.00 |
| 5 | [InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) | Vicuna-7B | 46.73 |
| 6 | [BLIP2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) | FlanT5xl | 44.91 |
| 7 | [LLaMA-Adapter-v2](https://github.com/OpenGVLab/LLaMA-Adapter/tree/main/llama_adapter_v2_multimodal7b) | LLaMA-7B | 43.45 |
| 8 | [Otter-Image](https://github.com/Luodian/Otter) | Otter-9B-LA-InContext | 41.64 |
| 9 | [LLaVA](https://github.com/haotian-liu/LLaVA) | MPT-7B | 41.64 |
| 10 | [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl) | LLaMA-7B | 40.91 |
| 11 | [Cheetah](https://github.com/DCDmllm/Cheetah) | Vicuna-7B | 40.00 |
| 12 | [BLIVA](https://github.com/mlpc-ucsd/BLIVA) | Vicuna-7B | 38.73 |
| 13 | [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) | Vicuna-7B | 37.64 |
| 14 | [VisualGLM-6B](https://github.com/THUDM/VisualGLM-6B) | VisualGLM-6B | 37.27 |
| 15 | [OFv2_4BI](https://github.com/mlfoundations/open_flamingo) | RedPajama-INCITE-Instruct-3B-v1 | 33.64 |
| 16 | [PandaGPT](https://github.com/yxuansu/PandaGPT) | Vicuna-7B | 33.45 |
| 17 | [Otter](https://github.com/Luodian/Otter) | Otter-9B | 29.82 |
| 18 | [VPGTrans](https://github.com/VPGTrans/VPGTrans) | Vicuna-7B | 27.27 |
| 19 | [LaVIN](https://github.com/luogen1996/LaVIN) | LLaMA-7B | 20.36 |
| 20 | [MIC](https://github.com/HaozheZhao/MIC) | FlanT5xl | 11.09 |

### Visual Perception
| Rank | Model | Version | Score |
| :--: | :--: | :--: | :--: |
| 🏅️ | **[Lynx](https://github.com/bytedance/lynx-llm)** | Vicuna-7B | **65.75** |
| 🥈 | **[Bard](https://bard.google.com/)** | Bard | **57.00** |
| 🥉 | **[InternLM-XComposer](https://github.com/InternLM/InternLM-XComposer)** | InternLM-XComposer-7B | **56.25** |
| 4 | [Qwen-VL-Chat](https://github.com/QwenLM/Qwen-VL) | Qwen-VL-Chat | 54.50 |
| 5 | [BLIP2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) | FlanT5xl | 49.00 |
| 6 | [InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) | Vicuna-7B | 48.00 |
| 7 | [BLIVA](https://github.com/mlpc-ucsd/BLIVA) | Vicuna-7B | 46.75 |
| 8 | [LLaMA-Adapter-v2](https://github.com/OpenGVLab/LLaMA-Adapter/tree/main/llama_adapter_v2_multimodal7b) | LLaMA-7B | 46.75 |
| 9 | [Cheetah](https://github.com/DCDmllm/Cheetah) | Vicuna-7B | 43.25 |
| 10 | [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl) | LLaMA-7B | 40.75 |
| 11 | [LLaVA](https://github.com/haotian-liu/LLaVA) | MPT-7B | 38.25 |
| 12 | [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) | Vicuna-7B | 37.75 |
| 13 | [OFv2_4BI](https://github.com/mlfoundations/open_flamingo) | RedPajama-INCITE-Instruct-3B-v1 | 37.25 |
| 14 | [Otter](https://github.com/Luodian/Otter) | Otter-9B | 37.00 |
| 15 | [VisualGLM-6B](https://github.com/THUDM/VisualGLM-6B) | VisualGLM-6B | 36.25 |
| 16 | [Otter-Image](https://github.com/Luodian/Otter) | Otter-9B-LA-InContext | 33.25 |
| 17 | [PandaGPT](https://github.com/yxuansu/PandaGPT) | Vicuna-7B | 33.00 |
| 18 | [VPGTrans](https://github.com/VPGTrans/VPGTrans) | Vicuna-7B | 31.25 |
| 19 | [LaVIN](https://github.com/luogen1996/LaVIN) | LLaMA-7B | 20.00 |
| 20 | [MIC](https://github.com/HaozheZhao/MIC) | FlanT5xl | 0.75 |

### Visual Knowledge Acquisition
| Rank | Model | Version | Score |
| :--: | :--: | :--: | :--: |
| 🏅️ | **[Bard](https://bard.google.com/)** | Bard | **68.14** |
| 🥈 | **[InternLM-XComposer](https://github.com/InternLM/InternLM-XComposer)** | InternLM-XComposer-7B | **66.57** |
| 🥉 | **[BLIP2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)** | FlanT5xl | **64.14** |
| 4 | [BLIVA](https://github.com/mlpc-ucsd/BLIVA) | Vicuna-7B | 63.43 |
| 5 | [InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) | Vicuna-7B | 61.71 |
| 6 | [Qwen-VL-Chat](https://github.com/QwenLM/Qwen-VL) | Qwen-VL-Chat | 55.14 |
| 7 | [VPGTrans](https://github.com/VPGTrans/VPGTrans) | Vicuna-7B | 49.86 |
| 8 | [Cheetah](https://github.com/DCDmllm/Cheetah) | Vicuna-7B | 46.86 |
| 9 | [VisualGLM-6B](https://github.com/THUDM/VisualGLM-6B) | VisualGLM-6B | 46.86 |
| 10 | [LLaMA-Adapter-v2](https://github.com/OpenGVLab/LLaMA-Adapter/tree/main/llama_adapter_v2_multimodal7b) | LLaMA-7B | 22.29 |
| 11 | [LLaVA](https://github.com/haotian-liu/LLaVA) | MPT-7B | 18.71 |
| 12 | [Lynx](https://github.com/bytedance/lynx-llm) | Vicuna-7B | 17.57 |
| 13 | [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) | Vicuna-7B | 17.57 |
| 14 | [OFv2_4BI](https://github.com/mlfoundations/open_flamingo) | RedPajama-INCITE-Instruct-3B-v1 | 17.29 |
| 15 | [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl) | LLaMA-7B | 16.14 |
| 16 | [Otter-Image](https://github.com/Luodian/Otter) | Otter-9B-LA-InContext | 15.14 |
| 17 | [Otter](https://github.com/Luodian/Otter) | Otter-9B | 12.71 |
| 18 | [MIC](https://github.com/HaozheZhao/MIC) | FlanT5xl | 7.71 |
| 19 | [PandaGPT](https://github.com/yxuansu/PandaGPT) | Vicuna-7B | 3.00 |
| 20 | [LaVIN](https://github.com/luogen1996/LaVIN) | LLaMA-7B | 2.14 |

### Visual Commonsense
| Rank | Model | Version | Score |
| :--: | :--: | :--: | :--: |
| 🏅️ | **[Bard](https://bard.google.com/)** | Bard | **59.60** |
| 🥈 | **[InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)** | Vicuna-7B | **59.20** |
| 🥉 | **[BLIVA](https://github.com/mlpc-ucsd/BLIVA)** | Vicuna-7B | **58.60** |
| 4 | [Lynx](https://github.com/bytedance/lynx-llm) | Vicuna-7B | 57.40 |
| 5 | [LLaMA-Adapter-v2](https://github.com/OpenGVLab/LLaMA-Adapter/tree/main/llama_adapter_v2_multimodal7b) | LLaMA-7B | 56.00 |
| 6 | [Qwen-VL-Chat](https://github.com/QwenLM/Qwen-VL) | Qwen-VL-Chat | 54.80 |
| 7 | [Otter-Image](https://github.com/Luodian/Otter) | Otter-9B-LA-InContext | 52.40 |
| 8 | [Cheetah](https://github.com/DCDmllm/Cheetah) | Vicuna-7B | 51.80 |
| 9 | [PandaGPT](https://github.com/yxuansu/PandaGPT) | Vicuna-7B | 51.80 |
| 10 | [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl) | LLaMA-7B | 50.60 |
| 11 | [InternLM-XComposer](https://github.com/InternLM/InternLM-XComposer) | InternLM-XComposer-7B | 50.40 |
| 12 | [LLaVA](https://github.com/haotian-liu/LLaVA) | MPT-7B | 49.40 |
| 13 | [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) | Vicuna-7B | 49.00 |
| 14 | [VPGTrans](https://github.com/VPGTrans/VPGTrans) | Vicuna-7B | 48.20 |
| 15 | [Otter](https://github.com/Luodian/Otter) | Otter-9B | 48.00 |
| 16 | [BLIP2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) | FlanT5xl | 44.00 |
| 17 | [OFv2_4BI](https://github.com/mlfoundations/open_flamingo) | RedPajama-INCITE-Instruct-3B-v1 | 39.20 |
| 18 | [VisualGLM-6B](https://github.com/THUDM/VisualGLM-6B) | VisualGLM-6B | 37.60 |
| 19 | [LaVIN](https://github.com/luogen1996/LaVIN) | LLaMA-7B | 35.00 |
| 20 | [MIC](https://github.com/HaozheZhao/MIC) | FlanT5xl | 24.20 |

### Object Hallucination
| Rank | Model | Version | Score |
| :--: | :--: | :--: | :--: |
| 🏅️ | **[Qwen-VL-Chat](https://github.com/QwenLM/Qwen-VL)** | Qwen-VL-Chat | **90.00** |
| 🥈 | **[Lynx](https://github.com/bytedance/lynx-llm)** | Vicuna-7B | **86.33** |
| 🥉 | **[InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)** | Vicuna-7B | **85.00** |
| 4 | [BLIP2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) | FlanT5xl | 82.67 |
| 5 | [Cheetah](https://github.com/DCDmllm/Cheetah) | Vicuna-7B | 77.00 |
| 6 | [BLIVA](https://github.com/mlpc-ucsd/BLIVA) | Vicuna-7B | 76.67 |
| 7 | [Otter-Image](https://github.com/Luodian/Otter) | Otter-9B-LA-InContext | 74.00 |
| 8 | [Bard](https://bard.google.com/) | Bard | 70.67 |
| 9 | [InternLM-XComposer](https://github.com/InternLM/InternLM-XComposer) | InternLM-XComposer-7B | 67.67 |
| 10 | [VPGTrans](https://github.com/VPGTrans/VPGTrans) | Vicuna-7B | 62.33 |
| 11 | [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl) | LLaMA-7B | 61.00 |
| 12 | [LLaMA-Adapter-v2](https://github.com/OpenGVLab/LLaMA-Adapter/tree/main/llama_adapter_v2_multimodal7b) | LLaMA-7B | 60.67 |
| 13 | [VisualGLM-6B](https://github.com/THUDM/VisualGLM-6B) | VisualGLM-6B | 54.00 |
| 14 | [Otter](https://github.com/Luodian/Otter) | Otter-9B | 53.33 |
| 15 | [PandaGPT](https://github.com/yxuansu/PandaGPT) | Vicuna-7B | 53.00 |
| 16 | [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) | Vicuna-7B | 50.67 |
| 17 | [MIC](https://github.com/HaozheZhao/MIC) | FlanT5xl | 50.33 |
| 18 | [OFv2_4BI](https://github.com/mlfoundations/open_flamingo) | RedPajama-INCITE-Instruct-3B-v1 | 49.00 |
| 19 | [LLaVA](https://github.com/haotian-liu/LLaVA) | MPT-7B | 49.00 |
| 20 | [LaVIN](https://github.com/luogen1996/LaVIN) | LLaMA-7B | 20.00 |

### Overall Score
| Rank | Model | Version | Score |
| :--: | :--: | :--: | :--: |
| 🏅️ | **[Bard](https://bard.google.com/)** | Bard | **319.59** |
| 🥈 | **[Qwen-VL-Chat](https://github.com/QwenLM/Qwen-VL)** | Qwen-VL-Chat | **316.81** |
| 🥉 | **[InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)** | Vicuna-7B | **300.64** |
| 4 | [InternLM-XComposer](https://github.com/InternLM/InternLM-XComposer) | InternLM-XComposer-7B | 288.89 |
| 5 | [BLIP2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) | FlanT5xl | 284.72 |
| 6 | [BLIVA](https://github.com/mlpc-ucsd/BLIVA) | Vicuna-7B | 284.17 |
| 7 | [Lynx](https://github.com/bytedance/lynx-llm) | Vicuna-7B | 279.24 |
| 8 | [Cheetah](https://github.com/DCDmllm/Cheetah) | Vicuna-7B | 258.91 |
| 9 | [LLaMA-Adapter-v2](https://github.com/OpenGVLab/LLaMA-Adapter/tree/main/llama_adapter_v2_multimodal7b) | LLaMA-7B | 229.16 |
| 10 | [VPGTrans](https://github.com/VPGTrans/VPGTrans) | Vicuna-7B | 218.91 |
| 11 | [Otter-Image](https://github.com/Luodian/Otter) | Otter-9B-LA-InContext | 216.43 |
| 12 | [VisualGLM-6B](https://github.com/THUDM/VisualGLM-6B) | VisualGLM-6B | 211.98 |
| 13 | [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl) | LLaMA-7B | 209.40 |
| 14 | [LLaVA](https://github.com/haotian-liu/LLaVA) | MPT-7B | 197.00 |
| 15 | [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) | Vicuna-7B | 192.62 |
| 16 | [Otter](https://github.com/Luodian/Otter) | Otter-9B | 180.87 |
| 17 | [OFv2_4BI](https://github.com/mlfoundations/open_flamingo) | RedPajama-INCITE-Instruct-3B-v1 | 176.37 |
| 18 | [PandaGPT](https://github.com/yxuansu/PandaGPT) | Vicuna-7B | 174.25 |
| 19 | [LaVIN](https://github.com/luogen1996/LaVIN) | LLaMA-7B | 97.51 |
| 20 | [MIC](https://github.com/HaozheZhao/MIC) | FlanT5xl | 94.09 |
