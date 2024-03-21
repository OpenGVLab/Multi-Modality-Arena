## Model checkpoints
Most weights and checkpoint files will be downloaded automatically when initialing the corresponding testers. However, there are some files you should download personally and put in a directory. Please note that the downloaded files should be organized as follows:

```bash
/path/to/VLP_web_data
├── llama_checkpoints
│   ├── 7B
│   │   ├── checklist.chk
│   │   ├── consolidated.00.pth
│   │   └── params.json
│   └── tokenizer.model
│   ├── 7B_hf
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── pytorch_model.bin
│   │   ├── pytorch_model.bin.index.json
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── tokenizer.model
│   └── tokenizer.model
├── MiniGPT-4
│   ├── alignment.txt
│   └── pretrained_minigpt4_7b.pth
├── llava_med
│   └── llava_med_in_text_60k
│       ├── added_tokens.json
│       ├── config.json
│       ├── generation_config.json
│       ├── pytorch_model-00001-of-00002.bin
│       ├── pytorch_model-00002-of-00002.bin
│       ├── pytorch_model.bin.index.json
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       └── tokenizer.model
├── VPGTrans_Vicuna
│
├── Radfm
│   └── pytorch_model.bin
│
├── MedVInT
│   └── MedVInT-TD
│   └── MedVInT-TE
│
└── otter-9b-hf
│
└── test_path.json
```

* For LLaMA-Adapter-v2, please obtain the LLaMA backbone weights using [this form](https://forms.gle/jk851eBVbX1m5TAv5).

* For MiniGPT-4, please download [alignment.txt](https://github.com/Vision-CAIR/MiniGPT-4/blob/22d8888ca2cf0aac862f537e7d22ef5830036808/prompts/alignment.txt#L3) and [pretrained_minigpt4_7b.pth](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing).

* For VPGTrans, please download [VPGTrans_Vicuna](https://drive.google.com/drive/folders/1YpBaEBNL-2a5DrU3h2mMtvqkkeBQaRWp?usp=sharing).

* For Otter, you can download the version we used in our evaluation from [this repo](https://huggingface.co/BellXP/otter-9b-hf). However, please note that the authors of Otter have updated their model, which is better than the version we used in evaluations, please check their [github repo](https://github.com/Luodian/Otter/tree/main) for the newest version.


* For LlaVA-Med, please follow the instructions in [here](https://github.com/microsoft/LLaVA-Med?tab=readme-ov-file#model-download) to prepare the weights of LLaVA-Med. Please note that the authors of LLaVA-Med have updated their model, you can download the latest version of the parameter for evaluation [here](xxxxx).

* For RadFM, please follow the instructions in [here](https://github.com/chaoyi-wu/RadFM) to prepare the environment and download the weights of the network.

* For MedVInT, please follow the instructions in [here](https://github.com/xiaoman-zhang/PMC-VQA/tree/master) to prepare the environment and download the weights of the network.

* For Med-flamingo, please follow the instructions in [here](https://github.com/snap-stanford/med-flamingo) to prepare the environment and download the weights of the network.

* For test_path.json in VLP_web_data, you need to add the path of the evaluated json file from OmniMedVQA.

* We strongly thank all the authors of the evaluated methods. We appreciate their contributions in building the LVLMs. If you utilize the above models for evaluation, remember to cite these works accordingly. Thanks for their wonderful works.





* Radfm:
* Radfm/test/test.sh
 
* Med-flamingo:
* Med-flamingo/scripts/test.sh
 
* MedVInT:
* Prefix_based_scores:
* MedVInT/src/MedVInT_TE/test.sh
* Question_answering_scores:
* MedVInT/src/MedVInT_TD/test.sh

