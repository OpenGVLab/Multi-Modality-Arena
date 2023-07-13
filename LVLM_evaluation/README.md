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

* For Otter, you can download the version we used in our evaluation from [this repo](https://huggingface.co/BellXP/otter-9b-hf). However, please note that the authors of Otter have updated their model, which is better than the version we used in evaluations, please check their [github repo](https://github.com/Luodian/Otter/tree/main) for the newest version.


## Evaluation

```bash
python eval.py \
--model_name $MODEL
--device $CUDA_DEVICE_INDEX \
--batch_size $EVAL_BATCH_SIZE \
--dataset_name $DATASET \
--question $QUESTION \
--max_new_tokens $MAX_NEW_TOKENS \
--answer_path $SAVE_DIR
--eval_(ocr/vqa/caption/kie/mrr/embod/cls) \
# please check the name of models/datasets in (models/task_datasets)/__init__.py
# do not need to specific question and max_new_tokens in default
```

For the evalution tasks can not using the default settings, here is the list:

| Dataset        | task             | `--dataset_name` | `--max_new_tokens` | `--question`                                                  | download                                           |
|----------------|------------------|------------------|--------------------|---------------------------------------------------------------|----------------------------------------------------|
| ImageNet1K     | `--eval_cls`     | ImageNet         | 64                 | The photo of the                                              | https://image-net.org/download.php                 |
| CIFAR10        | `--eval_cls`     | CIFAR10          | 64                 | The photo of the                                              | https://www.cs.toronto.edu/~kriz/cifar.html        |
| Pets37         | `--eval_cls`     | OxfordIIITPet    | 64                 | What is the specific category of the dog or cat in the image? | https://www.robots.ox.ac.uk/~vgg/data/pets/        |
| Flowers102     | `--eval_cls`     | Flowers102       | 64                 | What is the specific category of the flower in the image?     | https://www.robots.ox.ac.uk/~vgg/data/flowers/102/ |
| WHOOPS Caption | `--eval_caption` | WHOOPSCaption    | 16                 | A photo of                                                    | https://huggingface.co/datasets/nlphuji/whoops     |
| WHOOPS VQA     | `--eval_vqa`     | WHOOPSVQA        | 16                 | -                                                             | https://huggingface.co/datasets/nlphuji/whoops     |


## Datasets
For dataset preparation, you can download and process the datasets we use personally or use the version we provided in [here (passwd:1puw)](https://pan.baidu.com/s/1lvNLpCjZbReB_FAJOnR3PQ?pwd=1puw). Then please organize the datasets as follows and then replace the variable `DATA_DIR` in the `task_datasets/__init__.py` with the directory you save these datasets.

```bash
/path/to/DATA_DIR
├── Caption_Datasets
│   ├── Flickr_30k
│   │   ├── flickr30k-images
│   │   └── results_20130124.token
│   └── NoCaps
│       ├── nocaps_val_4500_captions.json
│       └── val_imgs
├── CLS_Datasets
│   ├── flowers-102
│   └── oxford-iiit-pet
├── Embodied_Datasets
│   ├── FrankaKitchen
│   ├── MetaWorld
│   ├── Minecraft
│   ├── MinecraftPolicy
│   └── VirtualHome
├── ImageNet
├── ImageNetVC
├── KIE_Datasets
│   ├── FUNSD
│   │   └── testing_data
│   └── SROIE
│       ├── gt_answers
│       └── images
├── MSCOCO
├── OCR_Datasets
│   ├── COCO-Text
│   ├── CTW
│   ├── CUTE80
│   ├── HOST
│   ├── IC13
│   ├── IC15
│   ├── IIIT5K
│   ├── SVT
│   ├── SVTP
│   ├── Total-Text
│   ├── WordArt
│   └── WOST
├── SNLI-VE
├── VCR
└── VQA_Datasets
    ├── DocVQA
    │   ├── val
    │   └── val.json
    ├── GQA
    │   ├── images
    │   └── questions
    │       └── testdev_balanced_questions.json
    ├── IconQA
    │   └── datasets
    │       └── test
    ├── OCRVQA
    │   ├── dataset.json
    │   └── loadDataset.py
    ├── OKVQA
    │   ├── mscoco_val2014_annotations.json
    │   ├── OpenEnded_mscoco_val2014_questions.json
    │   └── val2014
    ├── ScienceQA
    ├── STVQA
    │   ├── train_imgs
    │   └── train_task_3.json
    ├── TextVQA
    │   ├── TextVQA_0.5.1_val.json
    │   └── train_images
    ├── Visdial
    │   ├── images_val2018
    │   └── visdial_1.0_val.json
    ├── VizWiz
    │   ├── val
    │   └── val_grounding.json
    └── VSR
        ├── all_vsr_validated_data.jsonl
        └── images
```

* For Caption datasets,
    - Flickr_30k, please obtain the [flickr30k-images](https://uofi.box.com/s/1cpolrtkckn4hxr1zhmfg0ln9veo6jpl) and get the [results_20130124.token](http://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k.tar.gz).
    - NoCaps, please download [nocaps_val_4500_captions.json](https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json) at first and then download the images from the urls provided in nocaps_val_4500_captions.json and save them in `DATA_DIR/Caption_Datasets/NoCaps/val_imgs` with the filename provided in nocaps_val_4500_captions.json.

* For KIE datasets,
    - FUNSD, please download the [datasets of FUNSD](https://guillaumejaume.github.io/FUNSD/dataset.zip) and the extract the compressed file and organize it as the aforementioned directory tree.
    - SROIE, please download the test set images of task 3 from [SROIE's Github repo](https://github.com/zzzDavid/ICDAR-2019-SROIE) and then process then by following the process in [there](https://github.com/zzzDavid/ICDAR-2019-SROIE/tree/master/task3). After that, you will get the gt answers of each image. Then please put these images and gt answers in the `images` and `gt_answers` folder. Notably, we change the format of gt answer files from .json to .txt.

* For Embodied datasets.
    - We select some representive frames from [MetaWorld](https://github.com/Farama-Foundation/Metaworld), [FrankaKitchen](https://robotics.farama.org/envs/franka_kitchen/franka_kitchen/), [VirtualHome](http://virtual-home.org/) and [MineDojo](https://github.com/MineDojo/MineDojo). Please follow the instructions respectively to get the datasets.

* For VQA datasets,
    - DocVQA, please download the validation set of task 1 in the website of [robust reading competition](https://rrc.cvc.uab.es/?ch=17) as then put the data into the dataset directory organized as previous mentiond.
    - GQA, please download the [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip) and [questions](https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip) and make sure the file `testdev_balanced_questions.json` is placed in the correct directory.
    - IconQA, please first follow the instructions provided in [IconQA's github repo](https://github.com/lupantech/IconQA/tree/main) and then move the `IconQA/data/iconqa_data/iconqa` into `DATA_DIR/VQA_Datasets/IconQA/datasets`.
    - OCRVQA, please first download the `dataset.json` and `loadDataset.py` provided in this [google drive link](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_) and then download the images used in OCRVQA by running the `loadDataset.py`.
    - OKVQA, please first download the [questions](https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip) and [annotations](https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip) and then put the images of MSCOCO2014 validation set in the directroy shown above.
    - ScienceQA, in initialing the ScienceQA dataset, the python script will download the test split of ScienceQA from huggingface directly and then saving the samples with image provided in `DATA_DIR/VQA_Datasets/ScienceQA`.
    - STVQA, we use the end-to-end task 3 of STVQA, the dataset can be downloaded in [robust reading competition](https://rrc.cvc.uab.es/?ch=11). Please note that as the gt answers are not provided in the test set of STVQA's task 3, the split we use is the train set.
    - TextVQA, please download the [TextVQA_0.5.1_val.json](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json) and the [train_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip).
    - Visdial, please download the [visdial_1.0_val](https://www.dropbox.com/s/ibs3a0zhw74zisc/visdial_1.0_val.zip?dl=0) and [VisualDialog_val2018](https://www.dropbox.com/s/twmtutniktom7tu/VisualDialog_val2018.zip?dl=0). Then rename the `VisualDialog_val2018` as `images_val2018`.
    - VizWiz, please download the [val images](https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip) and [val annotations](https://vizwiz.cs.colorado.edu/VizWiz_final/VizWiz_grounding/annotations.zip) and then organize them as the directory tree shown above.
    - VSR, please first download the [all_vsr_validated_data.jsonl](https://github.com/cambridgeltl/visual-spatial-reasoning/blob/master/data/data_files/all_vsr_validated_data.jsonl) and then download the images from the `image_link` and save it as the `image` provided in `all_vsr_validated_data.jsonl`.

* For VQA datasets, most of them can be obtained by following the instructions shown in [MMOCR's text recognition dataset preperation page](https://mmocr.readthedocs.io/en/v0.6.1/datasets/recog.html), which is inclusive of COCO-Text, CUTE80, ICDAR 13 (IC13), ICDAR 15 (IC15), IIIT5K, SVT, SVTP and Total-Text. Then for WordArt dataset, it is available at [this google drive link](https://drive.google.com/file/d/1SanxRwTxd2q7UrQxlbC3BmP3nhFXwZ3g/view). For CTW dataset, please download its test set in [this link](https://ctwdataset.github.io/downloads.html). Finally, HOST and WOST are the heavy and weakly categories of occlusion scene text dataset, which can be downloaded from its [official github repo](https://github.com/wangyuxin87/VisionLAN). Notably, after getting these OCR datasets, please prepare a `test_label.txt` file for each of them. In each line of the `test_label.txt`, the relative path of the image file and its text label should be provided as follow (take the WordArt dataset as the example):
    ```bash
    WordArt/test_image/2097.png INDIE
    WordArt/test_image/2100.png adventure
    ```

* Then there are five individual datasets, which are ImageNet, ImageNetVC, MSCOCO, SNLI-VE and VCR.
    - ImageNet, please request to download ImageNet 2012 in [this website](https://image-net.org/download.php).
    - ImageNetVC, please download in from [its official github repo](https://github.com/hemingkx/ImageNetVC/tree/main/VaLM/BLIP-2).
    - MSCOCO, please download the [2014 val images](http://images.cocodataset.org/zips/val2014.zip) and put it in `MSCOCO/val2014`.
    - SNLI-VE, please download it from [this website](https://github.com/necla-ml/SNLI-VE).
    - VCR, please first download the VCR dataset in [this website](https://visualcommonsense.com/download/) and then rename it from `vcr1` to `VCR`.