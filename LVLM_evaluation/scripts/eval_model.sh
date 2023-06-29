model_name=$1
device=${2:-0}
batch_size=${3:-64}

# OCR
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_ocr
# Caption
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_caption --dataset_name NoCaps
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_caption --dataset_name Flickr --sample_num 1000
# KIE
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_kie --dataset_name SROIE
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_kie --dataset_name FUNSD
# VQA
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name TextVQA
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name DocVQA
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name STVQA --sample_num 4000
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name ScienceQA
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name OKVQA
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name GQA
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name VizWiz
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name IconQA
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name VSR
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name OCRVQA
# MRR
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_mrr --dataset_name Visdial
