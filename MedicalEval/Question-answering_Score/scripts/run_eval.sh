# Specify a model to be evaluated: MiniGPT-4 BLIP2 InstructBLIP LLaMA-Adapter-v2 LLaVA Otter mPLUG-Owl VPGTrans llava-med
MODEL=LLaMA-Adapter-v2


if [ "$MODEL" == 'llava-med' ]; then
    echo "Evaluating LLaVA-Med"
    python LLaVA-Med/llava/eval/model_med_eval.py --question-file /path/to/dataset
else
    echo $MODEL
    python eval_medical.py --model_name $MODEL --dataset_path /path/to/dataset
fi

