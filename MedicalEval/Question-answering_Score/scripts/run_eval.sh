# Specify a model to be evaluated: MiniGPT-4 BLIP2 InstructBLIP LLaMA-Adapter-v2 LLaVA Otter mPLUG-Owl VPGTrans llava-med
MODEL=LLaMA-Adapter-v2
DATA=/path/to/dataset

exp_name=$MODEL
mkdir -p output/"$exp_name"

if [ "$MODEL" == 'llava-med' ]; then
    echo "Evaluating LLaVA-Med"
    python LLaVA-Med/llava/eval/model_med_eval.py --question-file $DATA
else
    echo Evaluating $MODEL
    python eval_medical.py --model_name $MODEL --dataset_path $DATA
fi

