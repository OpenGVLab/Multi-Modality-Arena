# Specify a model to be evaluated: MiniGPT-4 BLIP2 InstructBLIP LLaMA-Adapter-v2 LLaVA Otter mPLUG-Owl VPGTrans llava-med
MODEL=llava-med


exp_name=blip2
mkdir -p output/"$exp_name"

DATA=/path/to/dataset


if [ "$MODEL" == 'MiniGPT-4' ]; then
    runpath="medical_minigpt4.py"
elif [ "$MODEL" == 'BLIP2' ]; then
    runpath="medical_blip2.py"
elif [ "$MODEL" == 'InstructBLIP' ]; then
    runpath="medical_instructblip.py"
elif [ "$MODEL" == 'LLaMA-Adapter-v2' ]; then
    runpath="medical_llama_adapter2.py"
elif [ "$MODEL" == 'LLaVA' ]; then
    runpath="medical_llava.py"
elif [ "$MODEL" == 'Otter' ]; then
    runpath="medical_otter.py"
elif [ "$MODEL" == 'mPLUG-Owl' ]; then
    runpath="medical_owl.py"
elif [ "$MODEL" == 'VPGTrans' ]; then
    runpath="medical_vpgtrans.py"
elif [ "$MODEL" == 'llava-med' ]; then
    runpath="LLaVA-Med/llava/eval/model_med_eval_sp.py"
else
    echo "Model not found"
fi

echo Evaluating $MODEL 
python $runpath --dataset_path /path/to/dataset