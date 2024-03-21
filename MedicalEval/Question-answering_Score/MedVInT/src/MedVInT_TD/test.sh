export PATH=/usr/local/cuda/bin:$PATH

OUTPUT_DIR="PATH1"
JSON_PATH="PATH2"


CUDA_LAUNCH_BLOCKING=1 \
srun --partition=partition_name --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=job_name --kill-on-bad-exit=1 --quotatype=auto --async -o "$OUTPUT_DIR"/log_test.out --mail-type=ALL \
    torchrun --nproc_per_node=1 --master_port 12345 test.py \
    --ckp "path_to_pretrained_parameter" \
    --output_dir "$OUTPUT_DIR" \
    --logging_dir "$OUTPUT_DIR" \
    --json_path "$JSON_PATH" \

