#!/bin/bash

python3 generate.py \
--config "configs/LYNX.yaml" \
--output_path "./result_0.jsonl" \
--device "cuda" \
--seed 42
