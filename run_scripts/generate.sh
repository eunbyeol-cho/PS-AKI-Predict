#!/bin/bash

gpu_id=0
seeds="0 1 2"

for seed in $seeds; do
    OMP_NUM_THREADS=8 \
    NUMEXPR_MAX_THREADS=128 \
    CUDA_VISIBLE_DEVICES=$gpu_id \
    python main.py with \
          task_test_table2table \
          seed=$seed \
          input_path=$input_path \
          target_study=$target_study
done
