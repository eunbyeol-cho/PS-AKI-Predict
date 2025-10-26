# OMP_NUM_THREADS=8 \
# NUMEXPR_MAX_THREADS=128 \
# CUDA_VISIBLE_DEVICES=${gpu_id} \
#   python main.py with task_train_table2table \
#   debug=True \
#   seed=${seed}

#1,2,3,4
# gpu_id=4
# seeds="2020 2021 2022"
# mask_schedulers="pow1 pow0.5 cosine exp"

gpu_id=4
seeds="0 1 2"
mask_schedulers="pow0.5"
# studys="AMC"
# studys="BRMH KNUH CAUH KUMC"
studys="AMC"

# Loop through each mask_scheduler and seed combination
for study in $studys; do
    for mask_scheduler in $mask_schedulers; do
        for seed in $seeds; do
            OMP_NUM_THREADS=8 \
            NUMEXPR_MAX_THREADS=128 \
            CUDA_VISIBLE_DEVICES=$gpu_id \
            python main.py with \
                task_train_table2table_$study \
                debug=False \
                wandb_project_name="mmtg-20240609" \
                seed=$seed \
                mask_scheduler=$mask_scheduler \
                study=$study \
                input_path='/home/data_storage/mimic3/snuh/20240609/'$study'/znorm' \
                dropout=0.1 \
                # num_loss_weight=1
                # per_gpu_batchsize=128
        done
    done
done
