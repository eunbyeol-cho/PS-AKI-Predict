# gpu_id=0
# seed=2020

# OMP_NUM_THREADS=8 \
# NUMEXPR_MAX_THREADS=128 \
# CUDA_VISIBLE_DEVICES=${gpu_id} \
#   python main.py with task_test_table2table \
#   seed=${seed}


# mask_schedulers="pow1 pow0.5 cosine exp"
# mask_schedulers="pow1 pow0.5"
# seeds="2020 2021 2022"
# n_iters="20 30"
# unmask_methods="random"

gpu_id=7
mask_scheduler="pow1"
seeds="1 2"
n_iter="20"
unmask_by="random"
study="SNUH"

# unmask_methods="random"
# trg_orgs='BRMH'
trg_orgs='BRMH KNUH CAUH'
trg_orgs='AMC SNUBH'
# trg_orgs = ['SNUH', 'SNUBH', 'BRMH', 'KNUH', 'CAUH', 'KUMC', 'AMC']
# Loop through each mask_scheduler and seed combination
for target_study in $trg_orgs; do
  for seed in $seeds; do
      OMP_NUM_THREADS=8 \
      NUMEXPR_MAX_THREADS=128 \
      CUDA_VISIBLE_DEVICES=$gpu_id \
      python main.py with \
          task_test_table2table_$study \
          debug=True \
          wandb_project_name="mmtg-20240505" \
          seed=$seed \
          mask_scheduler=$mask_scheduler \
          n_iter=$n_iter \
          null_sample=True \
          unmask_by=$unmask_by \
          input_path='/home/data_storage/mimic3/snuh/20240609/'$study'/znorm' \
          boxcox_transformation=True \
          dropout=0.1 \
          fixed_test_mask=True \
          target_study=$target_study
  done
done
