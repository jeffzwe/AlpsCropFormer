#!/bin/bash
set -e  # Stop if any command fails

############################
### Print Slurm Commands ###
############################
if [ "$SLURM_LOCALID" == "0" ]; then
    echo "===== SLURM ENVIRONMENT VARIABLES ====="
    env | grep ^SLURM_
    echo "======================================="
fi
############################

############################
### Set enviroment #########
############################
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
############################

############################
#### Set network ###########
############################
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355
############################
# EXAMPLE: Activate virtual environment *inside* the container, if needed
source "$SCRATCH/.venv/bin/activate"

# EXAMPLE: Set environment variables for distributed training
export CUDA_LAUNCH_BLOCKING=1

python  train_and_eval/seg_train_base_parallel.py \
        --config_file configs/Sentinel/TSViT_fold1.yaml

# Nsight Profiling
# nsys profile --output=nsys-$SLURM_JOB_ID \
#     --trace=cuda,osrt,nvtx \
#     --cuda-memory-usage=true \
#     --gpu-metrics-devices=cuda-visible \
#     --stop-on-exit=true \
#     --force-overwrite true \
#     --duration=500 \
#     python train_and_eval/seg_train_base_parallel.py \
#         --config_file configs/Sentinel/TSViT_fold1.yaml \
#         --device 0,1,2,3

# Single Node Training Example
# python train_and_eval/seg_train_base.py --config_file configs/Sentinel/TSViT_fold1.yaml




