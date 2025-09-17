#!/bin/bash
set -e  # Exit on error

############################
### Print Slurm Commands ###
############################
if [ "$SLURM_NODEID" == "0" ]; then
    echo "===== SLURM ENVIRONMENT VARIABLES ====="
    env | grep ^SLURM_
    echo "======================================="
fi
############################

# Activate your virtual environment inside the container
source .venv/bin/activate

# Run the actual preprocessing script
python data/Sentinel/data_preprocessing.py
