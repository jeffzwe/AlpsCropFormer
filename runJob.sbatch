#!/bin/bash -l

#############################
### Define Job Parameters ###
#############################
#SBATCH --account="go57"                # the group account
#SBATCH --job-name="TESTING"           # the name of the job
#SBATCH --time=08:00:00                 # max time allocated to the job
#SBATCH --nodes=1                       # number of nodes to use
#SBATCH --ntasks=4                  # For each GPU, we use 1 task
#SBATCH --cpus-per-task=16          # all CPU threads on the node for data loading
#SBATCH --gres=gpu:4                    # GPUs per node
# --mail-type=BEGIN, END, FAIL    # receive an email when the job begins, ends or fails
#############################

# Change directory to your project's working directory
cd "$SCRATCH/021_AlpsCropFormer"

# Webdataset version
chmod +x ./runTraining.sh
srun --environment=cropformer_web --container-workdir=$PWD --cpu-bind=cores bash -c "./runTraining.sh"

# SQFS version
# chmod +x ./runPreprocessing.sh
# srun --environment=cropformer_sqfs --container-mounts=$SCRATCH --container-workdir=$PWD --cpu-bind=cores bash -c "./runPreprocessing.sh"