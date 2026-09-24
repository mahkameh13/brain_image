#!/bin/bash

#SBATCH --job-name=autoencoder
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --mem=32000
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --output=logs/slurm/run_%j.out
#SBATCH --error=logs/slurm/run_%j.out

# Get random run id.
run_id=$(head /dev/urandom | tr -dc 'a-z0-9' | head -c 6)
mkdir -p logs/hydra
mkdir -p logs/wandb

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate latent_mri

# GPU info
export CUBLAS_WORKSPACE_CONFIG=:16:8
nvidia-smi

# Show hydra exceptions
HYDRA_FULL_ERROR=1

# Override params and run code
srun python -u train.py \
    hydra/job_logging=disabled \
    +job_id="$SLURM_JOB_ID" \
    +run_id="$run_id" \

