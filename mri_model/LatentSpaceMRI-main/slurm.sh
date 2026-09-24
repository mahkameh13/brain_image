#!/bin/bash

#SBATCH --job-name=dimensionality
#SBATCH --partition=general1
#SBATCH --nodes=1
#SBATCH --mem=256000
#SBATCH --time=240:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/run_%j.out
#SBATCH --error=logs/slurm/run_%j.out

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate latent_mri

# Override params and run code
srun python -u scripts/dimensionality_IXI.py

