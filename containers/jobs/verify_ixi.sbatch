#!/bin/bash
#SBATCH --job-name=ixi-verify
#SBATCH --account=dldevel
#SBATCH --partition=test
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/work/dldevel/momeni/logs/ixi-verify-%j.out
#SBATCH --error=/work/dldevel/momeni/logs/ixi-verify-%j.err
set -euo pipefail
WORK=/work/dldevel/momeni
export APPTAINERENV_SLURM_JOB_ID=$SLURM_JOB_ID
export APPTAINERENV_SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export APPTAINERENV_OMP_NUM_THREADS=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
apptainer exec --cleanenv --bind "$WORK:$WORK" \
  --bind "$WORK/datasets/IXI:$WORK/datasets/IXI:ro" \
  "$WORK/containers/synthstrip_1.8-gpu.sif" \
  python3 "$WORK/containers/jobs/verify_ixi_parallel.py"
apptainer exec --cleanenv --bind "$WORK:$WORK" \
  "$WORK/containers/synthstrip_1.8-gpu.sif" \
  python3 "$WORK/containers/jobs/ixi_qc.py" export "$WORK/datasets/IXI_skullstripped/qc_sample.npz"
MPLCONFIGDIR=/tmp/momeni-matplotlib "$WORK/env/bin/python" \
  "$WORK/containers/jobs/ixi_qc.py" render "$WORK/datasets/IXI_skullstripped/qc_sample.npz"
