#!/bin/bash
# Submit after a milestone: sbatch checkpoint_ixi.sbatch 100
#SBATCH --job-name=ixi-checkpoint
#SBATCH --account=dldevel
#SBATCH --partition=test
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/work/dldevel/momeni/logs/ixi-checkpoint-%j.out
#SBATCH --error=/work/dldevel/momeni/logs/ixi-checkpoint-%j.err
set -euo pipefail
CHECKPOINT=${1:?checkpoint count required}
case "$CHECKPOINT" in
  100|200|300|400|500) ;;
  *) echo "Unsupported checkpoint: $CHECKPOINT" >&2; exit 2 ;;
esac
WORK=/work/dldevel/momeni
START=$((CHECKPOINT - 99))
export APPTAINERENV_SLURM_JOB_ID=$SLURM_JOB_ID
export APPTAINERENV_SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export APPTAINERENV_OMP_NUM_THREADS=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
apptainer exec --cleanenv --bind "$WORK:$WORK" \
  --bind "$WORK/datasets/IXI:$WORK/datasets/IXI:ro" \
  "$WORK/containers/synthstrip_1.8-gpu.sif" \
  python3 "$WORK/containers/jobs/verify_ixi_parallel.py" \
  --start "$START" --limit 100 --tag "checkpoint-$CHECKPOINT"
