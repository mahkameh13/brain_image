#!/bin/bash
#SBATCH --job-name=ixi-synthstrip
#SBATCH --account=dldevel
#SBATCH --partition=gpu2
#SBATCH --qos=gpu2
#SBATCH --gres=gpu:1
#SBATCH --constraint=NVIDIA
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/work/dldevel/momeni/logs/skullstrip-%j.out
#SBATCH --error=/work/dldevel/momeni/logs/skullstrip-%j.err

set -euo pipefail
WORK=/work/dldevel/momeni
SIF=$WORK/containers/synthstrip_1.8-gpu.sif
LIC=$HOME/licenses/license.txt
test -s "$SIF" || { echo "Missing container: $SIF" >&2; exit 1; }
test -r "$LIC"
test -s "$WORK/containers/jobs/ixi_t1_manifest.json"
apptainer sif list "$SIF" >/dev/null
mkdir -p "$WORK/datasets/IXI_skullstripped"
exec 9>"$WORK/datasets/IXI_skullstripped/.run.lock"
flock -n 9 || { echo 'Another IXI run is active' >&2; exit 1; }
echo "host=$(hostname) date=$(date -Iseconds) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export APPTAINERENV_OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export APPTAINERENV_SLURM_JOB_ID=$SLURM_JOB_ID
apptainer exec --nv --cleanenv --bind "$WORK:$WORK" --bind "$WORK/datasets/IXI:$WORK/datasets/IXI:ro" --bind "$LIC:/licenses/license.txt:ro" --env FS_LICENSE=/licenses/license.txt "$SIF" python3 "$WORK/containers/jobs/ixi_synthstrip.py" --threads "$SLURM_CPUS_PER_TASK" "$@"
