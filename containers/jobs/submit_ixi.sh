#!/bin/bash
# Usage: bash submit_ixi.sh [--limit 1]; rerun without --limit to resume all T1s.
set -euo pipefail
WORK=/work/dldevel/momeni
SIF=$WORK/containers/synthstrip_1.8-gpu.sif
test -s "$SIF" || { echo "Build the GPU image first: bash $WORK/containers/jobs/build_synthstrip.sh" >&2; exit 1; }
test -r "$HOME/licenses/license.txt"
test -s "$WORK/containers/jobs/ixi_t1_manifest.json"
apptainer sif list "$SIF" >/dev/null
apptainer exec --cleanenv "$SIF" python3 -c 'import torch, nibabel, surfa; assert torch.version.cuda, "Image needs CUDA-enabled PyTorch"; print("PyTorch", torch.__version__, "CUDA", torch.version.cuda)'
mkdir -p "$WORK/logs"
sbatch --parsable "$WORK/containers/jobs/skullstrip_gpu.sbatch" "$@"
