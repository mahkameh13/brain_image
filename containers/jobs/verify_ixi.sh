#!/bin/bash
# CPU-only structural/content verification; no GPU allocation needed.
set -euo pipefail
WORK=/work/dldevel/momeni
apptainer exec --cleanenv --bind "$WORK:$WORK" \
  --bind "$WORK/datasets/IXI:$WORK/datasets/IXI:ro" \
  "$WORK/containers/synthstrip_1.8-gpu.sif" \
  python3 "$WORK/containers/jobs/ixi_synthstrip.py" --verify "$@"
