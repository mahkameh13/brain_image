#!/bin/bash
# Run on the login node; no GPU allocation is needed.
set -euo pipefail
WORK=/work/dldevel/momeni
SIF=$WORK/containers/synthstrip_1.8-gpu.sif
if [[ -s "$SIF" ]]; then
  apptainer sif list "$SIF"
  exit 0
fi
export APPTAINER_IGNORE_PROOT=1
export APPTAINER_TMPDIR=/tmp/momeni-synthstrip-build
mkdir -p "$APPTAINER_TMPDIR"
apptainer build --mksquashfs-args '-processors 4 -mem 1G' \
  "$SIF" docker://freesurfer/synthstrip:1.8-gpu
apptainer sif list "$SIF"
