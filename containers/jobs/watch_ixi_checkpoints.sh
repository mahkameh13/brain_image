#!/bin/bash
# Lightweight login-node watcher: submit CPU validation after each 100 outputs.
set -euo pipefail
WORK=/work/dldevel/momeni
DEST=$WORK/datasets/IXI_skullstripped
STATE=$WORK/containers/jobs/checkpoint_state
GPU_JOB=1774631
mkdir -p "$STATE"
for CHECKPOINT in 200 300 400 500; do
  MARKER=$STATE/checkpoint-$CHECKPOINT.submitted
  if [[ -s "$MARKER" ]]; then
    continue
  fi
  while true; do
    COUNT=$(find "$DEST" -type f -name '*_synthstrip.json' | wc -l)
    if (( COUNT >= CHECKPOINT )); then
      JOB=$(sbatch --parsable "$WORK/containers/jobs/checkpoint_ixi.sbatch" "$CHECKPOINT")
      printf '%s\n' "$JOB" > "$MARKER"
      printf '%s checkpoint=%s outputs=%s validation_job=%s\n' "$(date -Iseconds)" "$CHECKPOINT" "$COUNT" "$JOB"
      while [[ -n "$(squeue -h -j "$JOB")" ]]; do
        sleep 5
      done
      CHECK_STATE=$(sacct -X -j "$JOB" -n -o State | awk 'NF {print $1; exit}')
      printf '%s checkpoint=%s validation_job=%s state=%s\n' "$(date -Iseconds)" "$CHECKPOINT" "$JOB" "$CHECK_STATE"
      cat "$WORK/logs/ixi-checkpoint-$JOB.out"
      if [[ "$CHECK_STATE" != COMPLETED ]]; then
        cat "$WORK/logs/ixi-checkpoint-$JOB.err" >&2
        exit 1
      fi
      break
    fi
    STATE_NOW=$(sacct -X -j "$GPU_JOB" -n -o State | awk 'NF {print $1; exit}')
    case "$STATE_NOW" in
      RUNNING|PENDING|COMPLETING|'') ;;
      *) printf '%s GPU job ended in state %s before checkpoint %s (outputs=%s)\n' "$(date -Iseconds)" "$STATE_NOW" "$CHECKPOINT" "$COUNT" >&2; exit 1 ;;
    esac
    sleep 30
  done
done
printf '%s all milestone validation jobs submitted\n' "$(date -Iseconds)"
