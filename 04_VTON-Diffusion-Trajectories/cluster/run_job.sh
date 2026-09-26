#!/bin/bash
# HTCondor executable: one invocation = one trajectory-capture run, followed
# by its video/contact sheet. Metrics + cross-model comparison happen once
# all runs are done, in analyze.sh.
#
# Usage: run_job.sh <catvton|ootd|idm> <pair name> <person stem> <cloth stem>
#   -> outputs/<model>_<pair name>/
#
# Each model gets the inputs its own upstream VITON-HD test path uses:
#   catvton: agnostic-mask-catvton/<person>.png as the mask
#   ootd:    mask derived from image-parse-v3 + openpose_json (see ootd_trajectory.py)
#   idm:     agnostic-mask/<person>_mask.png + image-densepose/<person>.jpg
set -euo pipefail
MODEL="$1"; PAIR="$2"; PERSON="$3"; CLOTH="$4"
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1  # everything was fetched by setup.sh; execute nodes may have no internet

RUN_NAME="${MODEL}_${PAIR}"
PERSON_IMG="$DATA_ROOT/image/$PERSON.jpg"
CLOTH_IMG="$DATA_ROOT/cloth/$CLOTH.jpg"

echo "[run_job] host=$(hostname) run=$RUN_NAME person=$PERSON cloth=$CLOTH"
if ! nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; then
  echo "[run_job] FATAL: no GPU visible to this job (check request_gpus/requirements in vton.sub)" >&2
  exit 1
fi

case "$MODEL" in
  catvton)
    "$VENV_CATVTON/bin/python" scripts/catvton_trajectory.py \
      --person "$PERSON_IMG" --cloth "$CLOTH_IMG" \
      --mask "$DATA_ROOT/agnostic-mask-catvton/$PERSON.png" \
      --run-name "$RUN_NAME" --steps 50 --seed 42
    ;;
  ootd)
    "$VENV_OOTD/bin/python" scripts/ootd_trajectory.py \
      --person "$PERSON_IMG" --cloth "$CLOTH_IMG" \
      --run-name "$RUN_NAME" --steps 50 --seed 42
    ;;
  idm)
    "$VENV_IDM/bin/python" scripts/idmvton_trajectory.py \
      --person "$PERSON_IMG" --cloth "$CLOTH_IMG" \
      --run-name "$RUN_NAME" --steps 50 --seed 42
    ;;
  *)
    echo "[run_job] unknown model '$MODEL' (expected catvton|ootd|idm)" >&2
    exit 1
    ;;
esac

"$VENV_CATVTON/bin/python" scripts/viz.py --run-dir "outputs/$RUN_NAME"
echo "[run_job] done: outputs/$RUN_NAME"
