#!/bin/bash
# HTCondor executable: one invocation = one trajectory-capture run, followed
# by its video/contact sheet and per-run metrics (on the GPU node, where
# OpenPose is fast). The cross-model comparison runs once all runs are done,
# in analyze.sh.
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
# Condor jobs get a minimal PATH and some GPU nodes don't have nvidia-smi on
# it, so the real check is whether torch (what the models use) sees a GPU.
export PATH="$PATH:/usr/bin:/usr/local/bin:/usr/local/nvidia/bin"
echo "[run_job] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
if ! "$VENV_CATVTON/bin/python" -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit(1)
p = torch.cuda.get_device_properties(0)
print(f'[run_job] torch sees {p.name}, {p.total_memory / 2**30:.1f} GiB')"; then
  echo "[run_job] FATAL: torch sees no GPU on $(hostname) (check request_gpus/requirements in vton.sub)" >&2
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
"$VENV_CATVTON/bin/python" scripts/metrics.py --run-dir "outputs/$RUN_NAME"
echo "[run_job] done: outputs/$RUN_NAME"
