#!/bin/bash
# HTCondor executable for the benchmark scores: one invocation = one shard of
# one (model, sampler, setting), final images only (scripts/benchmark_generate.py).
#
# Usage: benchmark_job.sh <catvton|ootd|idm> <sampler> <unpaired|paired> <shard> <n shards>
#   -> outputs/_benchmark/<model>/<sampler>/<setting>/
# Images that already exist are skipped, so a retry after a walltime kill
# only does what's left.
set -euo pipefail
MODEL="$1"; SAMPLER="$2"; SETTING="$3"; SHARD="$4"; NSHARDS="$5"
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1  # everything was fetched by setup.sh; execute nodes may have no internet

echo "[benchmark_job] host=$(hostname) $MODEL/$SAMPLER/$SETTING shard $SHARD/$NSHARDS"
export PATH="$PATH:/usr/bin:/usr/local/bin:/usr/local/nvidia/bin"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
if ! "$VENV_CATVTON/bin/python" -c "import sys, torch; sys.exit(not torch.cuda.is_available())"; then
  echo "[benchmark_job] FATAL: torch sees no GPU on $(hostname)" >&2
  exit 1
fi

case "$MODEL" in
  catvton) PY="$VENV_CATVTON/bin/python" ;;
  ootd)    PY="$VENV_OOTD/bin/python" ;;
  idm)     PY="$VENV_IDM/bin/python" ;;
  *) echo "[benchmark_job] unknown model '$MODEL' (expected catvton|ootd|idm)" >&2; exit 1 ;;
esac
"$PY" scripts/benchmark_generate.py --model "$MODEL" --sampler "$SAMPLER" --setting "$SETTING" \
  --shard "$SHARD/$NSHARDS" ${LIMIT:+--limit "$LIMIT"}
