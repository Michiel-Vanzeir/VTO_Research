#!/bin/bash
# Cross-model comparison over every finished run in outputs/. Runs as the
# final DAG node after all trajectory jobs (see submit.sh), or by hand:
#   bash cluster/analyze.sh
# Per-run metrics are normally already computed by each job (run_job.sh);
# metrics.py only recomputes runs whose metrics.json is missing or outdated.
# Runs whose job failed have no run_config.json and are skipped.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PROJECT_ROOT"
PY="$VENV_CATVTON/bin/python"

for d in outputs/catvton_* outputs/ootd_* outputs/idm_*; do
  if [ -f "$d/run_config.json" ]; then
    "$PY" scripts/metrics.py --run-dir "$d"
  fi
done
"$PY" scripts/compare_trajectories.py
"$PY" scripts/change_decomposition.py   # within-model: which scales change when
"$PY" scripts/validate_axes.py          # sanity check that each metric axis measures its own property
echo "[analyze] done: see outputs/_comparison/ and outputs/_analysis/"
