#!/bin/bash
# Per-run metrics (whole frame + mask crop) for every finished run in
# outputs/, then the cross-model comparison. Runs as the final DAG node
# after all trajectory jobs (see submit.sh), or by hand at any time:
#   bash cluster/analyze.sh
# Runs whose job failed have no run_config.json and are skipped.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PROJECT_ROOT"
PY="$VENV_CATVTON/bin/python"

runs=()
for d in outputs/catvton_* outputs/ootd_* outputs/idm_*; do
  [ -f "$d/run_config.json" ] || continue
  runs+=("$d")
  "$PY" scripts/metrics.py --run-dir "$d"
  "$PY" scripts/metrics.py --run-dir "$d" --crop-to-mask
done
[ ${#runs[@]} -gt 0 ] || { echo "[analyze] no finished runs in outputs/"; exit 1; }

"$PY" scripts/compare_trajectories.py "${runs[@]}"
"$PY" scripts/compare_trajectories.py "${runs[@]}" --crop-to-mask
echo "[analyze] done: outputs/_comparison/onset_summary{,_masked}.csv and per-pair plots"
