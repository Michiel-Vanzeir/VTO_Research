#!/bin/bash
# Final node of the benchmark DAG (benchmark_metrics.sub), or by hand:
#   bash cluster/benchmark_metrics.sh                  # complete folders only
#   bash cluster/benchmark_metrics.sh --allow-partial  # also the 5-pair test folders
# FID/KID/SSIM/LPIPS of outputs/_benchmark/*/*/*/ -> outputs/_benchmark/metrics.json
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PROJECT_ROOT"
"$VENV_CATVTON/bin/python" scripts/benchmark_metrics.py "$@"
