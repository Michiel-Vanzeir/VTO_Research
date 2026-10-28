#!/bin/bash
# Submit the benchmark generation (final images on the full VITON-HD test
# split, for FID / KID / SSIM / LPIPS) plus the metrics as one HTCondor DAG.
# Run on the submit node after setup.sh:
#
#   bash cluster/submit_benchmark.sh                     # every config, both settings
#   CONFIGS="idm:ddpm30" bash cluster/submit_benchmark.sh   # subset (model:sampler ...)
#   SETTINGS="paired" bash cluster/submit_benchmark.sh   # one setting
#   SHARDS=8 bash cluster/submit_benchmark.sh            # jobs per (config, setting); default 4
#   LIMIT=5 bash cluster/submit_benchmark.sh             # test: first 5 pairs only, metrics on partial folders
#   GPU_NAME="NVIDIA GeForce RTX 3090 Ti" bash cluster/submit_benchmark.sh   # only on that card
#   DRY_RUN=1 bash cluster/submit_benchmark.sh           # write the DAG, don't submit
#
# Configs: CatVTON with DDIM 50 (its default and the trajectory sampler);
# OOTDiffusion and IDM-VTON with DDIM 50 (trajectory sampler) and with their
# own defaults, UniPC 20 and DDPM 30. Images that already exist are skipped,
# so resubmitting after a failure only does what's missing.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PROJECT_ROOT/cluster"

CONFIGS="${CONFIGS:-catvton:ddim50 ootd:ddim50 ootd:unipc20 idm:ddim50 idm:ddpm30}"
SETTINGS="${SETTINGS:-unpaired paired}"
SHARDS="${SHARDS:-4}"
LIMIT="${LIMIT:-}"
GPU_NAME="${GPU_NAME:-}"
RETRIES=2
declare -A GPUMEM=([catvton]=11000 [ootd]=11000 [idm]=23000)  # >= 11 GB also keeps the slow 8 GB cards out
declare -A MEM=([catvton]=16GB [ootd]=24GB [idm]=40GB)
WALLTIME=28800  # 8 h; at 4 shards the slowest job (IDM-VTON, DDIM 50: 508 images x 20 s) is ~3 h on a 3090 Ti

mkdir -p logs generated
DAG=generated/benchmark.dag
if [ -z "${DRY_RUN:-}" ] && command -v condor_q >/dev/null \
   && condor_q "$USER" -constraint 'JobUniverse == 7' -af Args 2>/dev/null | grep -q "benchmark.dag"; then
  echo "[submit_benchmark] a benchmark DAG is still running (condor_q -dag -nobatch)." >&2
  exit 1
fi

# Inception (torch-fidelity) and AlexNet (lpips) weights for the metrics node,
# fetched here because execute nodes may have no internet
if [ -z "${DRY_RUN:-}" ]; then
  "$VENV_CATVTON/bin/python" -c "
import lpips, torch_fidelity
lpips.LPIPS(net='alex', verbose=False)
torch_fidelity.feature_extractor_inceptionv3.FeatureExtractorInceptionV3('inception-v3-compat', ['2048'])
print('[submit_benchmark] metric weights cached in', __import__('torch').hub.get_dir())"
fi

: > "$DAG"
nodes=()
for cfg in $CONFIGS; do
  model="${cfg%%:*}"; sampler="${cfg#*:}"
  [ -n "${GPUMEM[$model]:-}" ] || { echo "unknown model '$model'" >&2; exit 1; }
  for setting in $SETTINGS; do
    for ((k = 0; k < SHARDS; k++)); do
      node="bench_${model}_${sampler}_${setting}_${k}of${SHARDS}"
      cat >> "$DAG" <<EOF
JOB $node benchmark.sub
VARS $node model="$model" sampler="$sampler" setting="$setting" shard="$k" nshards="$SHARDS" limit="$LIMIT" gpuname="$GPU_NAME" logname="$node" walltime="$WALLTIME" gpumem="${GPUMEM[$model]}" mem="${MEM[$model]}"
RETRY $node $RETRIES
SCRIPT POST $node post.sh \$RETURN \$RETRY $RETRIES
EOF
      nodes+=("$node")
    done
  done
done
echo "JOB bench_metrics benchmark_metrics.sub" >> "$DAG"
echo "VARS bench_metrics metrics_args=\"${LIMIT:+--allow-partial}\" logname=\"bench_metrics\"" >> "$DAG"
echo "PARENT ${nodes[*]} CHILD bench_metrics" >> "$DAG"
echo "[submit_benchmark] $DAG: ${#nodes[@]} generation job(s) + metrics"

if [ -n "${DRY_RUN:-}" ]; then
  cat "$DAG"
  exit 0
fi
command -v condor_submit_dag >/dev/null || { echo "condor_submit_dag not found -- run this on an ESAT condor submit node" >&2; exit 1; }
condor_submit_dag -f "$DAG"
echo "[submit_benchmark] watch with: condor_q -dag -nobatch   |   tail -f cluster/logs/bench_<model>_<sampler>_<setting>_<k>of<n>.out"
