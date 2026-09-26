#!/bin/bash
# Submit every (model, pair) trajectory run plus the final analysis as one
# HTCondor DAG. Run from anywhere on the submit node after setup.sh:
#
#   bash cluster/submit.sh                      # all models x all pairs in pairs.txt
#   MODELS="idm" bash cluster/submit.sh         # subset of models
#   FORCE=1 bash cluster/submit.sh              # also redo runs that already finished
#   DRY_RUN=1 bash cluster/submit.sh            # write the DAG, don't submit
#
# Runs that already have outputs/<model>_<pair>/run_config.json are skipped
# (so resubmitting after a partial failure only redoes what's missing). A
# failed run does not block the analysis node -- it just gets skipped there.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PROJECT_ROOT/cluster"

MODELS="${MODELS:-catvton ootd idm}"
# Per-model minimum GPU memory (MB) and job RAM. CatVTON (512x384) and
# OOTDiffusion (768x1024) both ran on an 8GB laptop GPU; IDM-VTON is SDXL-sized
# (two 2.6B-param UNets in fp16) and needs a 24GB-class card.
declare -A GPUMEM=([catvton]=7500 [ootd]=7500 [idm]=23000)
declare -A MEM=([catvton]=16GB [ootd]=24GB [idm]=40GB)

mkdir -p logs generated
DAG=generated/trajectories.dag
: > "$DAG"
nodes=()
while read -r pair person cloth; do
  [[ -z "$pair" || "$pair" == \#* ]] && continue
  for model in $MODELS; do
    [ -n "${GPUMEM[$model]:-}" ] || { echo "unknown model '$model'" >&2; exit 1; }
    node="${model}_${pair}"
    if [ -f "$PROJECT_ROOT/outputs/$node/run_config.json" ] && [ -z "${FORCE:-}" ]; then
      echo "[submit] skip $node (already finished; FORCE=1 to redo)"
      continue
    fi
    cat >> "$DAG" <<EOF
JOB $node vton.sub
VARS $node model="$model" pair="$pair" person="$person" cloth="$cloth" gpumem="${GPUMEM[$model]}" mem="${MEM[$model]}"
SCRIPT POST $node /bin/true
EOF
    nodes+=("$node")
  done
done < pairs.txt

echo "JOB analyze analyze.sub" >> "$DAG"
[ ${#nodes[@]} -gt 0 ] && echo "PARENT ${nodes[*]} CHILD analyze" >> "$DAG"
echo "[submit] $DAG: ${#nodes[@]} trajectory run(s) + analysis"

if [ -n "${DRY_RUN:-}" ]; then
  cat "$DAG"
  exit 0
fi
command -v condor_submit_dag >/dev/null || { echo "condor_submit_dag not found -- run this on an ESAT condor submit node" >&2; exit 1; }
condor_submit_dag -f "$DAG"
echo "[submit] watch with: condor_q -dag -nobatch   |   tail -f cluster/logs/<model>_<pair>.out"
