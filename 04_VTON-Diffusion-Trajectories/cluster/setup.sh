#!/bin/bash
# One-time setup on the ESAT login/submit node (needs outbound internet;
# condor execute nodes are not guaranteed to have it, so jobs run offline
# against what this script downloads). Idempotent: every stage skips work
# that is already done, so just rerun it after a failure.
#
# Usage:
#   bash cluster/setup.sh                 # everything
#   bash cluster/setup.sh envs weights    # only some stages
# Stages: tools repos envs weights data check
#
# Needs ~70GB free in the project dir: model weights ~45GB (IDM-VTON alone
# ~29GB), three Python envs ~15GB, VITON-HD test split ~3GB (plus the 5.3GB
# archive while extracting).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PROJECT_ROOT"

STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(tools repos envs weights data check)
want() { [[ " ${STAGES[*]} " == *" $1 "* ]]; }
log() { echo "[setup] $*"; }

# Upstream model repos, pinned to the commits this project was developed against.
declare -A REPO_URLS=(
  [CatVTON]=https://github.com/Zheng-Chong/CatVTON.git
  [OOTDiffusion]=https://github.com/levihsu/OOTDiffusion.git
  [IDM-VTON]=https://github.com/yisol/IDM-VTON.git
)
declare -A REPO_COMMITS=(
  [CatVTON]=7818397f25613beedb3d861a34769f607cfcf3b1
  [OOTDiffusion]=13ef0faba266cdde9febc8ad39be2395bbb89d9c
  [IDM-VTON]=0d5f3ec2d737487a9bb24e4100936ad254780383
)

if want tools; then
  if command -v uv >/dev/null; then
    log "uv found: $(command -v uv)"
  else
    log "installing uv into .tools/"
    mkdir -p .tools/bin
    curl -LsSf https://astral.sh/uv/install.sh \
      | env UV_INSTALL_DIR="$PROJECT_ROOT/.tools/install" UV_NO_MODIFY_PATH=1 sh
    ln -sf "$(find "$PROJECT_ROOT/.tools/install" -name uv -type f | head -1)" .tools/bin/uv
  fi
  uv --version
fi

if want repos; then
  mkdir -p repos
  for name in CatVTON OOTDiffusion IDM-VTON; do
    if [ ! -d "repos/$name/.git" ]; then
      log "cloning $name"
      git clone -q "${REPO_URLS[$name]}" "repos/$name"
    fi
    git -C "repos/$name" -c advice.detachedHead=false checkout -q "${REPO_COMMITS[$name]}"
    log "repos/$name @ ${REPO_COMMITS[$name]:0:10}"
  done
fi

build_env() {  # <venv dir> <python version> <requirements file>
  local venv="$1" py="$2" req="$3"
  if [ -x "$venv/bin/python" ]; then
    log "$(basename "$venv") exists, skipping (delete it to rebuild)"
    return
  fi
  log "building $(basename "$venv") (python $py, $req)"
  uv venv -q --python "$py" "$venv"
  # shellcheck disable=SC2086
  uv pip install -q --python "$venv/bin/python" $TORCH_SPEC --index-url "$TORCH_INDEX_URL"
  uv pip install -q --python "$venv/bin/python" -r "$req"
}

if want envs; then
  build_env "$VENV_CATVTON" 3.12 cluster/requirements/catvton.txt
  build_env "$VENV_OOTD" 3.12 cluster/requirements/ootd.txt
  build_env "$VENV_IDM" 3.10 cluster/requirements/idm.txt
fi

if want weights; then
  log "downloading model weights into $HF_HOME and repos/OOTDiffusion/checkpoints"
  "$VENV_CATVTON/bin/python" - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

ootd = Path("repos/OOTDiffusion")
jobs = [
    # CatVTON: loaded by repo id at runtime (see repos/CatVTON/model/pipeline.py)
    dict(repo_id="booksforcharlie/stable-diffusion-inpainting",
         allow_patterns=["model_index.json", "scheduler/*", "unet/*"]),
    dict(repo_id="stabilityai/sd-vae-ft-mse"),
    dict(repo_id="zhengchong/CatVTON"),
    # OOTDiffusion: loaded from local paths; HD (upper-body) checkpoint only
    dict(repo_id="levihsu/OOTDiffusion", local_dir=str(ootd),
         allow_patterns=["checkpoints/ootd/model_index.json", "checkpoints/ootd/feature_extractor/*",
                         "checkpoints/ootd/scheduler/*", "checkpoints/ootd/text_encoder/*",
                         "checkpoints/ootd/tokenizer/*", "checkpoints/ootd/vae/*",
                         "checkpoints/ootd/ootd_hd/*"]),
    dict(repo_id="openai/clip-vit-large-patch14", local_dir=str(ootd / "checkpoints/clip-vit-large-patch14"),
         allow_patterns=["*.json", "*.txt", "model.safetensors"]),
    # OpenPose body model for the POSE metric (scripts/pose.py)
    dict(repo_id="lllyasviel/Annotators", local_dir=str(ootd / "checkpoints/openpose/ckpts"),
         allow_patterns=["body_pose_model.pth"]),
    # IDM-VTON: loaded by repo id; skip its preprocessing models (we use the
    # dataset's precomputed densepose/masks, like its inference.py does)
    dict(repo_id="yisol/IDM-VTON",
         allow_patterns=["model_index.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*",
                         "text_encoder/*", "text_encoder_2/*", "image_encoder/*", "vae/*",
                         "unet/*", "unet_encoder/*"]),
]
for kw in jobs:
    print(f"[setup]   {kw['repo_id']}", flush=True)
    snapshot_download(**kw)
PY
fi

if want data; then
  if [ -d "$DATA_ROOT/image" ] && [ -d "$DATA_ROOT/agnostic-mask-catvton" ]; then
    log "dataset present at $DATA_ROOT"
  else
    log "downloading VITON-HD (zhengchong/VITON-HD mirror, 5.3GB) and extracting the test split"
    mkdir -p data
    "$VENV_CATVTON/bin/python" -c "
from huggingface_hub import hf_hub_download
hf_hub_download('zhengchong/VITON-HD', 'zalando-hd-resized.tar.gz', repo_type='dataset', local_dir='data')"
    tar -xzf data/zalando-hd-resized.tar.gz -C data \
      zalando-hd-resized/test zalando-hd-resized/test_pairs.txt zalando-hd-resized/test_unpairs.txt
    rm -f data/zalando-hd-resized.tar.gz
    rm -rf data/.cache
  fi
fi

if want check; then
  log "checking imports (torch.cuda.is_available() False is expected on a login node)"
  "$VENV_CATVTON/bin/python" -c "
import sys, torch, diffusers
sys.path[:0] = ['repos/CatVTON', 'scripts']
from model.pipeline import CatVTONPipeline
import metrics, pose
assert pose.BODY_WEIGHTS.exists(), pose.BODY_WEIGHTS
print('  .venv      ok  torch', torch.__version__, 'diffusers', diffusers.__version__, 'cuda', torch.cuda.is_available())"
  "$VENV_OOTD/bin/python" -c "
import sys, torch, diffusers
sys.path[:0] = ['repos/OOTDiffusion', 'repos/OOTDiffusion/ootd']
from ootd.pipelines_ootd.pipeline_ootd import OotdPipeline
print('  .venv-ootd ok  torch', torch.__version__, 'diffusers', diffusers.__version__)"
  "$VENV_IDM/bin/python" -c "
import sys, torch, diffusers
sys.path[:0] = ['repos/IDM-VTON']
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline
from src.unet_hacked_tryon import UNet2DConditionModel
print('  .venv-idm  ok  torch', torch.__version__, 'diffusers', diffusers.__version__)"
  missing=0
  while read -r pair _category person cloth; do
    [[ -z "$pair" || "$pair" == \#* ]] && continue
    for f in "image/$person.jpg" "cloth/$cloth.jpg" "cloth-mask/$cloth.jpg" "agnostic-mask-catvton/$person.png" \
             "agnostic-mask/${person}_mask.png" "image-densepose/$person.jpg" \
             "image-parse-v3/$person.png" "openpose_json/${person}_keypoints.json"; do
      [ -f "$DATA_ROOT/$f" ] || { echo "  MISSING $DATA_ROOT/$f"; missing=1; }
    done
  done < cluster/pairs.txt
  [ $missing -eq 0 ] && log "all pair inputs present" || { log "some inputs missing"; exit 1; }
  log "setup complete -- next: bash cluster/submit.sh"
fi
