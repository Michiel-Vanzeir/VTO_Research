# Sourced by setup.sh, run_job.sh and analyze.sh -- one place for every path
# and cache location, all kept INSIDE the project dir so that (a) a small
# $HOME quota is never hit (~70GB of weights/envs/data) and (b) condor
# execute nodes, which see the same shared filesystem, find everything at
# the same paths as the submit node.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT

export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_ROOT/.uv_cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$PROJECT_ROOT/.uv_python}"
export PATH="$PROJECT_ROOT/.tools/bin:$PATH"

# One CUDA build of torch for all three envs. cu124 wheels need an NVIDIA
# driver >= 525; if `nvidia-smi` on a GPU node reports an older driver, set
# e.g. TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 before setup.sh.
export TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
export TORCH_SPEC="torch==2.6.0 torchvision==0.21.0"
# NVIDIA wheel host (pypi.nvidia.com) is slow at times; uv's 30s default times out
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"

DATA_ROOT="$PROJECT_ROOT/data/zalando-hd-resized/test"
VENV_CATVTON="$PROJECT_ROOT/.venv"        # CatVTON + all analysis scripts
VENV_OOTD="$PROJECT_ROOT/.venv-ootd"
VENV_IDM="$PROJECT_ROOT/.venv-idm"
