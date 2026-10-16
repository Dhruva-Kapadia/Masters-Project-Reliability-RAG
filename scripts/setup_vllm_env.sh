#!/bin/bash
#SBATCH --job-name=vllm-setup
#SBATCH --output=log/setup-vllm-%j.out
#SBATCH --account=ss797
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#
# One-time install of vLLM (for scripts/serve_gpt_oss_local.slurm) into your
# own virtual environment on /project -- not $HOME, whose quota is too small
# for vLLM + PyTorch + CUDA libraries (~10 GB).
#
# Option A -- as a Slurm job (recommended; login nodes kill heavy processes):
#   mkdir -p log && sbatch scripts/setup_vllm_env.sh
#   tail -f log/setup-vllm-<jobid>.out
#   (if it fails right away with network errors, compute nodes have no
#    internet here -- use option B)
# Option B -- on a login node:
#   bash scripts/setup_vllm_env.sh
#
# Settings (env or .env): VLLM_VENV (where to install, default below; the
# server job reads the same variable), VLLM_VERSION (e.g. 0.11.0; default:
# latest release), PYTHON_VERSION (3.12). Re-running reuses the environment
# and only installs what is missing.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source scripts/_load_env.sh

VENV="${VLLM_VENV:-/project/ss797/dk694/envs/vllm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VLLM_SPEC="vllm${VLLM_VERSION:+==$VLLM_VERSION}"
# keep uv's download cache and managed Pythons on /project too
export UV_CACHE_DIR="${UV_CACHE_DIR:-/project/ss797/dk694/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/project/ss797/dk694/.local/uv-python}"

command -v uv >/dev/null || { echo "uv not found on PATH (it is what runs main.py here); install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }

echo "=== $(date -Is) installing $VLLM_SPEC into $VENV on $(hostname) ==="
mkdir -p "$(dirname "$VENV")"
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv "$VENV" --python "$PYTHON_VERSION" --seed
fi
uv pip install --python "$VENV/bin/python" "$VLLM_SPEC"

echo "--- check ---"
"$VENV/bin/python" - <<'PY'
import torch, vllm
print(f"vllm {vllm.__version__} | torch {torch.__version__} (CUDA {torch.version.cuda})")
PY
du -sh "$VENV"
echo "=== $(date -Is) done. Start the server: sbatch scripts/serve_gpt_oss_local.slurm ==="
[[ "$VENV" == /project/ss797/dk694/envs/vllm ]] || echo "(non-default location: put VLLM_VENV=$VENV in .env so the server job finds it)"
