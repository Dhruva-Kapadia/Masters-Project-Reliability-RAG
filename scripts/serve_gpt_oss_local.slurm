#!/bin/bash
#SBATCH --job-name=gpt-oss-server
#SBATCH --output=log/vllm-server-%j.out
#SBATCH --error=log/vllm-server-%j.err
#SBATCH --account=ss797
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00   # 72 h = the standard QOS maximum; getting GPUs takes hours, so keep one server for many runs
#
# Serve YOUR OWN copy of gpt-oss-120b (downloaded with scripts/download_gpt_oss.sh)
# with vLLM on a GPU node, for --model_name gpt-oss-120b-local. Submit from the
# repo root:
#
#   mkdir -p log                                     # Slurm writes this job's log there
#   sbatch scripts/serve_gpt_oss_local.slurm
#   squeue -u $USER                                  # wait for RUNNING
#   cat ~/vllm_local_server.txt                      # appears once the model is loaded, e.g. n0042:8123
#   bash scripts/wulver_benchmark.sh sweep submit    # experiments now use this server
#   scancel <jobid>                                  # when you're done (frees the 4 GPUs)
#
# The address file is written only after the model has loaded and answers
# /v1/models (loading takes several minutes), and is removed when the job
# ends, so clients never talk to a half-started or dead server. Clients
# (src/models.py, the grader and the LLM judge) read it on every request.
#
# Settings (env or .env): GPT_OSS_DIR (weights), LOCAL_VLLM_SERVER_FILE
# (address file, default ~/vllm_local_server.txt), PORT (default: a free port),
# TP (tensor-parallel GPUs, default: all GPUs of the job), QUANT (default
# mxfp4, the checkpoint's native format; set QUANT= to let vLLM decide),
# MAX_MODEL_LEN (32768), GPU_MEM_UTIL (0.90), VLLM_VENV (vLLM install from
# scripts/setup_vllm_env.sh, default /project/ss797/dk694/envs/vllm) or
# VLLM_ENV=<conda env> to use conda instead, CUDA_MODULE (CUDA/12.8.0; its nvcc
# is needed for vLLM's kernel JIT), CACHE_ROOT (compile caches, /project/ss797/dk694/.cache). Fewer GPUs: sbatch --gres=gpu:2 ...
# (mxfp4 weights are ~65 GB; 2x80 GB leaves less room for the KV cache).
# One-time setup: bash scripts/download_gpt_oss.sh (weights) and
#   sbatch scripts/setup_vllm_env.sh (vLLM). Flags follow HPC_GPTOSS_GUIDE.md, section 3.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/_load_env.sh

MODEL_DIR="${GPT_OSS_DIR:-/project/ss797/dk694/models/gpt-oss-120b}"
SERVER_FILE="${LOCAL_VLLM_SERVER_FILE:-$HOME/vllm_local_server.txt}"
SERVED_NAME="openai/gpt-oss-120b"   # the name src/models.py, the grader and the judge request
QUANT="${QUANT-mxfp4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

[[ -f "$MODEL_DIR/config.json" ]] || {
  echo "No model at $MODEL_DIR (config.json missing). Run on a login node first: bash scripts/download_gpt_oss.sh" >&2; exit 1; }

# vLLM environment: your own venv from scripts/setup_vllm_env.sh (default), or
# a conda env if you name one with VLLM_ENV (e.g. VLLM_ENV=vllm_env).
VLLM_VENV="${VLLM_VENV:-/project/ss797/dk694/envs/vllm}"
set +u   # activation scripts reference unset variables
if [[ -n "${VLLM_ENV:-}" ]]; then
  CONDA_SH="${CONDA_SH:-/apps/easybuild/el9_5.x86_64/software/Miniforge3/24.11.3-0/etc/profile.d/conda.sh}"
  source "$CONDA_SH"
  conda activate "$VLLM_ENV"
elif [[ -x "$VLLM_VENV/bin/vllm" ]]; then
  source "$VLLM_VENV/bin/activate"
else
  echo "vLLM is not installed at $VLLM_VENV. Install it once with: sbatch scripts/setup_vllm_env.sh" >&2
  echo "(or use an existing conda env: VLLM_ENV=<name> sbatch scripts/serve_gpt_oss_local.slurm)" >&2
  exit 1
fi

# CUDA toolkit: vLLM JIT-compiles some kernels (FlashInfer, Triton) while warming
# up -- AFTER the ~10 min model load -- and needs nvcc + CUDA_HOME for that. The
# pip wheels bring the CUDA runtime but not the compiler, so load the module.
CUDA_MODULE="${CUDA_MODULE:-CUDA/12.8.0}"
if ! type module >/dev/null 2>&1; then   # batch shells sometimes lack the Lmod function
  for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
    [[ -f "$f" ]] && { source "$f"; break; }
  done
fi
module load "$CUDA_MODULE" || { echo "could not 'module load $CUDA_MODULE' (see: module avail CUDA; set CUDA_MODULE=...)" >&2; exit 1; }
if ! command -v nvcc >/dev/null; then
  echo "nvcc not found after 'module load $CUDA_MODULE' -- vLLM's warm-up would fail with 'Could not find nvcc'" >&2; exit 1
fi
export CUDA_HOME="${CUDA_HOME:-${EBROOTCUDA:-$(dirname "$(dirname "$(command -v nvcc)")")}}"
set -u
echo "CUDA toolkit: $CUDA_HOME ($(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p'))"

# JIT / compile caches go to /project, not the small $HOME quota (reused by later jobs)
CACHE_ROOT="${CACHE_ROOT:-/project/ss797/dk694/.cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CACHE_ROOT/vllm}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$CACHE_ROOT}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor}"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
export HF_HOME="${HF_HOME:-/project/ss797/dk694/hf_cache}"
export HF_HUB_OFFLINE=1          # weights are local; never reach out to the Hub from the GPU node
export NCCL_TIMEOUT=3600

NGPU=$(nvidia-smi -L | wc -l)
TP="${TP:-$NGPU}"
PORT="${PORT:-$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')}"
ADDR="$(hostname):${PORT}"

QUANT_ARGS=()
[[ -n "$QUANT" ]] && QUANT_ARGS=(--quantization "$QUANT")

echo "=== $(date -Is) serving $MODEL_DIR as $SERVED_NAME on $ADDR (TP=$TP of $NGPU GPUs, vllm $(vllm --version 2>/dev/null || echo '?')) ==="

# Remove the address file when the job ends (scancel, time limit, crash) --
# but only if it still points at this server.
rm -f "$SERVER_FILE"
cleanup() {
  if [[ -f "$SERVER_FILE" && "$(cat "$SERVER_FILE")" == "$ADDR" ]]; then rm -f "$SERVER_FILE"; fi
  [[ -n "${VLLM_PID:-}" ]] && kill "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 143' TERM INT

vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  "${QUANT_ARGS[@]}" &
VLLM_PID=$!

# Publish the address only once the model answers (use /v1/models; /health is
# unreliable on some vLLM builds).
echo "waiting for the model to load ..."
for _ in $(seq 1 360); do      # up to 60 min
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited before becoming ready -- see log/vllm-server-${SLURM_JOB_ID}.err" >&2; exit 1
  fi
  if curl -s --max-time 5 "http://localhost:${PORT}/v1/models" | grep -q "$SERVED_NAME"; then
    echo "$ADDR" > "${SERVER_FILE}.tmp" && mv "${SERVER_FILE}.tmp" "$SERVER_FILE"
    echo "=== $(date -Is) READY: $ADDR written to $SERVER_FILE ==="
    break
  fi
  sleep 10
done
[[ -f "$SERVER_FILE" ]] || { echo "model did not become ready within 60 min" >&2; exit 1; }

wait "$VLLM_PID"
