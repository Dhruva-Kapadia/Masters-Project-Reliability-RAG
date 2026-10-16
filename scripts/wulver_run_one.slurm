#!/bin/bash
#SBATCH --job-name=RRAG-bench
# --output/--error below are only a bootstrap fallback for a bare `sbatch
# scripts/wulver_run_one.slurm` (Slurm needs a static path before the script
# runs and can compute RUN_DIR). scripts/wulver_benchmark.sh always submits
# with --output/--error pointed at <RUN_DIR>/log/ instead, overriding these.
#SBATCH --output=log/slurm-%A_%a.out
#SBATCH --error=log/slurm-%A_%a.err
#SBATCH --account=ss797
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
# No --gres=gpu here: inference runs against a gpt-oss-120b vLLM server --
# by default your own (scripts/serve_gpt_oss_local.slurm, address in
# ~/vllm_local_server.txt; answers, grading and judging all use it), not a
# local GPU. Run heavy client code inside a Slurm job, not on the login node.
#
# --account/--qos are required: dk694's default account
# (2026-fall-ds-733-amr239-dk694) only carries QOS=course, which is denied
# on the general partition. ss797 (the PI allocation this project uses)
# carries QOS debug/low/standard -- standard has a 3-day wall-time cap,
# comfortably above this job's --time.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# RUN_DIR, MODEL_NAME, CONFIG_FILE, MAX_SAMPLES and USE_CACHE are normally set
# by scripts/wulver_benchmark.sh (sbatch --export). Each line of CONFIG_FILE is
#   <dataset> <defense> <attack> [<position, 1-based, default 1>]
# The wrapper also points this job's own --output/--error slurm logs at
# <RUN_DIR>/log/. Falls back to a fresh runsN/ if this script is sbatch'd
# directly without going through the wrapper.
if [[ -z "${RUN_DIR:-}" ]]; then
  RUN_NUM=1
  while [[ -d "runs${RUN_NUM}" ]]; do
    RUN_NUM=$((RUN_NUM + 1))
  done
  RUN_DIR="runs${RUN_NUM}"
  echo "RUN_DIR not set (direct sbatch?) -- defaulting to $RUN_DIR"
fi

mkdir -p "$RUN_DIR"/{log,output,result,cache}

# NLI model for graph / MIS / sampleMIS. Cached on /project so it is downloaded
# once (bash scripts/prefetch_nli.sh on a login node) and shared by every array
# task; compute nodes then load it from disk. HF_TOKEN (if needed) comes from
# .env, which main.py loads.
export HF_HOME="${HF_HOME:-/project/ss797/dk694/hf_cache}"
# DeBERTa runs on CPU here (no --gres=gpu): match torch threads to the
# allocation instead of letting it oversubscribe the node.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

MODEL_NAME="${MODEL_NAME:-gpt-oss-120b-local}"
CONFIG_FILE="${CONFIG_FILE:-scripts/benchmark_configs.txt}"
# Cap dataset size for a "representative, not exhaustive" benchmark; the sweep
# passes its own (small) MAX_SAMPLES.
MAX_SAMPLES="${MAX_SAMPLES:-200}"
CACHE_ARGS=(--use_cache)
[[ "${USE_CACHE:-1}" == "1" ]] || CACHE_ARGS=()

CONFIG_LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CONFIG_FILE")
read -r DATASET DEFENSE ATTACK POSITION <<< "$CONFIG_LINE"
POSITION="${POSITION:-1}"
[[ -n "$DATASET" && -n "$DEFENSE" && -n "$ATTACK" ]] || { echo "bad config line $SLURM_ARRAY_TASK_ID in $CONFIG_FILE: '$CONFIG_LINE'" >&2; exit 1; }

echo "=== Task $SLURM_ARRAY_TASK_ID: model=$MODEL_NAME dataset=$DATASET defense=$DEFENSE attack=$ATTACK position=$POSITION max_samples=$MAX_SAMPLES run_dir=$RUN_DIR ==="

ATTACK_ARGS=()
if [[ "$ATTACK" != "none" ]]; then
  ATTACK_ARGS=(--attack_method "$ATTACK" --corruption_size 1 --attackpos "$((POSITION - 1))")
fi

# Same hyperparameters as scripts/run_local_groq.sh, so results are comparable.
# Thinking traces go to $RUN_DIR/log/<config>.traces.json (src/json_trace_log.py).
uv run python main.py \
  --model_name "$MODEL_NAME" \
  --dataset_name "$DATASET" \
  --top_k 10 \
  --defense_method "$DEFENSE" \
  --gamma 0.9 \
  --T 10 --m 2 \
  --rep 1 \
  --max_samples "$MAX_SAMPLES" \
  --use_open_model_api \
  --save_response \
  --debug \
  --run_dir "$RUN_DIR" \
  "${CACHE_ARGS[@]}" "${ATTACK_ARGS[@]}"

echo "=== Task $SLURM_ARRAY_TASK_ID done ==="
