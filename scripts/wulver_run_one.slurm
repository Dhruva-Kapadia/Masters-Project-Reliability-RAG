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
# No --gres=gpu here: inference runs against the shared gpt-oss-120b vLLM
# server, not a local GPU (see CONNECT_TO_SHARED_VLLM.md, usage rule 4:
# run heavy client code inside a Slurm job, not on the login node).
#
# --account/--qos are required: dk694's default account
# (2026-fall-ds-733-amr239-dk694) only carries QOS=course, which is denied
# on the general partition. ss797 (the PI allocation this project uses)
# carries QOS debug/low/standard -- standard has a 3-day wall-time cap,
# comfortably above this job's --time.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# RUN_DIR is normally set by scripts/wulver_benchmark.sh (sbatch --export),
# which also points this job's own --output/--error slurm logs at
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

# NLI model for graph / MIS / sampleMIS (src/nli_config.py). Cached on /project
# so it is downloaded once (bash scripts/prefetch_nli.sh on a login node) and
# shared by every array task; compute nodes then load it from disk.
export HF_HOME="${HF_HOME:-/project/ss797/dk694/hf_cache}"
# DeBERTa runs on CPU here (no --gres=gpu): match torch threads to the
# allocation instead of letting it oversubscribe the node.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

CONFIG_LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" scripts/benchmark_configs.txt)
read -r DATASET DEFENSE ATTACK <<< "$CONFIG_LINE"

echo "=== Task $SLURM_ARRAY_TASK_ID: dataset=$DATASET defense=$DEFENSE attack=$ATTACK run_dir=$RUN_DIR ==="

# Cap dataset size for a "representative, not exhaustive" benchmark; override
# with MAX_SAMPLES env var if you want the full dataset.
MAX_SAMPLES="${MAX_SAMPLES:-200}"

ATTACK_ARGS=()
if [[ "$ATTACK" != "none" ]]; then
  ATTACK_ARGS=(--attack_method "$ATTACK" --corruption_size 1 --attackpos 0)
fi

uv run python main.py \
  --model_name gpt-oss-120b \
  --dataset_name "$DATASET" \
  --top_k 10 \
  --defense_method "$DEFENSE" \
  --gamma 0.9 \
  --T 10 --m 2 \
  --rep 1 \
  --max_samples "$MAX_SAMPLES" \
  --use_open_model_api \
  --use_cache \
  --save_response \
  --debug \
  --log_graphs \
  --run_dir "$RUN_DIR" \
  "${ATTACK_ARGS[@]}"

echo "=== Task $SLURM_ARRAY_TASK_ID done ==="
