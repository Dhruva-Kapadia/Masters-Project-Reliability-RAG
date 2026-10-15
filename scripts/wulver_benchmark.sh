#!/bin/bash
# Enumerates all (dataset, defense, attack) configs for a benchmark of one
# vLLM-served model (gpt-oss-120b, qwen2.5-32b, or gpt-oss-120b-local) and submits one Slurm array
# job to run them.
#
# Each submission gets its own runsN/ folder (auto-incrementing: runs1,
# runs2, ...) holding that submission's log/, output/, result/ and cache/ --
# so repeated or partial benchmark submissions never mix or overwrite each
# other's files.
#
# Scope (see conversation with Claude, 2026-09-24):
#   - datasets: realtimeqa, open_nq  (both are proper free-form QA and load correctly)
#   - defenses: all EXCEPT `voting` (voting needs the -mc dataset variant,
#     whose format/metric isn't directly comparable to the rest) and
#     `decoding` (needs a local HF model, see below)
#   - attacks: none (clean baseline) and PIA
#
# Usage:
#   bash scripts/wulver_benchmark.sh                          # dry run (gpt-oss-120b): print config list and job count
#   bash scripts/wulver_benchmark.sh submit                   # allocate the next runsN/ and sbatch (gpt-oss-120b)
#   bash scripts/wulver_benchmark.sh submit qwen2.5-32b       # same sweep against the Qwen2.5-32B-Instruct server
#   bash scripts/wulver_benchmark.sh dry qwen2.5-32b          # dry run for Qwen
#   bash scripts/wulver_benchmark.sh submit gpt-oss-120b-local # your own server (scripts/serve_gpt_oss_local.slurm must be running)
# The model is recorded in <runsN>/run_info.txt and in the model_name column of every result CSV.

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-dry}"
MODEL_NAME="${2:-gpt-oss-120b}"
case "$MODEL_NAME" in
  gpt-oss-120b|qwen2.5-32b|gpt-oss-120b-local) ;;
  *) echo "Unknown model '$MODEL_NAME' (expected gpt-oss-120b, qwen2.5-32b or gpt-oss-120b-local)" >&2; exit 1 ;;
esac

DATASETS=(realtimeqa open_nq)
# `decoding` is excluded: it needs token-level probabilities from a locally
# loaded HF model, which API/vLLM-server models don't expose (main.py exits
# with an explanation if you try).
DEFENSES=(none keyword sampling astuterag instructrag_icl graph MIS sampling_keyword sampleMIS)
ATTACKS=(none PIA)

CONFIG_FILE="scripts/benchmark_configs.txt"
> "$CONFIG_FILE"

for dataset in "${DATASETS[@]}"; do
  for defense in "${DEFENSES[@]}"; do
    for attack in "${ATTACKS[@]}"; do
      echo "${dataset} ${defense} ${attack}" >> "$CONFIG_FILE"
    done
  done
done

N=$(wc -l < "$CONFIG_FILE")
echo "Wrote $N configs to $CONFIG_FILE"
echo "(${#DATASETS[@]} datasets x ${#DEFENSES[@]} defenses x ${#ATTACKS[@]} attacks) for model: $MODEL_NAME"

if [[ "$MODE" == "submit" ]]; then
  # Find the next unused runsN directory (runs1, runs2, ...).
  RUN_NUM=1
  while [[ -d "runs${RUN_NUM}" ]]; do
    RUN_NUM=$((RUN_NUM + 1))
  done
  RUN_DIR="runs${RUN_NUM}"
  mkdir -p "$RUN_DIR"/{log,output,result,cache}
  echo "Allocated $RUN_DIR/ for this submission's log/output/result/cache."
  { echo "model=$MODEL_NAME"; echo "submitted=$(date -Is)"; echo "configs=$N"; } > "$RUN_DIR/run_info.txt"

  echo "Submitting Slurm array job (array size $N, max 4 concurrent)..."
  ARRAY_JOB=$(sbatch --parsable \
    --array=1-"$N"%4 \
    --output="${RUN_DIR}/log/slurm-%A_%a.out" \
    --error="${RUN_DIR}/log/slurm-%A_%a.err" \
    --export=ALL,RUN_DIR="$RUN_DIR",MODEL_NAME="$MODEL_NAME" \
    scripts/wulver_run_one.slurm)
  ARRAY_JOB="${ARRAY_JOB%%;*}"
  echo "Submitted array job $ARRAY_JOB"

  # When every task has finished (even if some failed), build summary.csv and
  # the plots. Uses the same account/partition/qos as the array job.
  hdr() { sed -n "s/^#SBATCH --$1=//p" scripts/wulver_run_one.slurm | head -1; }
  AGG_JOB=$(sbatch --parsable \
    --dependency=afterany:"$ARRAY_JOB" \
    --job-name=RRAG-plots \
    --account="$(hdr account)" --partition="$(hdr partition)" --qos="$(hdr qos)" \
    --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:20:00 \
    --output="${RUN_DIR}/log/aggregate-%j.out" \
    --wrap "uv run python scripts/aggregate_results.py --run_dir ${RUN_DIR}")
  echo "Submitted aggregate+plot job ${AGG_JOB%%;*} (runs after $ARRAY_JOB)"

  echo "Results will land under ${RUN_DIR}/output/, ${RUN_DIR}/log/, ${RUN_DIR}/result/."
  echo "Summary + plots: ${RUN_DIR}/output/summary.csv and ${RUN_DIR}/output/plots/"
  echo "(re-run any time: uv run python scripts/aggregate_results.py --run_dir ${RUN_DIR})"
else
  echo "Dry run only. Re-run as: bash scripts/wulver_benchmark.sh submit $MODEL_NAME"
fi