#!/bin/bash
# Enumerates all (dataset, defense, attack) configs for the gpt-oss-120b
# benchmark and submits one Slurm array job to run them.
#
# Each submission gets its own runsN/ folder (auto-incrementing: runs1,
# runs2, ...) holding that submission's log/, output/, result/ and cache/ --
# so repeated or partial benchmark submissions never mix or overwrite each
# other's files.
#
# Scope (see conversation with Claude, 2026-09-24):
#   - datasets: realtimeqa, open_nq  (both are proper free-form QA and load correctly)
#   - defenses: graph, MIS, sampleMIS only (rerun after fixing the NLI model
#     path; the other defenses already have results). The full set is all 10
#     EXCEPT `voting` (voting needs the -mc dataset variant, whose
#     format/metric isn't directly comparable to the rest)
#   - attacks: none (clean baseline) and PIA
#
# Usage:
#   bash scripts/wulver_benchmark.sh            # print the config list and job count
#   bash scripts/wulver_benchmark.sh submit      # allocate the next runsN/ and sbatch the array job

set -euo pipefail
cd "$(dirname "$0")/.."

DATASETS=(realtimeqa open_nq)
# Currently: only the NLI-based defenses, rerunning after the NLI model-path fix
# (see src/nli_config.py). Full set, for a complete benchmark:
#   DEFENSES=(none keyword decoding sampling astuterag instructrag_icl graph MIS sampling_keyword sampleMIS)
DEFENSES=(graph MIS sampleMIS)
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
echo "(${#DATASETS[@]} datasets x ${#DEFENSES[@]} defenses x ${#ATTACKS[@]} attacks)"

if [[ "${1:-}" == "submit" ]]; then
  # Find the next unused runsN directory (runs1, runs2, ...).
  RUN_NUM=1
  while [[ -d "runs${RUN_NUM}" ]]; do
    RUN_NUM=$((RUN_NUM + 1))
  done
  RUN_DIR="runs${RUN_NUM}"
  mkdir -p "$RUN_DIR"/{log,output,result,cache}
  echo "Allocated $RUN_DIR/ for this submission's log/output/result/cache."

  echo "Submitting Slurm array job (array size $N, max 4 concurrent)..."
  sbatch \
    --array=1-"$N"%4 \
    --output="${RUN_DIR}/log/slurm-%A_%a.out" \
    --error="${RUN_DIR}/log/slurm-%A_%a.err" \
    --export=ALL,RUN_DIR="$RUN_DIR" \
    scripts/wulver_run_one.slurm

  echo "Results will land under ${RUN_DIR}/output/, ${RUN_DIR}/log/, ${RUN_DIR}/result/."
  echo "Once done: uv run python scripts/aggregate_results.py --run_dir ${RUN_DIR}"
else
  echo "Dry run only. Re-run as: bash scripts/wulver_benchmark.sh submit"
fi