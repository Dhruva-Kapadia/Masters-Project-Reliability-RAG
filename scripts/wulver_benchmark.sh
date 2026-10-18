#!/bin/bash
# Model server: by default the experiments use YOUR OWN gpt-oss-120b server
# (--model_name gpt-oss-120b-local). One-time setup, then start it:
#   bash scripts/download_gpt_oss.sh          # login node: weights -> /project/ss797/dk694/models/gpt-oss-120b
#   mkdir -p log && sbatch scripts/serve_gpt_oss_local.slurm
#   cat ~/vllm_local_server.txt               # appears once the model is loaded
# Submitting checks that the server answers first (SKIP_SERVER_CHECK=1 to skip).
#
# Submit ReliabilityRAG experiments on Wulver as one Slurm array job (one task
# per config, scripts/wulver_run_one.slurm) against a vLLM-served model
# (gpt-oss-120b, qwen2.5-32b, or gpt-oss-120b-local), followed by a job that
# aggregates the results and draws the plots.
#
# Each submission gets its own runsN/ folder (auto-incrementing: runs1,
# runs2, ...) holding that submission's log/, output/, result/, cache/ and its
# config list (configs.txt), so repeated or partial submissions never mix or
# overwrite each other's files. Every config also writes its thinking-trace
# JSON to <runsN>/log/<config>.traces.json.
#
# ── Experiment 1: sample experiment (sweep) -- same as run_local_groq.sh sweep ──
#   bash scripts/wulver_benchmark.sh sweep                                # dry run: list configs
#   bash scripts/wulver_benchmark.sh sweep submit                         # your gpt-oss-120b server, realtimeqa, 3 questions/config
#   bash scripts/wulver_benchmark.sh sweep submit gpt-oss-120b-local open_nq 50  # [MODEL] [DATASET] [MAX_SAMPLES]
#
#   Runs every combination of
#     attacks    Poison, PIA                    (SWEEP_ATTACKS="Poison PIA")
#     positions  1st, 5th, 10th of the top-10   (SWEEP_POSITIONS="1 5 10", 1-based)
#     defenses   MIS, sampleMIS, graph, none    (SWEEP_DEFENSES="MIS sampleMIS graph none")
#   = 2 x 3 x 4 = 24 configs. Plots keep each poisoned position separate
#   ("PIA @ pos 5"). The LLM response cache is shared by all tasks of the
#   submission (safe: writes are merged under a lock); SWEEP_NO_CACHE=1 turns
#   it off so every config gets fresh answers.
#
# ── Experiment 2: full benchmark ──
#   bash scripts/wulver_benchmark.sh                           # dry run
#   bash scripts/wulver_benchmark.sh submit                    # your own gpt-oss-120b server (default)
#   bash scripts/wulver_benchmark.sh submit gpt-oss-120b       # the shared lab server (if it is up)
#   bash scripts/wulver_benchmark.sh submit qwen2.5-32b        # Qwen2.5-32B-Instruct server
#   Scope (see conversation with Claude, 2026-09-24):
#     - datasets: realtimeqa, open_nq  (both are proper free-form QA and load correctly)
#     - defenses: all EXCEPT `voting` (needs the -mc dataset variant, whose
#       format/metric isn't directly comparable) and `decoding` (needs a local HF model)
#     - attacks: none (clean baseline) and PIA at the 1st position
#     - MAX_SAMPLES (env) questions per config, default 200
#
# Resume / re-run failures (both experiments): point RUN_DIR at an existing
# submission and only the configs without a result row are submitted again:
#   RUN_DIR=runs7 bash scripts/wulver_benchmark.sh sweep submit
#
# Other knobs (env): MAX_CONCURRENT (array tasks at once, default 4 -- the
# shared server asks for modest concurrency), MAX_SAMPLES.
# Before the first graph / MIS / sampleMIS run: bash scripts/prefetch_nli.sh
# on a login node (HF_TOKEN from .env is used if set).
# The model is recorded in <runsN>/run_info.txt and in the model_name column of every result CSV.

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_load_env.sh   # optional paths / tokens from .env

EXPERIMENT=benchmark
if [[ "${1:-}" == "sweep" ]]; then EXPERIMENT=sweep; shift; fi
MODE="${1:-dry}"
MODEL_NAME="${2:-gpt-oss-120b-local}"
case "$MODE" in
  dry|submit) ;;
  *) echo "Unknown mode '$MODE' (expected dry or submit)" >&2; exit 1 ;;
esac
case "$MODEL_NAME" in
  gpt-oss-120b|qwen2.5-32b|gpt-oss-120b-local) ;;
  *) echo "Unknown model '$MODEL_NAME' (expected gpt-oss-120b, qwen2.5-32b or gpt-oss-120b-local)" >&2; exit 1 ;;
esac

TOP_K=10
GAMMA=0.9
CONFIGS=()   # "dataset defense attack position(1-based)"

if [[ "$EXPERIMENT" == "sweep" ]]; then
  DATASET="${3:-realtimeqa}"
  MAX_SAMPLES="${4:-${MAX_SAMPLES:-3}}"
  read -r -a ATTACKS   <<< "${SWEEP_ATTACKS:-Poison PIA}"
  read -r -a POSITIONS <<< "${SWEEP_POSITIONS:-1 5 10}"
  read -r -a DEFENSES  <<< "${SWEEP_DEFENSES:-MIS sampleMIS graph none}"
  for p in "${POSITIONS[@]}"; do
    (( p >= 1 && p <= TOP_K )) || { echo "position $p is outside 1..$TOP_K" >&2; exit 1; }
  done
  for attack in "${ATTACKS[@]}"; do
    for pos in "${POSITIONS[@]}"; do
      for defense in "${DEFENSES[@]}"; do
        CONFIGS+=("$DATASET $defense $attack $pos")
      done
    done
  done
  DESC="sweep: dataset=$DATASET attacks=${ATTACKS[*]} positions=${POSITIONS[*]} defenses=${DEFENSES[*]}"
  USE_CACHE=1; [[ "${SWEEP_NO_CACHE:-0}" == "1" ]] && USE_CACHE=0
  PLOT_ARGS="--by_position"
else
  MAX_SAMPLES="${MAX_SAMPLES:-200}"
  DATASETS=(realtimeqa open_nq)
  # `decoding` is excluded: it needs token-level probabilities from a locally
  # loaded HF model, which API/vLLM-server models don't expose (main.py exits
  # with an explanation if you try).
  DEFENSES=(none keyword sampling astuterag instructrag_icl graph MIS sampling_keyword sampleMIS)
  ATTACKS=(none PIA)
  for dataset in "${DATASETS[@]}"; do
    for defense in "${DEFENSES[@]}"; do
      for attack in "${ATTACKS[@]}"; do
        CONFIGS+=("$dataset $defense $attack 1")
      done
    done
  done
  DESC="benchmark: ${#DATASETS[@]} datasets x ${#DEFENSES[@]} defenses x ${#ATTACKS[@]} attacks"
  USE_CACHE=1
  PLOT_ARGS=""
fi

# Result CSV that main.py writes for a config (mirrors src/helper.py:get_log_name);
# empty for defenses whose name carries extra hyper-parameters (never skipped).
result_csv() {
  local run_dir="$1" dataset="$2" defense="$3" attack="$4" position="$5" dstr
  case "$defense" in
    none|graph|MIS|astuterag|instructrag_icl) dstr="$defense" ;;
    sampleMIS) dstr="" ;;   # get_log_name has no sampleMIS branch
    *) echo ""; return ;;
  esac
  echo "$run_dir/output/${dataset}-${MODEL_NAME}-${dstr}-rep1-top${TOP_K}-attack${attack}-gamma${GAMMA}-attackpos$((position - 1))-corruptionsz1.csv"
}

# Resume: with RUN_DIR set to an existing submission, drop configs that are done.
RESUME=0
if [[ -n "${RUN_DIR:-}" ]]; then
  [[ -d "$RUN_DIR" ]] || { echo "RUN_DIR=$RUN_DIR does not exist" >&2; exit 1; }
  RESUME=1
  TODO=()
  for c in "${CONFIGS[@]}"; do
    read -r d def a p <<< "$c"
    csv="$(result_csv "$RUN_DIR" "$d" "$def" "$a" "$p")"
    if [[ -n "$csv" && -f "$csv" && $(wc -l < "$csv") -gt 1 ]]; then continue; fi
    TODO+=("$c")
  done
  echo "Resuming $RUN_DIR: $(( ${#CONFIGS[@]} - ${#TODO[@]} )) of ${#CONFIGS[@]} configs already done"
  CONFIGS=("${TODO[@]+"${TODO[@]}"}")
fi

N=${#CONFIGS[@]}
echo "$DESC"
echo "$N configs for model $MODEL_NAME, $MAX_SAMPLES questions each:"
printf '  %s\n' "${CONFIGS[@]+"${CONFIGS[@]}"}" | awk '{printf "  %-3d dataset=%-10s defense=%-10s attack=%-6s position=%s\n", NR, $1, $2, $3, $4}'

if [[ "$MODE" != "submit" ]]; then
  echo "Dry run only. Re-run as: bash scripts/wulver_benchmark.sh $([[ $EXPERIMENT == sweep ]] && echo 'sweep ')submit $MODEL_NAME"
  exit 0
fi
if (( N == 0 )); then
  echo "Nothing left to run. Re-plot with: uv run python scripts/aggregate_results.py --run_dir $RUN_DIR $PLOT_ARGS"
  exit 0
fi

# Make sure the model server is up before queueing jobs that need it
# (a dead server makes every answer "I don't know", silently).
if [[ "${SKIP_SERVER_CHECK:-0}" != "1" ]]; then
  case "$MODEL_NAME" in
    gpt-oss-120b-local) SERVER_FILE="${LOCAL_VLLM_SERVER_FILE:-$HOME/vllm_local_server.txt}" ;;
    qwen2.5-32b)        SERVER_FILE="${QWEN_VLLM_SERVER_FILE:-/project/ss797/cs757/KidnapRAG/vllm_server.txt}" ;;
    *)                  SERVER_FILE="${SHARED_VLLM_SERVER_FILE:-/project/ss797/ap2645/vllm_server.txt}" ;;
  esac
  if [[ ! -s "$SERVER_FILE" ]]; then
    echo "Model server for $MODEL_NAME is not up: $SERVER_FILE is missing." >&2
    [[ "$MODEL_NAME" == gpt-oss-120b-local ]] && echo "Start it with: mkdir -p log && sbatch scripts/serve_gpt_oss_local.slurm (wait until the file appears)" >&2
    echo "(SKIP_SERVER_CHECK=1 to submit anyway)" >&2
    exit 1
  fi
  ADDR="$(tr -d '[:space:]' < "$SERVER_FILE")"
  if ! curl -s --max-time 10 "http://${ADDR}/v1/models" | grep -q '"id"'; then
    echo "Model server for $MODEL_NAME at $ADDR (from $SERVER_FILE) does not answer /v1/models." >&2
    echo "Check it with: squeue -u \$USER ; tail log/vllm-server-*.err   (SKIP_SERVER_CHECK=1 to submit anyway)" >&2
    exit 1
  fi
  # one real dummy query (a server can list the model yet fail to generate)
  echo "Model server for $MODEL_NAME at $ADDR is up; sending a dummy query ..."
  if ! bash scripts/check_server.sh "$ADDR" | sed 's/^/  /'; then
    echo "Model server at $ADDR did not answer the dummy query (SKIP_SERVER_CHECK=1 to submit anyway)" >&2
    exit 1
  fi
fi

# Make sure the Python environment works BEFORE queueing jobs that wait hours
# for a slot: sync it once here (the jobs themselves run with --no-sync) and
# import what main.py needs. SKIP_ENV_CHECK=1 to skip.
if [[ "${SKIP_ENV_CHECK:-0}" != "1" ]]; then
  bash scripts/check_env.sh || exit 1
fi

if (( ! RESUME )); then
  # Find the next unused runsN directory (runs1, runs2, ...).
  RUN_NUM=1
  while [[ -d "runs${RUN_NUM}" ]]; do
    RUN_NUM=$((RUN_NUM + 1))
  done
  RUN_DIR="runs${RUN_NUM}"
fi
mkdir -p "$RUN_DIR"/{log,output,result,cache}
# per-submission config list: a later submission can't change a queued array's configs
CONFIG_FILE="$RUN_DIR/configs.txt"
if (( RESUME )); then CONFIG_FILE="$RUN_DIR/configs-resume-$(date +%Y%m%d-%H%M%S).txt"; fi
printf '%s\n' "${CONFIGS[@]}" > "$CONFIG_FILE"
echo "Using $RUN_DIR/ for log/output/result/cache; configs in $CONFIG_FILE"
{ echo "model=$MODEL_NAME"; echo "submitted=$(date -Is)"; echo "experiment=$EXPERIMENT"
  echo "$DESC"; echo "configs=$N max_samples=$MAX_SAMPLES cache=$([[ $USE_CACHE == 1 ]] && echo shared || echo off)"
  echo "config_file=$CONFIG_FILE"; } >> "$RUN_DIR/run_info.txt"

MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
echo "Submitting Slurm array job (array size $N, max $MAX_CONCURRENT concurrent)..."
ARRAY_JOB=$(sbatch --parsable \
  --array=1-"$N"%"$MAX_CONCURRENT" \
  --output="${RUN_DIR}/log/slurm-%A_%a.out" \
  --error="${RUN_DIR}/log/slurm-%A_%a.err" \
  --export=ALL,RUN_DIR="$RUN_DIR",MODEL_NAME="$MODEL_NAME",CONFIG_FILE="$CONFIG_FILE",MAX_SAMPLES="$MAX_SAMPLES",USE_CACHE="$USE_CACHE" \
  scripts/wulver_run_one.slurm)
ARRAY_JOB="${ARRAY_JOB%%;*}"
echo "Submitted array job $ARRAY_JOB"
echo "array_job=$ARRAY_JOB" >> "$RUN_DIR/run_info.txt"

# When every task has finished (even if some failed), build summary.csv and
# the plots. Uses the same account/partition/qos as the array job.
hdr() { sed -n "s/^#SBATCH --$1=//p" scripts/wulver_run_one.slurm | head -1; }
AGG_JOB=$(sbatch --parsable \
  --dependency=afterany:"$ARRAY_JOB" \
  --job-name=RRAG-plots \
  --account="$(hdr account)" --partition="$(hdr partition)" --qos="$(hdr qos)" \
  --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:20:00 \
  --output="${RUN_DIR}/log/aggregate-%j.out" \
  --wrap "uv run --no-sync python scripts/aggregate_results.py --run_dir ${RUN_DIR} ${PLOT_ARGS}")
AGG_JOB="${AGG_JOB%%;*}"
echo "Submitted aggregate+plot job ${AGG_JOB} (runs after $ARRAY_JOB)"
echo "aggregate_job=$AGG_JOB" >> "$RUN_DIR/run_info.txt"

echo "Results will land under ${RUN_DIR}/output/, ${RUN_DIR}/log/ (incl. *.traces.json), ${RUN_DIR}/result/."
echo "Summary + plots: ${RUN_DIR}/output/summary.csv and ${RUN_DIR}/output/plots/"
echo "(re-plot any time: uv run python scripts/aggregate_results.py --run_dir ${RUN_DIR} ${PLOT_ARGS})"
echo "(re-submit failed configs: RUN_DIR=${RUN_DIR} bash scripts/wulver_benchmark.sh $([[ $EXPERIMENT == sweep ]] && echo 'sweep ')submit $MODEL_NAME$([[ $EXPERIMENT == sweep ]] && echo " $DATASET $MAX_SAMPLES"))"
