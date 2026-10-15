#!/bin/bash
# Run ReliabilityRAG on your own machine against Groq's hosted gpt-oss-120b
# (no Wulver, no Slurm, no GPU). The answer model, the grader and the LLM judge
# all use Groq (main.py switches the grader/judge automatically for
# --model_name gpt-oss-120b-groq).
#
# Wulver/server configs (gpt-oss-120b, qwen2.5-32b, gpt-oss-120b-local and the
# scripts/wulver_*.sh/.slurm files) are untouched by this.
#
# Setup (once):
#   uv sync                                # same env as on the server
#   export GROQ_API_KEY=gsk_...            # https://console.groq.com/keys
#
# ── Mode 1: one config ──────────────────────────────────────────────────────
#   bash scripts/run_local_groq.sh [DATASET] [DEFENSE] [ATTACK] [MAX_SAMPLES] [POSITION]
#   bash scripts/run_local_groq.sh                              # realtimeqa keyword none 10
#   bash scripts/run_local_groq.sh realtimeqa MIS PIA 5         # poison at the 1st position
#   bash scripts/run_local_groq.sh realtimeqa MIS Poison 5 10   # poison at the 10th position
#
#   DEFENSE:  none keyword sampling astuterag instructrag_icl graph MIS sampling_keyword sampleMIS
#             (decoding needs a local HF model and doesn't work with API models)
#   ATTACK:   none PIA Poison
#   POSITION: 1-based rank of the poisoned document among the top-10 (default 1)
#   Each call gets its own local_runs/runN/ (log/ output/ result/ cache/). After
#   each run, plots of ALL single-config local runs so far are (re)written to
#   local_runs/plots/ (scripts/plot_results.py).
#
# ── Mode 2: sample experiment (sweep) ───────────────────────────────────────
#   bash scripts/run_local_groq.sh sweep [DATASET] [MAX_SAMPLES]
#   bash scripts/run_local_groq.sh sweep                        # realtimeqa, 3 questions per config
#   bash scripts/run_local_groq.sh sweep open_nq 5
#
#   Runs every combination of
#     attacks    Poison, PIA                    (SWEEP_ATTACKS="Poison PIA")
#     positions  1st, 5th, 10th of the top-10   (SWEEP_POSITIONS="1 5 10", 1-based)
#     defenses   MIS, sampleMIS, graph, none    (SWEEP_DEFENSES="MIS sampleMIS graph none")
#   = 2 x 3 x 4 = 24 configs, all in one local_runs/sweepN/ (log/ output/
#   result/ cache/), plots in local_runs/sweepN/plots/ with each attack position
#   shown separately. Each config also writes its thinking-trace JSON to
#   log/<config>.traces.json.
#
#   The sweep shares one LLM response cache: a clean document gets the same
#   per-document prompt whatever the poison position, and MIS / graph ask the
#   same per-document prompts, so those answers are reused across configs
#   (and their cached thinking is still logged). Start from a fresh cache with
#   SWEEP_NO_CACHE=1.
#
#   Resume: a config whose result CSV already has a row is skipped, so after a
#   rate-limit failure re-run with  SWEEP_DIR=local_runs/sweepN  to finish it.
#   Failed configs are listed at the end; the sweep keeps going past them.
#
# graph / MIS / sampleMIS download a ~1.7 GB NLI model from HuggingFace on first
# use (cached afterwards); `sampling` uses a small local embedding model unless
# OPENAI_API_KEY is set (see NLI_MODEL / SAMPLING_EMBEDDER in src/defense.py).
#
# Groq rate limits: MIS / graph / sampleMIS make ~11 LLM calls per question (+1
# grading call), so keep MAX_SAMPLES small. 429s are retried with backoff
# automatically (GROQ_MAX_RETRIES, default 8); a call that still fails is
# scored as "I don't know", so check the log for "query failed" before
# trusting numbers.

set -euo pipefail
cd "$(dirname "$0")/.."

# Windows (Git Bash) defaults Python's file encoding to cp1252, which can't
# read the UTF-8 datasets; force UTF-8 mode everywhere (harmless on Linux/macOS).
export PYTHONUTF8=1

[[ -n "${GROQ_API_KEY:-}" ]] || { echo "GROQ_API_KEY is not set (export GROQ_API_KEY=gsk_...)" >&2; exit 1; }

MODEL=gpt-oss-120b-groq
TOP_K=10
GAMMA=0.9

# run_config RUN_DIR DATASET DEFENSE ATTACK MAX_SAMPLES POSITION(1-based) [extra main.py args...]
# Same hyperparameters as scripts/wulver_run_one.slurm, so results are comparable.
run_config() {
  local run_dir="$1" dataset="$2" defense="$3" attack="$4" max_samples="$5" position="$6"
  shift 6
  local attack_args=()
  if [[ "$attack" != "none" ]]; then
    attack_args=(--attack_method "$attack" --corruption_size 1 --attackpos "$((position - 1))")
  fi
  uv run python main.py \
    --model_name "$MODEL" \
    --dataset_name "$dataset" \
    --top_k "$TOP_K" \
    --defense_method "$defense" \
    --gamma "$GAMMA" \
    --T 10 --m 2 \
    --rep 1 \
    --max_samples "$max_samples" \
    --use_open_model_api \
    --save_response \
    --debug \
    --run_dir "$run_dir" \
    "${attack_args[@]}" "$@"
}

# Result CSV that main.py writes for a config (mirrors src/helper.py:get_log_name)
result_csv() {
  local run_dir="$1" dataset="$2" defense="$3" attack="$4" position="$5" dstr
  case "$defense" in
    none|graph|MIS|astuterag|instructrag_icl) dstr="$defense" ;;
    sampleMIS) dstr="" ;;   # get_log_name has no sampleMIS branch
    *) echo ""; return ;;   # not needed by the sweep
  esac
  echo "$run_dir/output/${dataset}-${MODEL}-${dstr}-rep1-top${TOP_K}-attack${attack}-gamma${GAMMA}-attackpos$((position - 1))-corruptionsz1.csv"
}

next_dir() {  # next free local_runs/<prefix>N
  local n=1
  while [[ -d "local_runs/$1$n" ]]; do n=$((n + 1)); done
  echo "local_runs/$1$n"
}

# ─────────────────────────────── sweep ───────────────────────────────
if [[ "${1:-}" == "sweep" ]]; then
  DATASET="${2:-realtimeqa}"
  MAX_SAMPLES="${3:-3}"
  read -r -a ATTACKS   <<< "${SWEEP_ATTACKS:-Poison PIA}"
  read -r -a POSITIONS <<< "${SWEEP_POSITIONS:-1 5 10}"
  read -r -a DEFENSES  <<< "${SWEEP_DEFENSES:-MIS sampleMIS graph none}"
  for p in "${POSITIONS[@]}"; do
    (( p >= 1 && p <= TOP_K )) || { echo "position $p is outside 1..$TOP_K" >&2; exit 1; }
  done

  SWEEP_DIR="${SWEEP_DIR:-$(next_dir sweep)}"
  mkdir -p "$SWEEP_DIR"/{log,output,result,cache}
  CACHE_ARGS=(--use_cache)
  [[ "${SWEEP_NO_CACHE:-0}" == "1" ]] && CACHE_ARGS=()

  TOTAL=$(( ${#ATTACKS[@]} * ${#POSITIONS[@]} * ${#DEFENSES[@]} ))
  { echo "model=$MODEL"; echo "started=$(date -Iseconds)"
    echo "mode=sweep dataset=$DATASET max_samples=$MAX_SAMPLES configs=$TOTAL"
    echo "attacks=${ATTACKS[*]} positions(1-based)=${POSITIONS[*]} defenses=${DEFENSES[*]}"
    echo "cache=$([[ ${#CACHE_ARGS[@]} -gt 0 ]] && echo shared || echo off)"; } >> "$SWEEP_DIR/run_info.txt"

  echo "=== sweep: $TOTAL configs, $MAX_SAMPLES questions each, dataset=$DATASET -> $SWEEP_DIR ==="
  i=0; FAILED=(); SKIPPED=0
  for attack in "${ATTACKS[@]}"; do
    for pos in "${POSITIONS[@]}"; do
      for defense in "${DEFENSES[@]}"; do
        i=$((i + 1))
        tag="[$i/$TOTAL] attack=$attack position=$pos defense=$defense"
        csv="$(result_csv "$SWEEP_DIR" "$DATASET" "$defense" "$attack" "$pos")"
        if [[ -n "$csv" && -f "$csv" && $(wc -l < "$csv") -gt 1 ]]; then
          echo "$tag: already done, skipping"; SKIPPED=$((SKIPPED + 1)); continue
        fi
        echo "=== $tag ==="
        if run_config "$SWEEP_DIR" "$DATASET" "$defense" "$attack" "$MAX_SAMPLES" "$pos" "${CACHE_ARGS[@]}"; then
          echo "$(date -Iseconds) ok     $attack pos=$pos $defense" >> "$SWEEP_DIR/run_info.txt"
        else
          echo "$(date -Iseconds) FAILED $attack pos=$pos $defense" >> "$SWEEP_DIR/run_info.txt"
          FAILED+=("$attack pos=$pos $defense")
        fi
      done
    done
  done

  echo "=== sweep done: $((TOTAL - SKIPPED - ${#FAILED[@]})) ran, $SKIPPED skipped, ${#FAILED[@]} failed -> $SWEEP_DIR ==="
  if (( ${#FAILED[@]} )); then
    printf '  failed: %s\n' "${FAILED[@]}"
    echo "  re-run the missing ones with:  SWEEP_DIR=$SWEEP_DIR bash scripts/run_local_groq.sh sweep $DATASET $MAX_SAMPLES"
  fi
  uv run python scripts/plot_results.py "$SWEEP_DIR" --by_position --out_dir "$SWEEP_DIR/plots" \
    && echo "=== plots + results table: $SWEEP_DIR/plots/ ==="
  (( ${#FAILED[@]} == 0 ))
  exit $?
fi

# ───────────────────────────── one config ─────────────────────────────
DATASET="${1:-realtimeqa}"
DEFENSE="${2:-keyword}"
ATTACK="${3:-none}"
MAX_SAMPLES="${4:-10}"
POSITION="${5:-1}"
(( POSITION >= 1 && POSITION <= TOP_K )) || { echo "POSITION must be in 1..$TOP_K" >&2; exit 1; }

RUN_DIR="$(next_dir run)"
mkdir -p "$RUN_DIR"/{log,output,result,cache}
{ echo "model=$MODEL"; echo "started=$(date -Iseconds)"
  echo "dataset=$DATASET defense=$DEFENSE attack=$ATTACK max_samples=$MAX_SAMPLES position=$POSITION"; } > "$RUN_DIR/run_info.txt"

echo "=== $MODEL: dataset=$DATASET defense=$DEFENSE attack=$ATTACK position=$POSITION max_samples=$MAX_SAMPLES -> $RUN_DIR ==="
run_config "$RUN_DIR" "$DATASET" "$DEFENSE" "$ATTACK" "$MAX_SAMPLES" "$POSITION" --use_cache
echo "=== done: results in $RUN_DIR/output/ ==="

# Each local run is one config, so plot all single-config local runs together.
# Configs run more than once are averaged; delete old local_runs/runN/ to drop them.
uv run python scripts/plot_results.py local_runs/run* --out_dir local_runs/plots \
  && echo "=== plots (all local runs so far): local_runs/plots/ ==="
