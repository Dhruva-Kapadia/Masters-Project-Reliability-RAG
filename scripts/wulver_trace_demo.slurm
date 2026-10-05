#!/bin/bash
#SBATCH --job-name=RRAG-trace-demo
#SBATCH --output=log/slurm-trace-%j.out
#SBATCH --error=log/slurm-trace-%j.err
#SBATCH --account=ss797
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#
# Small demo run that shows the model's thinking steps (Query -> Thought -> Output)
# for a few questions. Same account/partition/QOS as wulver_run_one.slurm; no GPU
# needed -- inference goes to the shared gpt-oss-120b vLLM server.
#
# Submit from the repo root (the log/ dir must exist):
#   sbatch scripts/wulver_trace_demo.slurm
# Override anything with --export, e.g.
#   sbatch --export=ALL,MAX_SAMPLES=5,DEFENSE=sampleMIS,ATTACK=none scripts/wulver_trace_demo.slurm
#
# Result:
#   log/slurm-trace-<jobid>.out           the rendered traces are printed here
#   runs_trace_<jobid>/traces/*.jsonl     raw traces (every LLM call)
#   runs_trace_<jobid>/traces/*.md        readable Markdown version
#   runs_trace_<jobid>/traces/*_calls.csv one row per LLM call

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# ---- demo settings (override with sbatch --export=ALL,VAR=value) ----------
DATASET="${DATASET:-open_nq}"          # open_nq | realtimeqa | ...
DEFENSE="${DEFENSE:-keyword}"          # keyword | none | sampling_keyword | graph | MIS | sampleMIS | ...
ATTACK="${ATTACK:-PIA}"                # none | PIA | Poison
MAX_SAMPLES="${MAX_SAMPLES:-3}"        # number of questions -- keep small for a demo
TOP_K="${TOP_K:-5}"                    # docs per question; keyword makes TOP_K+1 LLM calls per question
GAMMA="${GAMMA:-0.9}"
SHOW_PROMPTS="${SHOW_PROMPTS:-1}"      # 0 = hide the prompt text of each step in the printout
export GPTOSS_REASONING_EFFORT="${GPTOSS_REASONING_EFFORT:-}"   # optional: low | medium | high

RUN_DIR="${RUN_DIR:-runs_trace_${SLURM_JOB_ID}}"
mkdir -p "$RUN_DIR"/{log,output,result,traces}

export HF_HOME="${HF_HOME:-/project/ss797/dk694/hf_cache}"   # NLI model for graph/MIS/sampleMIS
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

echo "=== Trace demo: dataset=$DATASET defense=$DEFENSE attack=$ATTACK samples=$MAX_SAMPLES top_k=$TOP_K run_dir=$RUN_DIR ==="

# ---- 1. pre-flight: is the shared server up, and does it return reasoning? ----
uv run python - <<'PY'
import os
from src.vllm_config import make_client, load_connection
from src.models import SharedVLLMModel
conn = load_connection()
print(f"vLLM server: {conn['base_url']}")
kw = {}
if os.environ.get("GPTOSS_REASONING_EFFORT"):
    kw["extra_body"] = {"reasoning_effort": os.environ["GPTOSS_REASONING_EFFORT"]}
chat = make_client().chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "What is 17 + 25? Answer with the number only."}],
    max_tokens=512, temperature=0, **kw)
msg = chat.choices[0].message
thinking = SharedVLLMModel._extract_reasoning(msg)
print(f"pre-flight answer  : {msg.content!r}")
print(f"pre-flight thinking: {thinking[:200]!r}" if thinking else
      "pre-flight thinking: NOT returned as a separate field -- will fall back to splitting raw "
      "'analysis...assistantfinal' text; if traces show no Thought, the server needs a reasoning parser")
PY

# ---- 2. the run: --save_traces, and NO --use_cache (cached answers have no thinking) ----
ATTACK_ARGS=()
if [[ "$ATTACK" != "none" ]]; then
  ATTACK_ARGS=(--attack_method "$ATTACK" --corruption_size 1 --attackpos 0)
fi
GRAPH_ARGS=()
if [[ "$DEFENSE" == "graph" || "$DEFENSE" == "MIS" || "$DEFENSE" == "sampleMIS" ]]; then
  GRAPH_ARGS=(--log_graphs)
fi

uv run python main.py \
  --model_name gpt-oss-120b \
  --dataset_name "$DATASET" \
  --top_k "$TOP_K" \
  --defense_method "$DEFENSE" \
  --gamma "$GAMMA" \
  --T 10 --m 2 \
  --rep 1 \
  --max_samples "$MAX_SAMPLES" \
  --use_open_model_api \
  --save_traces \
  --save_response \
  --run_dir "$RUN_DIR" \
  "${ATTACK_ARGS[@]}" "${GRAPH_ARGS[@]}"

# ---- 3. render the traces and print them into this job's .out file ----------
TRACE_FILE=$(ls -t "$RUN_DIR"/traces/*.jsonl | head -1)
RENDER_ARGS=()
[[ "$SHOW_PROMPTS" == "0" ]] && RENDER_ARGS=(--no_prompt)
uv run python scripts/render_traces.py "$TRACE_FILE" --csv "${TRACE_FILE%.jsonl}_calls.csv" "${RENDER_ARGS[@]}"

echo
echo "################################ THINKING TRACES ################################"
cat "${TRACE_FILE%.jsonl}.md"
echo "#################################################################################"
echo
echo "Accuracy / ASR for this run:"
cat "$RUN_DIR"/output/*.csv
echo
echo "=== Done. Raw traces: $TRACE_FILE ==="
