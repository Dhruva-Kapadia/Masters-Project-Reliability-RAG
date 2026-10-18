#!/bin/bash
# One command for a Wulver experiment on YOUR OWN gpt-oss-120b server:
#   1. reuses your server if it is already up -- or, if a server job is
#      already queued / loading, waits for that one -- and only otherwise
#      queues a new 72-hour server job (scripts/serve_gpt_oss_local.slurm);
#   2. a small CPU "launcher" job starts as soon as the server job starts
#      running, waits until the model has loaded and answers, then submits the
#      experiment (scripts/wulver_benchmark.sh, same arguments).
# The server is LEFT RUNNING afterwards, so follow-up experiments (run this
# script again) start right away instead of waiting hours for GPUs again.
# Stop it yourself when you are done: scancel <server jobid>  (STOP_SERVER=1
# cancels it automatically after this experiment's results + plots are done).
# Everything waits inside Slurm, so you can log out right after running it.
#
# Usage (arguments after the script name go to wulver_benchmark.sh; default "sweep submit"):
#   bash scripts/wulver_run_with_server.sh                                   # sweep, realtimeqa, 3 questions/config
#   bash scripts/wulver_run_with_server.sh sweep submit gpt-oss-120b-local open_nq 50
#   bash scripts/wulver_run_with_server.sh submit                            # full benchmark
#   RUN_DIR=runs7 bash scripts/wulver_run_with_server.sh sweep submit        # resume runs7
#
# Follow it with:
#   squeue -u $USER                     # gpt-oss-server, RRAG-launcher, RRAG-bench, RRAG-plots ...
#   tail -f log/run-with-server-<launcher jobid>.out
#
# Knobs (env): WAIT_MINUTES (how long the launcher waits for the model to
# load, default 90), STOP_SERVER=1, SERVER_TIME (server --time, default the
# script's 3-00:00:00 = 72 h), plus everything wulver_benchmark.sh /
# serve_gpt_oss_local.slurm accept (GPT_OSS_DIR, VLLM_VENV, MAX_SAMPLES, ...).
# One-time setup first: bash scripts/download_gpt_oss.sh ; sbatch scripts/setup_vllm_env.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source scripts/_load_env.sh

SERVER_FILE="${LOCAL_VLLM_SERVER_FILE:-$HOME/vllm_local_server.txt}"
WAIT_MINUTES="${WAIT_MINUTES:-90}"
hdr() { sed -n "s/^#SBATCH --$1=//p" scripts/wulver_run_one.slurm | head -1; }   # account / partition / qos for CPU jobs

server_up() {   # true if the address file points at a server that answers
  [[ -s "$SERVER_FILE" ]] || return 1
  curl -s --max-time 10 "http://$(tr -d '[:space:]' < "$SERVER_FILE")/v1/models" | grep -q '"id"'
}

# ───────────── phase 2 (inside the launcher job): wait, then submit ─────────────
if [[ "${1:-}" == "--launch-when-ready" ]]; then
  SERVER_JOB="$2"; shift 2
  echo "=== $(date -Is) launcher on $(hostname): waiting up to ${WAIT_MINUTES} min for server job ${SERVER_JOB:-<existing>} ==="
  deadline=$(( $(date +%s) + WAIT_MINUTES * 60 ))
  until server_up; do
    if [[ -n "$SERVER_JOB" ]]; then
      state="$(squeue -h -j "$SERVER_JOB" -o %T 2>/dev/null || true)"
      if [[ -z "$state" || ! "$state" =~ ^(RUNNING|CONFIGURING|PENDING|COMPLETING)$ ]]; then
        echo "server job $SERVER_JOB is no longer running (state: ${state:-gone}) -- see log/vllm-server-${SERVER_JOB}.err" >&2
        exit 1
      fi
    fi
    if (( $(date +%s) > deadline )); then
      echo "model not ready after ${WAIT_MINUTES} min -- see log/vllm-server-${SERVER_JOB}.out/.err; the server keeps running" >&2
      exit 1
    fi
    sleep 30
  done
  echo "=== $(date -Is) server is up at $(cat "$SERVER_FILE"); submitting the experiment ==="
  # submit as if from the login node: don't leak this launcher job's settings into the new jobs
  while read -r v; do unset "$v"; done < <(compgen -e | grep -E '^(SLURM|SBATCH)_' || true)

  out="$(SKIP_SERVER_CHECK=1 bash scripts/wulver_benchmark.sh "$@" 2>&1)" || { echo "$out"; echo "experiment submission failed" >&2; exit 1; }
  echo "$out"
  agg_job="$(sed -n 's/^Submitted aggregate+plot job \([0-9]*\).*/\1/p' <<< "$out")"

  if [[ -n "$SERVER_JOB" && "${STOP_SERVER:-0}" == "1" && -n "$agg_job" ]]; then
    stop_job=$(sbatch --parsable --dependency=afterany:"$agg_job" --job-name=RRAG-stop-server \
      --account="$(hdr account)" --partition="$(hdr partition)" --qos="$(hdr qos)" \
      --ntasks=1 --cpus-per-task=1 --mem=1G --time=00:05:00 \
      --output=log/stop-server-%j.out \
      --wrap "echo stopping server job $SERVER_JOB after aggregate job $agg_job; scancel $SERVER_JOB")
    echo "=== server job $SERVER_JOB will be cancelled when job $agg_job (results + plots) ends (job ${stop_job%%;*}) ==="
  else
    left="$( [[ -n "$SERVER_JOB" ]] && squeue -h -j "$SERVER_JOB" -o %L 2>/dev/null || true)"
    echo "=== server ${SERVER_JOB:+job $SERVER_JOB }keeps running${left:+ ($left left)} for follow-up runs; scancel it when done ==="
  fi
  exit 0
fi

# ───────────── phase 1 (you, on a login node): queue server + launcher ─────────────
ARGS=("$@")
(( ${#ARGS[@]} )) || ARGS=(sweep submit)
# the server only matters for gpt-oss-120b-local (the default model)
for a in "${ARGS[@]}"; do
  case "$a" in gpt-oss-120b|qwen2.5-32b)
    echo "model $a does not use your own server; run bash scripts/wulver_benchmark.sh ${ARGS[*]} directly" >&2; exit 1 ;;
  esac
done
[[ " ${ARGS[*]} " == *" submit "* ]] || { echo "nothing to do: arguments must include 'submit' (try a dry run with bash scripts/wulver_benchmark.sh ${ARGS[*]})" >&2; exit 1; }

mkdir -p log
# broken Python environment? find out now, not after hours of queueing
bash scripts/check_env.sh
# dry run first, so a typo in the arguments fails now and not after an hour in the queue
DRY_ARGS=("${ARGS[@]/submit/dry}")
dry_out="$(SKIP_SERVER_CHECK=1 bash scripts/wulver_benchmark.sh "${DRY_ARGS[@]}" 2>&1)" || { echo "$dry_out" >&2; exit 1; }
head -2 <<< "$dry_out"

# an already queued / loading server job of yours (don't queue a second one)
QUEUED="$(squeue -h -u "${USER:-$(id -un)}" -n gpt-oss-server -t PENDING,CONFIGURING,RUNNING -o '%i %T' 2>/dev/null | head -1 || true)"
if server_up; then
  echo "Your server is already up at $(cat "$SERVER_FILE") -- reusing it."
  SERVER_JOB=""
  DEP=()
elif [[ -n "$QUEUED" ]]; then
  read -r SERVER_JOB state <<< "$QUEUED"
  echo "Server job $SERVER_JOB is already $state (not answering yet) -- waiting for it instead of queueing another."
  DEP=(--dependency=after:"$SERVER_JOB")
else
  TIME_ARGS=(); [[ -n "${SERVER_TIME:-}" ]] && TIME_ARGS=(--time="$SERVER_TIME")
  SERVER_JOB=$(sbatch --parsable "${TIME_ARGS[@]}" scripts/serve_gpt_oss_local.slurm)
  SERVER_JOB="${SERVER_JOB%%;*}"
  echo "Queued vLLM server job $SERVER_JOB for ${SERVER_TIME:-72 h} (log/vllm-server-$SERVER_JOB.out); it stays up for follow-up runs"
  DEP=(--dependency=after:"$SERVER_JOB")   # launcher starts once the server job has started
fi

printf -v ARGS_Q '%q ' "${ARGS[@]}"
LAUNCHER=$(sbatch --parsable "${DEP[@]}" --job-name=RRAG-launcher \
  --account="$(hdr account)" --partition="$(hdr partition)" --qos="$(hdr qos)" \
  --ntasks=1 --cpus-per-task=1 --mem=1G --time=$(( WAIT_MINUTES + 15 )) \
  --output=log/run-with-server-%j.out \
  --export=ALL \
  --wrap "bash scripts/wulver_run_with_server.sh --launch-when-ready '${SERVER_JOB}' ${ARGS_Q}")
LAUNCHER="${LAUNCHER%%;*}"
echo "Queued launcher job $LAUNCHER: once the model answers it runs  wulver_benchmark.sh ${ARGS[*]}"
echo "Follow:  squeue -u \$USER   and   tail -f log/run-with-server-$LAUNCHER.out"
