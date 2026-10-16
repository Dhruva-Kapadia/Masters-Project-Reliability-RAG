#!/bin/bash
#SBATCH --job-name=gpt-oss-download
#SBATCH --output=log/download-gpt-oss-%j.out
#SBATCH --account=ss797
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#
# One-time download of the gpt-oss-120b weights into your own directory on
# Wulver, for scripts/serve_gpt_oss_local.slurm to serve (~65 GB).
#
# Option A -- as a Slurm job (recommended: no login-node memory/CPU killer):
#   mkdir -p log && sbatch scripts/download_gpt_oss.sh
#   tail -f log/download-gpt-oss-<jobid>.out
#   (if the job log shows connection errors right away, compute nodes have no
#    internet here -- use option B)
# Option B -- on a login node, detached so it survives logout:
#   nohup bash scripts/download_gpt_oss.sh > log/download-gpt-oss.out 2>&1 &
#   tail -f log/download-gpt-oss.out
#
# Other directory: bash scripts/download_gpt_oss.sh /path/to/dir  (or GPT_OSS_DIR in .env)
#
# Robust against stalls: downloads use plain HTTPS (HF_HUB_DISABLE_XET=1)
# instead of the Xet backend, whose many parallel streams stall behind the
# cluster network and use enough memory to get killed on a login node; a
# stalled connection times out after DOWNLOAD_TIMEOUT s (60) and the whole
# download is retried up to RETRIES (30) times, each attempt resuming partial
# files where they stopped. MAX_WORKERS (2) files are fetched in parallel.
# Only the Hugging Face-format weights are fetched: original/ (a second full
# copy of the checkpoint) and metal/ are skipped. HF_TOKEN from .env is used if set.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source scripts/_load_env.sh

MODEL_ID="${MODEL_ID:-openai/gpt-oss-120b}"
MODEL_DIR="${1:-${GPT_OSS_DIR:-/project/ss797/dk694/models/gpt-oss-120b}}"
RETRIES="${RETRIES:-30}"
export MAX_WORKERS="${MAX_WORKERS:-2}"

export HF_HOME="${HF_HOME:-/project/ss797/dk694/hf_cache}"   # keep HF's cache out of $HOME
export HF_HUB_DISABLE_XET=1                                  # plain, resumable HTTPS downloads
export HF_HUB_DOWNLOAD_TIMEOUT="${DOWNLOAD_TIMEOUT:-60}"     # a stalled connection fails instead of hanging
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"  # no per-file bars; a size line every minute instead

mkdir -p "$MODEL_DIR"
echo "=== $(date -Is) downloading $MODEL_ID -> $MODEL_DIR on $(hostname) ==="
df -h "$MODEL_DIR" | tail -1 | awk '{print "free space on that filesystem: " $4 " (needs ~65 GB)"}'

# progress: print the downloaded size once a minute
( while sleep 60; do echo "  $(date +%T)  $(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1) downloaded"; done ) &
TICKER=$!
disown "$TICKER"   # no "Terminated" notice when it is stopped at the end
trap 'pkill -P $TICKER 2>/dev/null; kill $TICKER 2>/dev/null || true' EXIT

for attempt in $(seq 1 "$RETRIES"); do
  echo "--- attempt $attempt/$RETRIES ($(date +%T)), $(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1) so far ---"
  status=0
  uv run python - "$MODEL_ID" "$MODEL_DIR" <<'PY' || status=$?
import os, sys, time
from huggingface_hub import snapshot_download
repo, dst = sys.argv[1], sys.argv[2]
t = time.time()
snapshot_download(
    repo_id=repo,
    local_dir=dst,
    ignore_patterns=["original/*", "metal/*"],
    token=os.environ.get("HF_TOKEN") or None,
    max_workers=int(os.environ["MAX_WORKERS"]),
)
print(f"OK: all files of {repo} are in {dst} ({time.time() - t:.0f}s this attempt)")
PY
  if (( status == 0 )); then break; fi
  if (( attempt == RETRIES )); then
    echo "giving up after $RETRIES attempts (last exit $status). Re-run the same command to continue from here." >&2
    exit 1
  fi
  if (( status == 137 )); then echo "(exit 137 = the process was killed, e.g. by the login-node memory limit -- prefer sbatch)" >&2; fi
  echo "attempt $attempt failed (exit $status); retrying in 20s -- partial files are resumed" >&2
  sleep 20
done

# sanity check: every shard listed in the index is present and non-empty
uv run python - "$MODEL_DIR" <<'PY'
import json, os, sys
d = sys.argv[1]
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
shards = sorted(set(idx["weight_map"].values()))
missing = [s for s in shards if not os.path.isfile(os.path.join(d, s)) or os.path.getsize(os.path.join(d, s)) == 0]
assert os.path.isfile(os.path.join(d, "config.json")), "config.json missing"
if missing:
    sys.exit(f"incomplete: missing {missing} -- re-run the download")
print(f"verified: config.json + {len(shards)} weight shards present")
PY
du -sh "$MODEL_DIR"
echo "=== $(date -Is) done. Start the server with: mkdir -p log && sbatch scripts/serve_gpt_oss_local.slurm ==="
echo "(non-default directory? GPT_OSS_DIR=$MODEL_DIR sbatch scripts/serve_gpt_oss_local.slurm, or put GPT_OSS_DIR=... in .env)"
