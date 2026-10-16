#!/bin/bash
# One-time download of the gpt-oss-120b weights into your own directory on
# Wulver, for scripts/serve_gpt_oss_local.slurm to serve. Run it on a LOGIN
# node (compute nodes may have no internet). It takes a while (~65 GB), so run
# it inside tmux/screen, or with nohup:
#
#   bash scripts/download_gpt_oss.sh                        # -> $GPT_OSS_DIR (default below)
#   bash scripts/download_gpt_oss.sh /project/ss797/dk694/models/gpt-oss-120b
#   nohup bash scripts/download_gpt_oss.sh > log/download_gpt_oss.out 2>&1 &
#
# Re-running resumes / verifies (already-complete files are skipped). Only the
# Hugging Face-format weights are fetched: the repo's original/ (a second full
# copy of the checkpoint) and metal/ folders are skipped, which halves the size.
# HF_TOKEN from .env is used if set (the repo is public; a token avoids
# anonymous rate limits).

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_load_env.sh

MODEL_ID="${MODEL_ID:-openai/gpt-oss-120b}"
MODEL_DIR="${1:-${GPT_OSS_DIR:-/project/ss797/dk694/models/gpt-oss-120b}}"

mkdir -p "$MODEL_DIR"
echo "=== downloading $MODEL_ID -> $MODEL_DIR ==="
df -h "$MODEL_DIR" | tail -1 | awk '{print "free space on that filesystem: " $4}'
echo "(needs ~65 GB; check your /project quota with: quota -s  or  df -h /project/ss797)"

# keep HF's own cache out of $HOME (small quota): metadata goes next to the weights
export HF_HOME="${HF_HOME:-/project/ss797/dk694/hf_cache}"

uv run python - "$MODEL_ID" "$MODEL_DIR" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
repo, dst = sys.argv[1], sys.argv[2]
path = snapshot_download(
    repo_id=repo,
    local_dir=dst,
    ignore_patterns=["original/*", "metal/*"],
    token=os.environ.get("HF_TOKEN") or None,
    max_workers=8,
)
print(f"OK: {repo} is in {path}")
PY

[[ -f "$MODEL_DIR/config.json" ]] || { echo "download incomplete: $MODEL_DIR/config.json missing" >&2; exit 1; }
du -sh "$MODEL_DIR"
echo "=== done. Start the server with: sbatch scripts/serve_gpt_oss_local.slurm ==="
echo "(if you used a non-default directory: GPT_OSS_DIR=$MODEL_DIR sbatch scripts/serve_gpt_oss_local.slurm, or put GPT_OSS_DIR=... in .env)"
