#!/bin/bash
# Check that the repo's Python environment (.venv, managed by uv) can import
# what main.py needs -- run before queueing jobs, so a broken install fails in
# seconds here instead of in every job after hours in the queue. Syncs the
# environment to uv.lock first (the jobs themselves never sync: --no-sync).
#   bash scripts/check_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Checking the Python environment (uv sync + imports) ..."
uv sync --frozen --quiet || { echo "uv sync failed -- fix the environment before submitting" >&2; exit 1; }
if ! env_err="$(uv run --no-sync python -c '
import openai, sentencepiece, nltk, numpy, pandas
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForSequenceClassification
print("ok")' 2>&1)"; then
  echo "The Python environment is broken -- every job would fail at import:" >&2
  echo "$env_err" | tail -3 >&2
  echo "Repair it on a login node:  uv sync --reinstall-package sentencepiece   (or rebuild: rm -rf .venv && uv sync)" >&2
  exit 1
fi
echo "Python environment OK"
