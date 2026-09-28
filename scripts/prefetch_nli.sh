#!/bin/bash
# One-time download of the NLI model used by the graph / MIS / sampleMIS
# defenses into the shared Hugging Face cache that
# scripts/wulver_run_one.slurm points HF_HOME at. Run on a Wulver LOGIN node
# (they have internet; compute nodes may not):
#
#   bash scripts/prefetch_nli.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-/project/ss797/dk694/hf_cache}"
mkdir -p "$HF_HOME"
echo "Downloading NLI model into HF_HOME=$HF_HOME ..."
uv run python - <<'PY'
from src.nli_config import NLI_MODEL, load_nli
tok, model, contra = load_nli(device="cpu")
print(f"OK: {NLI_MODEL} loaded; contradiction label index = {contra}")
PY
