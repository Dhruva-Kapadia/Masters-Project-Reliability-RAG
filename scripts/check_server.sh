#!/bin/bash
# Smoke-test a running vLLM server with one dummy chat query: prints the
# answer, the start of the reasoning trace, token usage and latency, and exits
# non-zero if the server does not produce a real answer.
#
#   bash scripts/check_server.sh                      # your own server (address in ~/vllm_local_server.txt)
#   bash scripts/check_server.sh n0024:41234          # a given host:port
#   SERVER_FILE=/path/to/addr.txt bash scripts/check_server.sh
#
# Runs fine on a login node (one small HTTP request). wulver_benchmark.sh calls
# it before submitting (SKIP_SERVER_CHECK=1 skips).

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_load_env.sh

if [[ -n "${1:-}" ]]; then
  ADDR="$1"
else
  SERVER_FILE="${SERVER_FILE:-${LOCAL_VLLM_SERVER_FILE:-$HOME/vllm_local_server.txt}}"
  [[ -s "$SERVER_FILE" ]] || { echo "no server address: $SERVER_FILE is missing (server not running or still loading)" >&2; exit 1; }
  ADDR="$(tr -d '[:space:]' < "$SERVER_FILE")"
fi
MODEL="${MODEL:-openai/gpt-oss-120b}"
QUESTION="${QUESTION:-What is the capital of France? Answer with one word.}"
EXPECT="${EXPECT:-Paris}"

echo "server : http://$ADDR/v1   model: $MODEL"
echo "query  : $QUESTION"
body=$(printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":2048,"temperature":0}' "$MODEL" "$QUESTION")
start=$(date +%s.%N)
resp="$(curl -s --max-time "${TIMEOUT:-180}" -H 'Content-Type: application/json' -d "$body" "http://$ADDR/v1/chat/completions")" \
  || { echo "FAILED: no response from http://$ADDR within ${TIMEOUT:-180}s" >&2; exit 1; }
secs=$(awk -v s="$start" -v e="$(date +%s.%N)" 'BEGIN{printf "%.1f", e-s}')

PY=python3; command -v python3 >/dev/null || PY="uv run python"
$PY - "$resp" "$secs" "$EXPECT" <<'PY'
import json, sys
raw, secs, expect = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    r = json.loads(raw)
except ValueError:
    sys.exit(f"FAILED: not a JSON reply: {raw[:300]}")
if "error" in r or not r.get("choices"):
    sys.exit(f"FAILED: server error: {json.dumps(r.get('error', r))[:500]}")
m = r["choices"][0]["message"]
answer = (m.get("content") or "").strip()
reasoning = m.get("reasoning") or m.get("reasoning_content") or ""
u = r.get("usage") or {}
print(f"answer : {answer!r}")
print(f"think  : {reasoning[:200]!r}{' ...' if len(reasoning) > 200 else ''}")
print(f"tokens : prompt={u.get('prompt_tokens')} completion={u.get('completion_tokens')}"
      f"  finish={r['choices'][0].get('finish_reason')}  latency={secs}s")
if not answer:
    sys.exit("FAILED: empty answer (reasoning used the whole token budget?)")
if not reasoning:
    print("note   : no reasoning field returned -- thinking traces in the logs will be empty")
print("OK: server answers" + ("" if expect.lower() in answer.lower() else f" (but did not say {expect!r})"))
PY
