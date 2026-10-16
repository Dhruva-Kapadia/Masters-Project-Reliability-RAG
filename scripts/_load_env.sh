# Source from a bash script (after cd to the repo root) to load API keys and
# paths from .env (git-ignored; see .env.example). Variables already set in the
# shell win; empty values are skipped. Handles quotes, CRLF and "export ".
if [[ -f .env ]]; then
  while IFS='=' read -r _k _v || [[ -n "$_k" ]]; do
    _k="${_k%$'\r'}"; _v="${_v%$'\r'}"
    _k="${_k#export }"; _k="${_k// /}"
    [[ "$_k" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    _v="${_v#"${_v%%[![:space:]]*}"}"; _v="${_v%"${_v##*[![:space:]]}"}"
    _v="${_v#\"}"; _v="${_v%\"}"; _v="${_v#\'}"; _v="${_v%\'}"
    if [[ -n "$_v" && -z "${!_k:-}" ]]; then export "$_k=$_v"; fi
  done < .env
  unset _k _v
fi
