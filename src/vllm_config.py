"""Connection details for the shared gpt-oss-120b vLLM server.

Everything is read from one connection config file (see
CONNECT_TO_SHARED_VLLM.md), re-read on every call so a restart on a new node
only requires editing that file:

    VLLM_HOST=n0005
    VLLM_PORT=8000
    VLLM_MODEL=openai/gpt-oss-120b
    VLLM_API_KEY=dummy

Lookup order for the config path:
  1. $VLLM_CONNECTION_CONFIG
  2. /project/ss797/dk694/vllm_connection.conf   (shared copy on Wulver)
  3. vllm_connection.conf in the repo root        (optional local copy)
If none exists, falls back to the legacy NODE:PORT file in
$SHARED_VLLM_SERVER_FILE (default /project/ss797/ap2645/vllm_server.txt).
"""
import os

DEFAULT_CONFIG_PATH = "/project/ss797/dk694/vllm_connection.conf"
_REPO_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "vllm_connection.conf")
LEGACY_SERVER_FILE = os.environ.get("SHARED_VLLM_SERVER_FILE", "/project/ss797/ap2645/vllm_server.txt")


def config_path():
    env = os.environ.get("VLLM_CONNECTION_CONFIG")
    if env:
        return env
    for p in (DEFAULT_CONFIG_PATH, _REPO_CONFIG_PATH):
        if os.path.exists(p):
            return p
    return None


def _parse(path):
    conf = {}
    with open(path, "r") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" in line:
                k, v = line.split("=", 1)
                conf[k.strip()] = v.strip().strip('"').strip("'")
    return conf


def load_connection():
    """Return {'host','port','model','api_key','base_url','source'}; read fresh each call."""
    path = config_path()
    if path is not None:
        c = _parse(path)
        host, port = c["VLLM_HOST"], c.get("VLLM_PORT", "8000")
        model = c.get("VLLM_MODEL", "openai/gpt-oss-120b")
        api_key = c.get("VLLM_API_KEY", "dummy") or "dummy"
        source = path
    else:  # legacy NODE:PORT file
        with open(LEGACY_SERVER_FILE, "r") as f:
            host, _, port = f.read().strip().partition(":")
        port = port or "8000"
        model, api_key, source = "openai/gpt-oss-120b", "dummy", LEGACY_SERVER_FILE
    return {
        "host": host, "port": port, "model": model, "api_key": api_key,
        "base_url": f"http://{host}:{port}/v1", "source": source,
    }


def base_url():
    return load_connection()["base_url"]


def make_client():
    from openai import OpenAI
    c = load_connection()
    return OpenAI(base_url=c["base_url"], api_key=c["api_key"])
