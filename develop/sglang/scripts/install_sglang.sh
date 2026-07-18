#!/usr/bin/env bash
# Create venv under develop/sglang/.venv and install sglang + transfer engines.
# Requires network. Does NOT touch system python.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SGLANG_DEV_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
VENV_DIR="$SGLANG_DEV_ROOT/.venv"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

echo "[install] creating venv at $VENV_DIR"
uv venv --python 3.12 "$VENV_DIR"
PY="$VENV_DIR/bin/python"

echo "[install] sglang[all] + nixl + mooncake-transfer-engine (network required)"
uv pip install --python "$PY" "sglang[all]"
uv pip install --python "$PY" nixl mooncake-transfer-engine

echo "[install] versions:"
"$PY" -m sglang.launch_server --version || true
"$PY" -c "import nixl; print('nixl ok', nixl.__file__)" || echo "nixl import failed (may need source build with ucx_path)"
"$PY" -c "import mooncake; print('mooncake ok')" || echo "mooncake import failed"

echo "[install] done. activate: source $VENV_DIR/bin/activate"
