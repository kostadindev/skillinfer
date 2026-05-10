#!/usr/bin/env bash
# End-to-end test: install both packages from PyPI into a fresh venv
# and exercise them. Run from the repo root or from this directory.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${E2E_VENV:-$HERE/.venv}"
PY="${PYTHON:-python3}"

echo "[e2e] creating venv at $VENV"
rm -rf "$VENV"
"$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install --quiet --upgrade pip

echo "[e2e] installing skillinfer + skillinfer-mcp from PyPI"
# --no-cache-dir ensures we hit PyPI, not a local wheel cache.
pip install --no-cache-dir --quiet skillinfer skillinfer-mcp

echo "[e2e] versions:"
pip show skillinfer skillinfer-mcp | grep -E "^(Name|Version):"

echo "[e2e] running skillinfer test"
python "$HERE/test_skillinfer_pypi.py"

echo "[e2e] running skillinfer-mcp test"
python "$HERE/test_mcp_pypi.py"

echo "[e2e] all green"
