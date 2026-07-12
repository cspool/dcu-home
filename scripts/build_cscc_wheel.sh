#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${DIST_DIR:-$ROOT/dist}"
MAX_JOBS="${MAX_JOBS:-16}"
VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-rocm}"

cd "$ROOT"
mkdir -p "$DIST_DIR"

export MAX_JOBS
export VLLM_TARGET_DEVICE

python3 setup.py build_py --force
python3 setup.py bdist_wheel --dist-dir "$DIST_DIR"
sha256sum "$DIST_DIR"/*.whl
