#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${DIST_DIR:-$ROOT/dist}"
MAX_JOBS="${MAX_JOBS:-16}"
VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-rocm}"
BUILD_BASE="${BUILD_BASE:-}"

cleanup_build_base=0
if [[ -z "$BUILD_BASE" ]]; then
    BUILD_BASE="$(mktemp -d "${TMPDIR:-/tmp}/vllm-cscc-build.XXXXXX")"
    cleanup_build_base=1
elif [[ -e "$BUILD_BASE" ]]; then
    echo "BUILD_BASE must not already exist: $BUILD_BASE" >&2
    exit 2
else
    mkdir -p "$BUILD_BASE"
fi

cleanup() {
    if [[ "$cleanup_build_base" == "1" ]]; then
        rm -rf -- "$BUILD_BASE"
    fi
}
trap cleanup EXIT

cd "$ROOT"
mkdir -p "$DIST_DIR"

export MAX_JOBS
export VLLM_TARGET_DEVICE

python3 setup.py \
    build --build-base "$BUILD_BASE" \
    bdist_wheel --bdist-dir "$BUILD_BASE/bdist" --dist-dir "$DIST_DIR"
sha256sum "$DIST_DIR"/*.whl
