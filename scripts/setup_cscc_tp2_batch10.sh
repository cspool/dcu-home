#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DIST_DIR="${DIST_DIR:-$ROOT/dist-cscc-tp2-batch10}"
MAX_JOBS="${MAX_JOBS:-16}"
WHEEL="${WHEEL:-}"
OFFICIAL_TEST_DIR="${OFFICIAL_TEST_DIR:-}"

fail() {
    echo "setup_cscc_tp2_batch10: ERROR: $*" >&2
    exit 1
}

command -v "$PYTHON_BIN" >/dev/null 2>&1 || \
    fail "Python executable not found: $PYTHON_BIN"

if [[ -z "$WHEEL" ]]; then
    echo "[1/4] Building a clean TP2 batch10 wheel"
    DIST_DIR="$DIST_DIR" MAX_JOBS="$MAX_JOBS" \
        bash "$ROOT/scripts/build_cscc_wheel.sh"

    newest_wheel=""
    while IFS= read -r -d '' candidate; do
        if [[ -z "$newest_wheel" || "$candidate" -nt "$newest_wheel" ]]; then
            newest_wheel="$candidate"
        fi
    done < <(find "$DIST_DIR" -maxdepth 1 -type f -name 'vllm-*.whl' \
        -print0)
    [[ -n "$newest_wheel" ]] || fail "no wheel produced under $DIST_DIR"
    WHEEL="$newest_wheel"
else
    echo "[1/4] Reusing the supplied wheel"
fi

[[ -f "$WHEEL" ]] || fail "wheel not found: $WHEEL"
WHEEL="$(realpath "$WHEEL")"

echo "[2/4] Verifying source and wheel contents"
bash "$ROOT/scripts/verify_cscc_repro.sh" "$WHEEL"

echo "[3/4] Installing without resolving any additional dependency"
"$PYTHON_BIN" -m pip install --no-deps --force-reinstall "$WHEEL"

echo "[4/4] Checking the installed package and native extensions"
"$PYTHON_BIN" -I - "$ROOT" <<'PY'
import hashlib
import importlib
from pathlib import Path
import sys

source_root = Path(sys.argv[1]).resolve()

vllm = importlib.import_module("vllm")
importlib.import_module("vllm._C")
importlib.import_module("vllm._rocm_C")

installed_root = Path(vllm.__file__).resolve().parent
source_scheduler = source_root / "vllm/v1/core/sched/scheduler.py"
installed_scheduler = installed_root / "v1/core/sched/scheduler.py"

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

if source_root == installed_root or source_root in installed_root.parents:
    raise SystemExit(f"import resolved to source checkout, not installation: {installed_root}")
if not installed_scheduler.is_file():
    raise SystemExit(f"installed scheduler missing: {installed_scheduler}")
if sha256(source_scheduler) != sha256(installed_scheduler):
    raise SystemExit("installed scheduler differs from submitted source")

print(f"installed vLLM version: {vllm.__version__}")
print(f"installed package: {installed_root}")
print(f"scheduler SHA-256: {sha256(installed_scheduler)}")
print("native imports: vllm._C=OK, vllm._rocm_C=OK")
PY

if [[ -n "$OFFICIAL_TEST_DIR" ]]; then
    [[ -d "$OFFICIAL_TEST_DIR" ]] || \
        fail "OFFICIAL_TEST_DIR not found: $OFFICIAL_TEST_DIR"
    for script in run_throughput.sh run_accuracy.sh; do
        path="$OFFICIAL_TEST_DIR/$script"
        [[ -f "$path" ]] || fail "official entry point missing: $path"
        bash -n "$path" || fail "official entry point has invalid shell syntax: $path"
    done

    echo "Official entry points were only checked, not modified or executed:"
    sha256sum \
        "$OFFICIAL_TEST_DIR/run_throughput.sh" \
        "$OFFICIAL_TEST_DIR/run_accuracy.sh"
fi

wheel_sha="$(sha256sum "$WHEEL" | awk '{print $1}')"
echo
echo "SETUP PASS"
echo "wheel: $WHEEL"
echo "wheel SHA-256: $wheel_sha"
echo "next: MODEL_DIR=/path/to/Qwen3.5-27B bash $ROOT/scripts/serve_cscc_tp2_batch10.sh"
echo "or use scripts/run_cscc_tp2_batch10_e2e.sh for readiness checking and unchanged official entry points"
