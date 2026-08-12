#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${1:-$ROOT/../Qwen3.5-27B}}"
PORT="${PORT:-8001}"
VLLM_BIN="${VLLM_BIN:-vllm}"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1}"
VLLM_RUN_DIR="${VLLM_RUN_DIR:-${TMPDIR:-/tmp}}"

if [[ "$#" -gt 1 ]]; then
    echo "usage: MODEL_DIR=/path/to/Qwen3.5-27B $0" >&2
    echo "   or: $0 [/path/to/Qwen3.5-27B]" >&2
    exit 2
fi
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    echo "error: model config not found: $MODEL_DIR/config.json" >&2
    exit 2
fi
if ! [[ "$PORT" =~ ^[1-9][0-9]*$ ]] || ((PORT > 65535)); then
    echo "error: PORT must be in 1..65535" >&2
    exit 2
fi
if ! command -v "$VLLM_BIN" >/dev/null 2>&1; then
    echo "error: vllm executable not found: $VLLM_BIN" >&2
    exit 2
fi
if [[ ! -d "$VLLM_RUN_DIR" ]]; then
    echo "error: VLLM_RUN_DIR not found: $VLLM_RUN_DIR" >&2
    exit 2
fi

MODEL_DIR="$(realpath "$MODEL_DIR")"
VLLM_BIN="$(command -v "$VLLM_BIN")"
VLLM_RUN_DIR="$(realpath "$VLLM_RUN_DIR")"
case "$VLLM_RUN_DIR" in
    "$ROOT" | "$ROOT"/*)
        echo "error: VLLM_RUN_DIR must be outside the source checkout" >&2
        exit 2
        ;;
esac

IFS=',' read -r -a visible_devices <<<"$HIP_VISIBLE_DEVICES"
if [[ "${#visible_devices[@]}" -ne 2 ||
      -z "${visible_devices[0]}" ||
      -z "${visible_devices[1]}" ||
      "${visible_devices[0]}" == "${visible_devices[1]}" ]]; then
    echo "error: HIP_VISIBLE_DEVICES must name exactly two distinct devices" >&2
    exit 2
fi

# Keep the frozen M=4096 profile available for matching TP2 GEMMs. Shapes not
# present in the profile continue through PyTorch TunableOp's normal fallback.
source "$ROOT/scripts/cscc_gfx936_env.sh"
export HIP_VISIBLE_DEVICES
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"

echo "starting TP=2, DP=1 on HIP devices $HIP_VISIBLE_DEVICES"
echo "model=$MODEL_DIR port=$PORT evaluation_concurrency=10"
echo "runtime_dir=$VLLM_RUN_DIR executable=$VLLM_BIN"

# Running a console script from the source checkout places that checkout at the
# front of Python's import path. Use a neutral directory so the verified wheel,
# including its native extensions, is the package that actually serves traffic.
cd "$VLLM_RUN_DIR"

exec "$VLLM_BIN" serve "$MODEL_DIR" \
    --served-model-name Qwen3.5-27B \
    --port "$PORT" \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-num-seqs 128 \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.95 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
