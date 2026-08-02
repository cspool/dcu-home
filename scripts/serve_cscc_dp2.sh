#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${1:-}}"
PORT="${PORT:-8001}"
VLLM_BIN="${VLLM_BIN:-vllm}"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1}"

if [[ "$#" -gt 1 || -z "$MODEL_DIR" ]]; then
    echo "usage: MODEL_DIR=/path/to/Qwen3.5-27B $0" >&2
    echo "   or: $0 /path/to/Qwen3.5-27B" >&2
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

IFS=',' read -r -a visible_devices <<<"$HIP_VISIBLE_DEVICES"
if [[ "${#visible_devices[@]}" -ne 2 ||
      -z "${visible_devices[0]}" ||
      -z "${visible_devices[1]}" ||
      "${visible_devices[0]}" == "${visible_devices[1]}" ]]; then
    echo "error: HIP_VISIBLE_DEVICES must name exactly two distinct devices" >&2
    exit 2
fi

source "$ROOT/scripts/cscc_gfx936_env.sh"
export HIP_VISIBLE_DEVICES
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"

echo "starting internal DP=2, TP=1 on HIP devices $HIP_VISIBLE_DEVICES"
echo "model=$MODEL_DIR port=$PORT backend=mp"

exec "$VLLM_BIN" serve "$MODEL_DIR" \
    --served-model-name Qwen3.5-27B \
    --port "$PORT" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --data-parallel-size 2 \
    --data-parallel-backend mp \
    --max-num-seqs 128 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.95 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
