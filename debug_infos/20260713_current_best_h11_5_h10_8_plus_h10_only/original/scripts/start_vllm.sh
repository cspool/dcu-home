#!/usr/bin/env bash
set -u
set -o pipefail

MODEL_DIR="${MODEL_DIR:-../Qwen3.5-27B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-27B}"
VLLM_PORT="${VLLM_PORT:-8001}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"

vllm serve "$MODEL_DIR" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --port "$VLLM_PORT" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
