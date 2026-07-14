#!/usr/bin/env bash
set -u
set -o pipefail

MODEL_DIR="${MODEL_DIR:-../Qwen3.5-27B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-27B}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8001}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
REQUEST_RATE="${REQUEST_RATE:-1}"
CUSTOM_OUTPUT_LEN="${CUSTOM_OUTPUT_LEN:-1024}"
NUM_WARMUPS="${NUM_WARMUPS:-2}"
RESULT_ROOT="${RESULT_ROOT:-./test}"

DATASET="${1:-all}"
NUM_PROMPTS="${2:-}"

run_one() {
    local label="$1"
    local dataset_path="$2"
    local result_dir="$3"
    local n_prompts="${NUM_PROMPTS:-$(wc -l < "$dataset_path")}"

    mkdir -p "$result_dir"
    echo "===== throughput: $label ====="

    vllm bench serve \
        --backend openai-chat \
        --host "$VLLM_HOST" \
        --port "$VLLM_PORT" \
        --endpoint /v1/chat/completions \
        --model "$SERVED_MODEL_NAME" \
        --tokenizer "$MODEL_DIR" \
        --dataset-name custom \
        --dataset-path "$dataset_path" \
        --num-prompts "$n_prompts" \
        --no-oversample \
        --max-concurrency "$MAX_CONCURRENCY" \
        --request-rate "$REQUEST_RATE" \
        --temperature 0 \
        --disable-shuffle \
        --custom-output-len "$CUSTOM_OUTPUT_LEN" \
        --num-warmups "$NUM_WARMUPS" \
        --save-detailed \
        --extra-body '{"temperature":0.0}' \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,95,99 \
        --save-result \
        --result-dir "$result_dir" \
        --result-filename result.json
}

case "$DATASET" in
    all)
        run_one "4-8K" "./4-8K_throughput.jsonl" "$RESULT_ROOT/4-8K_throughput"
        run_one "8-16K" "./8-16K_throughput.jsonl" "$RESULT_ROOT/8-16K_throughput"
        run_one "16-32K" "./16-32K_throughput.jsonl" "$RESULT_ROOT/16-32K_throughput"
        ;;
    4-8K)
        run_one "4-8K" "./4-8K_throughput.jsonl" "$RESULT_ROOT/4-8K_throughput"
        ;;
    8-16K)
        run_one "8-16K" "./8-16K_throughput.jsonl" "$RESULT_ROOT/8-16K_throughput"
        ;;
    16-32K)
        run_one "16-32K" "./16-32K_throughput.jsonl" "$RESULT_ROOT/16-32K_throughput"
        ;;
    *)
        echo "usage: $0 [all|4-8K|8-16K|16-32K] [num_prompts]"
        exit 1
        ;;
esac

echo
echo "===== result files ====="
find "$RESULT_ROOT" -name result.json -type f -print
