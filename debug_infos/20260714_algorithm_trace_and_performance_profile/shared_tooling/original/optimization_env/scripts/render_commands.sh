#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TESTDATA_DIR="$(cd "$ENV_DIR/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ENV_DIR/.env}"
MATRIX="${MATRIX:-$ENV_DIR/benchmark_matrix.tsv}"
CASE_FILTER="${1:-all}"

if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
fi

MODEL_DIR="${MODEL_DIR:-../Qwen3.5-27B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-27B}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8001}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
REQUEST_RATE="${REQUEST_RATE:-1}"
CUSTOM_OUTPUT_LEN="${CUSTOM_OUTPUT_LEN:-1024}"
NUM_WARMUPS="${NUM_WARMUPS:-2}"
NUM_PROMPTS="${NUM_PROMPTS:-}"

echo "# Dry-run only. These commands are printed, not executed."
echo "# Start each server command in one terminal, then run the benchmark command in another terminal."
echo

tail -n +2 "$MATRIX" | while IFS=$'\t' read -r case_id context max_num_batched_tokens max_num_seqs gpu_memory_utilization purpose; do
    if [ -z "${case_id:-}" ]; then
        continue
    fi
    if [ "$CASE_FILTER" != "all" ] && [ "$CASE_FILTER" != "$case_id" ]; then
        continue
    fi

    result_root="./optimization_env/results/$case_id"
    bench_args="\"$context\""
    if [ -n "$NUM_PROMPTS" ]; then
        bench_args="$bench_args \"$NUM_PROMPTS\""
    fi
    cat <<EOF
# case: $case_id
# purpose: $purpose
cd "$TESTDATA_DIR"
MODEL_DIR="$MODEL_DIR" SERVED_MODEL_NAME="$SERVED_MODEL_NAME" VLLM_PORT="$VLLM_PORT" MAX_NUM_SEQS="$max_num_seqs" MAX_NUM_BATCHED_TOKENS="$max_num_batched_tokens" GPU_MEMORY_UTILIZATION="$gpu_memory_utilization" ./start_vllm.sh

cd "$TESTDATA_DIR"
MODEL_DIR="$MODEL_DIR" SERVED_MODEL_NAME="$SERVED_MODEL_NAME" VLLM_HOST="$VLLM_HOST" VLLM_PORT="$VLLM_PORT" RESULT_ROOT="$result_root" MAX_CONCURRENCY="$MAX_CONCURRENCY" REQUEST_RATE="$REQUEST_RATE" CUSTOM_OUTPUT_LEN="$CUSTOM_OUTPUT_LEN" NUM_WARMUPS="$NUM_WARMUPS" ./run_throughput.sh $bench_args

EOF
done
