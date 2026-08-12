#!/usr/bin/env bash
set -euo pipefail

ROOT="/public/home/tangyu408/Qwen_DCU_Worker_0/modular_validation/results/gqa_page784_k5120_final_dp2_5355cea_20260811/throughput"
DATA_DIR="$ROOT/data"
RESULT_DIR="$ROOT/results"
MODEL_DIR="/root/Qwen3.5-27B"
VLLM_SITE="/tmp/qwen35-dp2-full-5355cea/site"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
export PYTHONPATH="$VLLM_SITE"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RESULT_DIR"

for concurrency in 2 4 8; do
    for bucket in 4-8K 8-16K 16-32K; do
        case_name="${bucket}_c${concurrency}"
        case_dir="$RESULT_DIR/$case_name"
        mkdir -p "$case_dir"

        echo "===== $case_name: 50 prompts, request_rate=inf, warmups=2 ====="
        python3 -m vllm.entrypoints.cli.main bench serve \
            --backend openai-chat \
            --host 127.0.0.1 \
            --port 8001 \
            --endpoint /v1/chat/completions \
            --model Qwen3.5-27B \
            --tokenizer "$MODEL_DIR" \
            --dataset-name custom \
            --dataset-path "$DATA_DIR/${bucket}_throughput.jsonl" \
            --num-prompts 50 \
            --no-oversample \
            --max-concurrency "$concurrency" \
            --request-rate inf \
            --temperature 0 \
            --disable-shuffle \
            --custom-output-len 1024 \
            --num-warmups 2 \
            --save-detailed \
            --extra-body '{"temperature":0.0}' \
            --percentile-metrics ttft,tpot,itl,e2el \
            --metric-percentiles 50,95,99 \
            --save-result \
            --result-dir "$case_dir" \
            --result-filename result.json \
            2>&1 | tee "$case_dir/client.log"
    done
done
