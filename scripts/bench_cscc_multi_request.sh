#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-}"
DATA_DIR="${DATA_DIR:-}"
RESULT_ROOT="${RESULT_ROOT:-}"
RUN_LABEL="${RUN_LABEL:-dp2}"
PORT="${PORT:-8001}"
VLLM_BIN="${VLLM_BIN:-vllm}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
NUM_WARMUPS="${NUM_WARMUPS:-2}"
IGNORE_EOS="${IGNORE_EOS:-1}"
read -r -a datasets <<<"${DATASETS:-4-8K 8-16K 16-32K}"
read -r -a concurrencies <<<"${CONCURRENCIES:-2 4 8}"

required_path() {
    local name="$1"
    local value="$2"
    if [[ -z "$value" ]]; then
        echo "error: set $name" >&2
        exit 2
    fi
}

positive_integer() {
    local name="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: $name must be a positive integer" >&2
        exit 2
    fi
}

required_path MODEL_DIR "$MODEL_DIR"
required_path DATA_DIR "$DATA_DIR"
required_path RESULT_ROOT "$RESULT_ROOT"
[[ -f "$MODEL_DIR/config.json" ]] || {
    echo "error: missing model config: $MODEL_DIR/config.json" >&2
    exit 2
}
[[ -d "$DATA_DIR" ]] || {
    echo "error: DATA_DIR is not a directory: $DATA_DIR" >&2
    exit 2
}
if ! command -v "$VLLM_BIN" >/dev/null 2>&1; then
    echo "error: vllm executable not found: $VLLM_BIN" >&2
    exit 2
fi
if [[ ! "$RUN_LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "error: RUN_LABEL contains unsupported characters" >&2
    exit 2
fi
positive_integer NUM_PROMPTS "$NUM_PROMPTS"
positive_integer OUTPUT_LEN "$OUTPUT_LEN"
positive_integer NUM_WARMUPS "$NUM_WARMUPS"
if [[ "$IGNORE_EOS" != 0 && "$IGNORE_EOS" != 1 ]]; then
    echo "error: IGNORE_EOS must be 0 or 1" >&2
    exit 2
fi
if [[ "$REQUEST_RATE" != "inf" &&
      ! "$REQUEST_RATE" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
    echo "error: REQUEST_RATE must be a positive number or inf" >&2
    exit 2
fi
if [[ "$REQUEST_RATE" != "inf" &&
      "$REQUEST_RATE" =~ ^0*([.]0*)?$ ]]; then
    echo "error: REQUEST_RATE must be greater than zero" >&2
    exit 2
fi
for concurrency in "${concurrencies[@]}"; do
    positive_integer CONCURRENCIES "$concurrency"
done
for dataset in "${datasets[@]}"; do
    case "$dataset" in
        4-8K|8-16K|16-32K) ;;
        *)
            echo "error: unsupported dataset: $dataset" >&2
            exit 2
            ;;
    esac
    [[ -f "$DATA_DIR/${dataset}_throughput.jsonl" ]] || {
        echo "error: missing dataset: $DATA_DIR/${dataset}_throughput.jsonl" >&2
        exit 2
    }
done
if ! curl --noproxy '*' -fsS \
    "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo "error: service is not ready on port $PORT" >&2
    exit 2
fi

run_root="$RESULT_ROOT/$RUN_LABEL"
mkdir -p "$run_root"
metadata="$run_root/run-metadata.txt"
if [[ ! -e "$metadata" ]]; then
    {
        echo "source_commit=$(git -C "$ROOT" rev-parse HEAD)"
        echo "model_dir=$MODEL_DIR"
        echo "data_dir=$DATA_DIR"
        echo "port=$PORT"
        echo "num_prompts=$NUM_PROMPTS"
        echo "output_len=$OUTPUT_LEN"
        echo "request_rate=$REQUEST_RATE"
        echo "num_warmups=$NUM_WARMUPS"
        echo "ignore_eos=$IGNORE_EOS"
        echo "datasets=${datasets[*]}"
        echo "concurrencies=${concurrencies[*]}"
        echo "vllm_bin=$(command -v "$VLLM_BIN")"
    } >"$metadata"
fi

export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy

ignore_eos_args=()
if [[ "$IGNORE_EOS" == 1 ]]; then
    ignore_eos_args+=(--ignore-eos)
fi

for concurrency in "${concurrencies[@]}"; do
    for dataset in "${datasets[@]}"; do
        dataset_path="$DATA_DIR/${dataset}_throughput.jsonl"
        result_dir="$run_root/c${concurrency}-r${REQUEST_RATE}/$dataset"
        if [[ -e "$result_dir/result.json" ]]; then
            echo "error: refusing to overwrite $result_dir/result.json" >&2
            exit 2
        fi
        mkdir -p "$result_dir"
        echo "===== $RUN_LABEL $dataset concurrency=$concurrency ====="
        "$VLLM_BIN" bench serve \
            --backend openai-chat \
            --host 127.0.0.1 \
            --port "$PORT" \
            --endpoint /v1/chat/completions \
            --model Qwen3.5-27B \
            --tokenizer "$MODEL_DIR" \
            --dataset-name custom \
            --dataset-path "$dataset_path" \
            --num-prompts "$NUM_PROMPTS" \
            --no-oversample \
            --max-concurrency "$concurrency" \
            --request-rate "$REQUEST_RATE" \
            --temperature 0 \
            --disable-shuffle \
            --custom-output-len "$OUTPUT_LEN" \
            "${ignore_eos_args[@]}" \
            --num-warmups "$NUM_WARMUPS" \
            --save-detailed \
            --extra-body '{"temperature":0.0}' \
            --percentile-metrics ttft,tpot,itl,e2el \
            --metric-percentiles 50,95,99 \
            --save-result \
            --result-dir "$result_dir" \
            --result-filename result.json \
            2>&1 | tee "$result_dir/benchmark.log"
    done
done

python3 - "$run_root" <<'PY' | tee "$run_root/summary.md"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("| case | completed | failed | duration_s | output_tok_s | p99_ttft_ms | p99_tpot_ms |")
print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
for path in sorted(root.glob("c*-r*/*/result.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    case = f"{path.parent.parent.name}/{path.parent.name}"
    print(
        f"| {case} | {data['completed']} | {data['failed']} | "
        f"{data['duration']:.6f} | {data['output_throughput']:.6f} | "
        f"{data['p99_ttft_ms']:.6f} | {data['p99_tpot_ms']:.6f} |"
    )
PY
