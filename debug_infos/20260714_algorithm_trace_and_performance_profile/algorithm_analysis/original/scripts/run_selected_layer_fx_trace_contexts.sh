#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_DIR="${MODEL_DIR:-../Qwen3.5-27B}"
PORT="${PORT:-8002}"
FX_TRACE_RUN_DIR="${FX_TRACE_RUN_DIR:-./profile_runs/selected_layer_fx_$(date +%Y%m%d_%H%M%S)}"
BENCH_DIR="$FX_TRACE_RUN_DIR/bench_results"
KEEP_SERVER="${KEEP_SERVER:-0}"
STOP_EXISTING="${STOP_EXISTING:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
NUM_WARMUPS="${NUM_WARMUPS:-0}"
CONTEXTS="${CONTEXTS:-4-8K,8-16K,16-32K}"
FX_PATCH_PYTHONPATH="$SCRIPT_DIR/fx_trace_patch"
FX_PHASES="${FX_PHASES:-prefill_chunk}"
FX_TRACING_MODE="${FX_TRACING_MODE:-fake}"
FX_TRACE_ENFORCE_EAGER="${FX_TRACE_ENFORCE_EAGER:-1}"
FX_LAYERWISE_SOURCE="${FX_LAYERWISE_SOURCE:-./profile_runs/patch_trace_20260707_codex}"
FX_RATIONALE="${FX_RATIONALE:-layer3 is the dominant full-attention outlier across all contexts; layer31 is the long-context-specific outlier; layer59 is a stable late-layer representative. Event ids use the hottest observed chunk for each selected layer/context from the layerwise patch trace.}"
ACTIVE_SERVER_PID_FILE=""

export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-1800000}"

mkdir -p "$FX_TRACE_RUN_DIR" "$BENCH_DIR"

log() {
  printf '[selected-layer-fx] %s\n' "$*"
}

vllm_ready() {
  curl --noproxy 127.0.0.1,localhost -fsS \
    "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1
}

stop_existing_vllm() {
  if [ "$STOP_EXISTING" != "1" ]; then
    return
  fi
  local pids
  pids="$(pgrep -f 'vllm serve' || true)"
  if [ -n "$pids" ]; then
    log "Stopping existing vLLM serve processes: $pids"
    kill $pids || true
    sleep 5
  fi
}

targets_for_context() {
  local label="$1"
  case "$label" in
    4-8K)
      printf '%s\n' "${FX_TARGETS_4_8K:-1:3,1:31,1:59}"
      ;;
    8-16K)
      printf '%s\n' "${FX_TARGETS_8_16K:-1:3,4:31,3:59}"
      ;;
    16-32K)
      printf '%s\n' "${FX_TARGETS_16_32K:-1:3,2:31,4:59}"
      ;;
    *)
      printf '%s\n' "${FX_TARGETS:-1:3}"
      ;;
  esac
}

start_fx_server() {
  local label="$1"
  local fx_dir="$2"
  local targets="$3"
  local server_log="$4"
  local server_pid_file="$5"
  local arm_file="$fx_dir/trace_enabled"
  local extra_args=()

  mkdir -p "$fx_dir"
  rm -f "$arm_file"
  if [ "$FX_TRACE_ENFORCE_EAGER" = "1" ]; then
    extra_args+=(--enforce-eager)
  fi

  log "Starting vLLM FX server for context=$label targets=$targets log=$server_log"
  VLLM_SELECTED_LAYER_FX_ENABLE=1 \
  VLLM_SELECTED_LAYER_FX_CONTEXT="$label" \
  VLLM_SELECTED_LAYER_FX_DIR="$fx_dir" \
  VLLM_SELECTED_LAYER_FX_ARM_FILE="$arm_file" \
  VLLM_SELECTED_LAYER_FX_TARGETS="$targets" \
  VLLM_SELECTED_LAYER_FX_PHASES="$FX_PHASES" \
  VLLM_SELECTED_LAYER_FX_TRACING_MODE="$FX_TRACING_MODE" \
  VLLM_SELECTED_LAYER_FX_LAYERWISE_SOURCE="$FX_LAYERWISE_SOURCE" \
  VLLM_SELECTED_LAYER_FX_RATIONALE="$FX_RATIONALE" \
  PYTHONPATH="$FX_PATCH_PYTHONPATH:${PYTHONPATH:-}" \
  nohup vllm serve "$MODEL_DIR" \
    --served-model-name Qwen3.5-27B \
    --port "$PORT" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.95 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    "${extra_args[@]}" \
    >"$server_log" 2>&1 &
  printf '%s\n' "$!" > "$server_pid_file"
  ACTIVE_SERVER_PID_FILE="$server_pid_file"
}

wait_for_server() {
  local server_log="$1"
  local server_pid_file="$2"
  log "Waiting for vLLM on 127.0.0.1:$PORT"
  for attempt in $(seq 1 240); do
    if vllm_ready; then
      log "vLLM server is ready"
      return
    fi
    if [ -f "$server_pid_file" ] && ! kill -0 "$(cat "$server_pid_file")" 2>/dev/null; then
      tail -n 160 "$server_log" >&2 || true
      exit 1
    fi
    log "Still waiting ($attempt/240)"
    sleep 5
  done
  tail -n 160 "$server_log" >&2 || true
  exit 1
}

cleanup() {
  if [ "$KEEP_SERVER" != "1" ] && [ -n "$ACTIVE_SERVER_PID_FILE" ] && [ -f "$ACTIVE_SERVER_PID_FILE" ]; then
    local pid
    pid="$(cat "$ACTIVE_SERVER_PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
      log "Stopping FX trace server pid $pid"
      kill "$pid" || true
    fi
    ACTIVE_SERVER_PID_FILE=""
  fi
}
trap cleanup EXIT

stop_fx_server() {
  cleanup
  sleep 3
}

run_one_fx_trace() {
  local label="$1"
  local dataset_path="$2"
  local context_dir="$FX_TRACE_RUN_DIR/contexts/$label"
  local result_dir="$BENCH_DIR/$label"
  local fx_dir="$context_dir/fx_trace"
  local arm_file="$fx_dir/trace_enabled"
  local targets
  local server_log="$context_dir/vllm_fx_server.log"
  local server_pid_file="$context_dir/vllm_fx_server.pid"

  targets="$(targets_for_context "$label")"
  mkdir -p "$context_dir" "$result_dir" "$fx_dir"
  start_fx_server "$label" "$fx_dir" "$targets" "$server_log" "$server_pid_file"
  wait_for_server "$server_log" "$server_pid_file"

  log "Running selected-layer FX trace context=$label dataset=$dataset_path"
  : > "$arm_file"
  vllm bench serve \
    --backend openai-chat \
    --host 127.0.0.1 \
    --port "$PORT" \
    --endpoint /v1/chat/completions \
    --model Qwen3.5-27B \
    --tokenizer "$MODEL_DIR" \
    --dataset-name custom \
    --dataset-path "$dataset_path" \
    --num-prompts 1 \
    --no-oversample \
    --max-concurrency 1 \
    --request-rate 1 \
    --temperature 0 \
    --disable-shuffle \
    --custom-output-len "$OUTPUT_LEN" \
    --num-warmups "$NUM_WARMUPS" \
    --save-detailed \
    --extra-body '{"temperature":0.0}' \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99 \
    --save-result \
    --result-dir "$result_dir" \
    --result-filename result.json \
    | tee "$context_dir/bench.log"
  rm -f "$arm_file"

  python3 - "$result_dir/result.json" <<'PY'
import json
import sys

result_path = sys.argv[1]
with open(result_path, encoding="utf-8") as handle:
    result = json.load(handle)
if result.get("completed") != 1 or result.get("failed") != 0:
    raise SystemExit(
        f"invalid benchmark result: completed={result.get('completed')} "
        f"failed={result.get('failed')}"
    )
PY

  cp -a "$result_dir/result.json" "$context_dir/result.json"
  stop_fx_server
  log "Context $label selected-layer FX artifacts saved under $context_dir"
}

context_enabled() {
  local label="$1"
  case ",$CONTEXTS," in
    *",$label,"*) return 0 ;;
    *) return 1 ;;
  esac
}

main() {
  stop_existing_vllm
  cat > "$FX_TRACE_RUN_DIR/profile_plan.tsv" <<'EOF'
4-8K	./4-8K_throughput.jsonl
8-16K	./8-16K_throughput.jsonl
16-32K	./16-32K_throughput.jsonl
EOF

  while IFS="$(printf '\t')" read -r label dataset_path; do
    if ! context_enabled "$label"; then
      log "Skipping context=$label because CONTEXTS=$CONTEXTS"
      continue
    fi
    run_one_fx_trace "$label" "$dataset_path"
  done < "$FX_TRACE_RUN_DIR/profile_plan.tsv"

  python3 "$SCRIPT_DIR/summarize_selected_layer_fx_trace.py" "$FX_TRACE_RUN_DIR"

  log "All selected-layer FX traces complete: $FX_TRACE_RUN_DIR"
  find "$FX_TRACE_RUN_DIR/contexts" -maxdepth 6 -type f -printf '%p %s\n' \
    | sort \
    | tee "$FX_TRACE_RUN_DIR/artifacts.txt"
}

main "$@"
