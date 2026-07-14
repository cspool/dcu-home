#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_DIR="${MODEL_DIR:-../Qwen3.5-27B}"
PORT="${PORT:-8001}"
PATCH_TRACE_RUN_DIR="${PATCH_TRACE_RUN_DIR:-./profile_runs/patch_trace_$(date +%Y%m%d_%H%M%S)}"
BENCH_DIR="$PATCH_TRACE_RUN_DIR/bench_results"
KEEP_SERVER="${KEEP_SERVER:-0}"
STOP_EXISTING="${STOP_EXISTING:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
NUM_WARMUPS="${NUM_WARMUPS:-0}"
CONTEXTS="${CONTEXTS:-4-8K,8-16K,16-32K}"
TRACE_PATCH_PYTHONPATH="$SCRIPT_DIR/trace_patch"
ACTIVE_SERVER_PID_FILE=""

export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-1800000}"

mkdir -p "$PATCH_TRACE_RUN_DIR" "$BENCH_DIR"

log() {
  printf '[patch-trace] %s\n' "$*"
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

start_trace_server() {
  local label="$1"
  local trace_dir="$2"
  local server_log="$3"
  local server_pid_file="$4"
  local arm_file="$trace_dir/trace_enabled"

  mkdir -p "$trace_dir"
  rm -f "$arm_file"

  log "Starting vLLM trace server for context=$label; log=$server_log"
  VLLM_TRACE_PATCH_ENABLE=1 \
  VLLM_TRACE_PATCH_CONTEXT="$label" \
  VLLM_TRACE_PATCH_DIR="$trace_dir" \
  VLLM_TRACE_PATCH_ARM_FILE="$arm_file" \
  PYTHONPATH="$TRACE_PATCH_PYTHONPATH:${PYTHONPATH:-}" \
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
    >"$server_log" 2>&1 &
  printf '%s\n' "$!" > "$server_pid_file"
  ACTIVE_SERVER_PID_FILE="$server_pid_file"
}

wait_for_server() {
  local server_log="$1"
  local server_pid_file="$2"
  log "Waiting for vLLM on 127.0.0.1:$PORT"
  for attempt in $(seq 1 180); do
    if vllm_ready; then
      log "vLLM server is ready"
      return
    fi
    if [ -f "$server_pid_file" ] && ! kill -0 "$(cat "$server_pid_file")" 2>/dev/null; then
      tail -n 120 "$server_log" >&2 || true
      exit 1
    fi
    log "Still waiting ($attempt/180)"
    sleep 5
  done
  tail -n 120 "$server_log" >&2 || true
  exit 1
}

cleanup() {
  if [ "$KEEP_SERVER" != "1" ] && [ -n "$ACTIVE_SERVER_PID_FILE" ] && [ -f "$ACTIVE_SERVER_PID_FILE" ]; then
    local pid
    pid="$(cat "$ACTIVE_SERVER_PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
      log "Stopping trace server pid $pid"
      kill "$pid" || true
    fi
    ACTIVE_SERVER_PID_FILE=""
  fi
}
trap cleanup EXIT

stop_trace_server() {
  cleanup
  sleep 3
}

run_one_trace() {
  local label="$1"
  local dataset_path="$2"
  local context_dir="$PATCH_TRACE_RUN_DIR/contexts/$label"
  local result_dir="$BENCH_DIR/$label"
  local trace_dir="$context_dir/patch_trace"
  local arm_file="$trace_dir/trace_enabled"
  local server_log="$context_dir/vllm_trace_server.log"
  local server_pid_file="$context_dir/vllm_trace_server.pid"

  mkdir -p "$context_dir" "$result_dir" "$trace_dir"
  start_trace_server "$label" "$trace_dir" "$server_log" "$server_pid_file"
  wait_for_server "$server_log" "$server_pid_file"

  log "Tracing context=$label dataset=$dataset_path"
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
with open(result_path, encoding="utf-8") as f:
    result = json.load(f)
if result.get("completed") != 1 or result.get("failed") != 0:
    raise SystemExit(
        f"invalid benchmark result: completed={result.get('completed')} "
        f"failed={result.get('failed')}"
    )
PY

  cp -a "$result_dir/result.json" "$context_dir/result.json"
  stop_trace_server
  log "Context $label patch trace artifacts saved under $context_dir"
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
  cat > "$PATCH_TRACE_RUN_DIR/profile_plan.tsv" <<'EOF'
4-8K	./4-8K_throughput.jsonl
8-16K	./8-16K_throughput.jsonl
16-32K	./16-32K_throughput.jsonl
EOF

  while IFS="$(printf '\t')" read -r label dataset_path; do
    if ! context_enabled "$label"; then
      log "Skipping context=$label because CONTEXTS=$CONTEXTS"
      continue
    fi
    run_one_trace "$label" "$dataset_path"
  done < "$PATCH_TRACE_RUN_DIR/profile_plan.tsv"

  python3 "$SCRIPT_DIR/summarize_patch_trace.py" "$PATCH_TRACE_RUN_DIR"

  log "All patch traces complete: $PATCH_TRACE_RUN_DIR"
  find "$PATCH_TRACE_RUN_DIR/contexts" -maxdepth 4 -type f -printf '%p %s\n' \
    | sort \
    | tee "$PATCH_TRACE_RUN_DIR/artifacts.txt"
}

main "$@"
