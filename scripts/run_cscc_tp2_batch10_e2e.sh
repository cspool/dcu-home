#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${1:-$ROOT/../Qwen3.5-27B}}"
OFFICIAL_TEST_DIR="${OFFICIAL_TEST_DIR:-}"
TEST_TARGET="${TEST_TARGET:-all}"
RUN_SETUP="${RUN_SETUP:-1}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"
PORT="${PORT:-8001}"
CONCURRENCY="${CONCURRENCY:-10}"
LOG_DIR="${LOG_DIR:-$ROOT/run-cscc-tp2-batch10-$(date -u +%Y%m%dT%H%M%SZ)}"

fail() {
    echo "run_cscc_tp2_batch10_e2e: ERROR: $*" >&2
    exit 1
}

case "$TEST_TARGET" in
    health | throughput | accuracy | all) ;;
    *) fail "TEST_TARGET must be health, throughput, accuracy, or all" ;;
esac
case "$RUN_SETUP" in
    0 | 1) ;;
    *) fail "RUN_SETUP must be 0 or 1" ;;
esac
[[ -f "$MODEL_DIR/config.json" ]] || \
    fail "model config not found: $MODEL_DIR/config.json"
[[ "$READY_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || \
    fail "READY_TIMEOUT must be a positive integer"
[[ "$PORT" =~ ^[1-9][0-9]*$ ]] && ((PORT <= 65535)) || \
    fail "PORT must be in 1..65535"
command -v curl >/dev/null 2>&1 || fail "curl is required"
command -v setsid >/dev/null 2>&1 || fail "setsid is required"

if [[ "$TEST_TARGET" != "health" ]]; then
    [[ "$PORT" == "8001" ]] || \
        fail "the supplied official scripts use port 8001; set PORT=8001"
    [[ -d "$OFFICIAL_TEST_DIR" ]] || \
        fail "set OFFICIAL_TEST_DIR to the unpacked official evaluator"
fi

if [[ "$RUN_SETUP" == "1" ]]; then
    WHEEL="${WHEEL:-}" \
    DIST_DIR="${DIST_DIR:-$ROOT/dist-cscc-tp2-batch10}" \
    MAX_JOBS="${MAX_JOBS:-16}" \
    OFFICIAL_TEST_DIR="$OFFICIAL_TEST_DIR" \
        bash "$ROOT/scripts/setup_cscc_tp2_batch10.sh"
fi

# Refuse to attach to an unrelated service. This keeps cleanup scoped to the
# process group created by this script.
python3 - "$PORT" <<'PY'
import socket
import sys

sock = socket.socket()
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
except OSError as exc:
    raise SystemExit(f"port {sys.argv[1]} is already in use: {exc}")
finally:
    sock.close()
PY

mkdir -p "$LOG_DIR"
server_log="$LOG_DIR/server.log"
server_pid=""

stop_server() {
    if [[ -z "$server_pid" ]] || ! kill -0 "$server_pid" 2>/dev/null; then
        return
    fi
    echo "Stopping service process group $server_pid"
    kill -INT -- "-$server_pid" 2>/dev/null || true
    for _ in $(seq 1 60); do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            wait "$server_pid" 2>/dev/null || true
            return
        fi
        sleep 0.5
    done
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Starting TP2 service; log: $server_log"
setsid env MODEL_DIR="$MODEL_DIR" PORT="$PORT" \
    bash "$ROOT/scripts/serve_cscc_tp2_batch10.sh" \
    >"$server_log" 2>&1 &
server_pid=$!

deadline=$((SECONDS + READY_TIMEOUT))
until curl --noproxy '*' --fail --silent --show-error \
    "http://127.0.0.1:$PORT/v1/models" >"$LOG_DIR/models.json"; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        wait "$server_pid" || true
        tail -n 120 "$server_log" >&2 || true
        fail "service exited before becoming ready"
    fi
    if ((SECONDS >= deadline)); then
        tail -n 120 "$server_log" >&2 || true
        fail "service did not become ready within ${READY_TIMEOUT}s"
    fi
    sleep 2
done
echo "SERVICE READY: http://127.0.0.1:$PORT"

if [[ "$TEST_TARGET" == "health" ]]; then
    echo "HEALTH PASS; official throughput and accuracy were not run"
    exit 0
fi

run_throughput() {
    [[ -f "$OFFICIAL_TEST_DIR/run_throughput.sh" ]] || \
        fail "missing $OFFICIAL_TEST_DIR/run_throughput.sh"
    (
        cd "$OFFICIAL_TEST_DIR"
        MODEL_DIR="$MODEL_DIR" CONCURRENCY="$CONCURRENCY" \
            bash ./run_throughput.sh all
    ) 2>&1 | tee "$LOG_DIR/official-throughput.log"
}

run_accuracy() {
    [[ -f "$OFFICIAL_TEST_DIR/run_accuracy.sh" ]] || \
        fail "missing $OFFICIAL_TEST_DIR/run_accuracy.sh"
    (
        cd "$OFFICIAL_TEST_DIR"
        MODEL_PATH="$MODEL_DIR" bash ./run_accuracy.sh all
    ) 2>&1 | tee "$LOG_DIR/official-accuracy.log"
}

case "$TEST_TARGET" in
    throughput) run_throughput ;;
    accuracy) run_accuracy ;;
    all)
        run_throughput
        run_accuracy
        ;;
esac

echo "E2E PASS; logs: $LOG_DIR"
