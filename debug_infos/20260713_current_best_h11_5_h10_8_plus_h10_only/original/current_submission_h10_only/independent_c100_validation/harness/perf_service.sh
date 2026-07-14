#!/usr/bin/env bash
# Fresh C100 performance service: candidate wheel, H10 on, S32/Hg3 off.

set -euo pipefail

HERE=/public/home/tangyu408/testdata/goal_runs/20260712_r27_p9_c100_fixed_all3_600/harness
P9_PREP=/public/home/tangyu408/testdata/goal_runs/20260712_r27_s32_hg3_production_prep/P9_output_attribution_prep
RUNTIME_GUARD=/public/home/tangyu408/testdata/goal_runs/20260712_r27_h10_18_production_closure_prep/runtime_guard.py
START=/public/home/tangyu408/testdata/start_vllm.sh
START_SHA=7c3e8c5ecdf02109e02af8c3b5ba05050b26339c7f50869b5288eea359364fad
PROFILE=gfx936_qwen3_5_27b_bf16_tn_m4096
PROFILE_SHA=41742b4c5d071fdf9085c46ad4ec1743d7e4f410431c05ff39b0e0f293548a0b
PORT=8001

die() { echo "ERROR: $*" >&2; exit 1; }
[[ ${P9_ALLOW_C100_PERFORMANCE-} == YES_600S_AUTHORIZED ]] || die "missing C100 performance authorization"
[[ $# -eq 2 ]] || die "usage: perf_service.sh {start|stop} STATE_DIR"
ACTION=$1
STATE_DIR=$2
PID_FILE=$STATE_DIR/service.pid
PGID_FILE=$STATE_DIR/service.pgid
STATE_FILE=$STATE_DIR/service_state.json
SERVER_LOG=$STATE_DIR/server.log

cleanup_failed_start() {
  local rc=$?
  if [[ $rc -ne 0 && -f "$PGID_FILE" ]]; then
    local pgid
    pgid=$(<"$PGID_FILE")
    [[ $pgid =~ ^[1-9][0-9]*$ ]] && kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
  return "$rc"
}

case "$ACTION" in
  start)
    [[ ! -e "$STATE_DIR" ]] || die "refusing reused service directory: $STATE_DIR"
    mkdir -p "$STATE_DIR"
    trap cleanup_failed_start EXIT
    python3 "$RUNTIME_GUARD" --port "$PORT" --require-clean >"$STATE_DIR/runtime_guard.before.json"
    python3 "$P9_PREP/verify_source_guard.py" --report "$STATE_DIR/source_guard.before.json" >/dev/null
    (
      cd /tmp
      python3 "$P9_PREP/verify_wheel_site.py" --kind candidate \
        --report "$STATE_DIR/installed_identity.before.json" >/dev/null
    )
    echo "$START_SHA  $START" | sha256sum -c - >/dev/null

    loader_hits=0
    for _ in $(seq 1 10); do
      if python3 -c 'import torch' >/dev/null 2>&1; then
        loader_hits=$((loader_hits + 1))
        [[ $loader_hits -ge 3 ]] && break
      else
        loader_hits=0
      fi
      sleep 1
    done
    [[ $loader_hits -ge 3 ]] || die "DTK loader stability gate failed"

    env_args=(
      -u PYTORCH_TUNABLEOP_FILENAME
      -u PYTORCH_TUNABLEOP_VERBOSE
      -u PYTORCH_TUNABLEOP_VEROBSE
      VLLM_PORT="$PORT"
      VLLM_ROCM_TUNABLEOP_PROFILE="$PROFILE"
      VLLM_ROCM_TUNABLEOP_PROFILE_SHA256="$PROFILE_SHA"
      VLLM_CSCC_ENABLE_GQA6_S32_DECODE=0
      VLLM_CSCC_ENABLE_GDN_HG3_CHUNK_O=0
      PYTORCH_TUNABLEOP_ENABLED=1
      PYTORCH_TUNABLEOP_TUNING=0
      PYTORCH_TUNABLEOP_RECORD_UNTUNED=0
      PYTORCH_TUNABLEOP_ROCBLAS_ENABLED=1
      PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
    )
    {
      printf 'cd %q\n' /public/home/tangyu408/testdata
      printf 'setsid env '; printf '%q ' "${env_args[@]}"; printf 'bash %q\n' "$START"
    } >"$STATE_DIR/start.command.txt"
    cat >"$STATE_DIR/feature_env.txt" <<EOF
state=C100
purpose=fixed_all3_screen_performance
VLLM_ROCM_TUNABLEOP_PROFILE=$PROFILE
VLLM_ROCM_TUNABLEOP_PROFILE_SHA256=$PROFILE_SHA
VLLM_CSCC_ENABLE_GQA6_S32_DECODE=0
VLLM_CSCC_ENABLE_GDN_HG3_CHUNK_O=0
PYTORCH_TUNABLEOP_ENABLED=1
PYTORCH_TUNABLEOP_TUNING=0
PYTORCH_TUNABLEOP_RECORD_UNTUNED=0
PYTORCH_TUNABLEOP_VERBOSE=<unset>
PYTORCH_TUNABLEOP_VEROBSE=<unset>
PYTORCH_TUNABLEOP_ROCBLAS_ENABLED=1
PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
EOF
    (
      cd /public/home/tangyu408/testdata
      setsid env "${env_args[@]}" bash "$START" >"$SERVER_LOG" 2>&1 &
      echo $! >"$PID_FILE"
    )
    pid=$(<"$PID_FILE")
    [[ $pid =~ ^[1-9][0-9]*$ ]] || die "invalid service PID"
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null && break; sleep 1; done
    kill -0 "$pid" 2>/dev/null || die "service exited before PGID capture"
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
    [[ $pgid == "$pid" ]] || die "setsid invariant failed: pid=$pid pgid=$pgid"
    echo "$pgid" >"$PGID_FILE"

    ready=0
    ready_hits=0
    for _ in $(seq 1 1800); do
      kill -0 "$pid" 2>/dev/null || die "service exited before readiness"
      if curl --noproxy '*' --silent --show-error --fail --max-time 2 \
        "http://127.0.0.1:$PORT/v1/models" >"$STATE_DIR/models.json.tmp" 2>/dev/null; then
        ready_hits=$((ready_hits + 1))
        if [[ $ready_hits -ge 3 ]]; then
          mv "$STATE_DIR/models.json.tmp" "$STATE_DIR/models.json"
          ready=1
          break
        fi
      else
        ready_hits=0
      fi
      sleep 1
    done
    [[ $ready == 1 ]] || die "service readiness timeout"
    wheel_sha=$(python3 - "$STATE_DIR/installed_identity.before.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["wheel_sha256"])
PY
)
    python3 - "$STATE_FILE" "$wheel_sha" "$pid" "$pgid" <<'PY'
import json,os,sys
path,wheel,pid,pgid=sys.argv[1:]
payload={
 "schema":"p9-c100-performance-service-v1","status":"ready","state":"C100",
 "wheel_kind":"candidate","wheel_sha256":wheel,
 "features":{"h10_confirmed":1,"s32":0,"hg3":0},
 "pid":int(pid),"pgid":int(pgid),"fresh_process":True,
 "purpose":"fixed_all3_screen_performance",
 "tunable_controls":{"verbose":"unset","tuning":0,"record_untuned":0},
}
tmp=path+".tmp"
with open(tmp,"w",encoding="utf-8") as f: json.dump(payload,f,indent=2,sort_keys=True); f.write("\n")
os.replace(tmp,path)
PY
    trap - EXIT
    echo "C100 performance service ready: pid=$pid pgid=$pgid"
    ;;
  stop)
    [[ -f "$PGID_FILE" ]] || die "missing PGID file"
    pgid=$(<"$PGID_FILE")
    [[ $pgid =~ ^[1-9][0-9]*$ ]] || die "invalid PGID"
    if kill -0 -- "-$pgid" 2>/dev/null; then
      kill -TERM -- "-$pgid"
      for _ in $(seq 1 120); do kill -0 -- "-$pgid" 2>/dev/null || break; sleep 1; done
    fi
    if kill -0 -- "-$pgid" 2>/dev/null; then
      kill -KILL -- "-$pgid"
      for _ in $(seq 1 30); do kill -0 -- "-$pgid" 2>/dev/null || break; sleep 1; done
    fi
    kill -0 -- "-$pgid" 2>/dev/null && die "service process group remains live"
    python3 "$RUNTIME_GUARD" --port "$PORT" --require-clean >"$STATE_DIR/runtime_guard.after.json"
    python3 "$P9_PREP/verify_source_guard.py" --report "$STATE_DIR/source_guard.after.json" >/dev/null
    echo '{"status":"stopped","runtime_clean":true}' >"$STATE_DIR/service_stop.json"
    echo "C100 performance service stopped"
    ;;
  *) die "unknown action: $ACTION" ;;
esac
