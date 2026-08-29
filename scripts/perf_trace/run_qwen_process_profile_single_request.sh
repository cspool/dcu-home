#!/usr/bin/env bash
set -euo pipefail

: "${ROOT_DIR:?set ROOT_DIR}"
: "${CONFIG:?set CONFIG}"
: "${TAG:?set TAG}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${RUNTIME_ARTIFACT_ROOT:?set RUNTIME_ARTIFACT_ROOT}"
: "${CONTRACT_PATH:?set CONTRACT_PATH}"
: "${MODEL_ROOT:?set MODEL_ROOT}"
: "${SERVED_MODEL_NAME:?set SERVED_MODEL_NAME}"
: "${MAX_NEW_TOKENS:?set MAX_NEW_TOKENS}"
: "${WARMUP_ITERS:?set WARMUP_ITERS}"
: "${DCU_DEVICE:?set DCU_DEVICE}"
: "${PRA_BACKEND_PERF_PROCESS_PROFILE:?set PRA_BACKEND_PERF_PROCESS_PROFILE}"

PRA_BACKEND_PERF_PROCESS_TARGETS="${PRA_BACKEND_PERF_PROCESS_TARGETS:-}"
PRA_BACKEND_PERF_PROCESS_TARGETS_FILE="${PRA_BACKEND_PERF_PROCESS_TARGETS_FILE:-}"
PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS="${PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS:-}"
PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS_FILE="${PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS_FILE:-}"
PRA_BACKEND_TUNABLEOP_PROFILE_SHA256_OVERRIDE="${PRA_BACKEND_TUNABLEOP_PROFILE_SHA256_OVERRIDE:-}"
WORKFLOW05_LIVE_UTILIZATION_MODE="${WORKFLOW05_LIVE_UTILIZATION_MODE:-disabled}"
PRA_BACKEND_LIVE_UTIL_COLLECTOR="${PRA_BACKEND_LIVE_UTIL_COLLECTOR:-}"
PRA_BACKEND_LIVE_UTIL_COLLECTOR_SHA256="${PRA_BACKEND_LIVE_UTIL_COLLECTOR_SHA256:-}"
PRA_BACKEND_LIVE_UTIL_INTERVAL_US="${PRA_BACKEND_LIVE_UTIL_INTERVAL_US:-500}"
FRESH_RUN_LINEAGE_MANIFEST="${FRESH_RUN_LINEAGE_MANIFEST:-}"

if [[ "${CONFIG}" != "qwen3.5-27b-vllm-pra-eager-gfx936" ]]; then
    echo "unreviewed CONFIG: ${CONFIG}" >&2
    exit 2
fi
if [[ "${MAX_NEW_TOKENS}" != "32" ]]; then
    echo "MAX_NEW_TOKENS must be 32" >&2
    exit 2
fi
if [[ "${WARMUP_ITERS}" != "1" ]]; then
    echo "WARMUP_ITERS must be 1" >&2
    exit 2
fi
if [[ "${PRA_BACKEND_PERF_PROCESS_PROFILE}" != "1" ]]; then
    echo "process profiling must equal 1" >&2
    exit 2
fi
if [[ -z "${PRA_BACKEND_PERF_PROCESS_TARGETS}" ]] &&
    [[ -z "${PRA_BACKEND_PERF_PROCESS_TARGETS_FILE}" ]]; then
    echo "process target CSV or file must not be empty" >&2
    exit 2
fi
if [[ -z "${PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS}" ]] &&
    [[ -z "${PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS_FILE}" ]]; then
    echo "exact process range target CSV or file must not be empty" >&2
    exit 2
fi
PROCESS_RANGE_PATTERN='^pra\.fx_process\.input[0-9]+_layer[0-9]+\.[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)?(,pra\.fx_process\.input[0-9]+_layer[0-9]+\.[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)?)*$'
if [[ -n "${PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS}" ]] &&
    [[ ! "${PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS}" =~ ${PROCESS_RANGE_PATTERN} ]]; then
    echo "exact process range target list has invalid syntax" >&2
    exit 2
fi
if [[ "${DCU_DEVICE}" != "1" ]]; then
    echo "reviewed device policy requires physical DCU 1" >&2
    exit 2
fi
if [[ ! "${TAG}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "TAG contains unsupported characters" >&2
    exit 2
fi

ROOT_DIR="$(readlink -f "${ROOT_DIR}")"
RUNTIME_ARTIFACT_ROOT="$(readlink -f "${RUNTIME_ARTIFACT_ROOT}")"
RUNTIME_RUN_ROOT="$(readlink -f "${RUNTIME_ARTIFACT_ROOT}/../..")"
OUTPUT_PARENT="$(readlink -f "$(dirname "${OUTPUT_DIR}")")"
OUTPUT_DIR="${OUTPUT_PARENT}/$(basename "${OUTPUT_DIR}")"
case "${RUNTIME_ARTIFACT_ROOT}/" in
    "${RUNTIME_RUN_ROOT}/artifacts/"*) ;;
    *)
        echo "RUNTIME_ARTIFACT_ROOT must remain under the current run artifacts root" >&2
        exit 2
        ;;
esac
if [[ -n "${FRESH_RUN_LINEAGE_MANIFEST}" ]] &&
    [[ -n "$(find "${RUNTIME_ARTIFACT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "refusing non-empty fresh-run RUNTIME_ARTIFACT_ROOT: ${RUNTIME_ARTIFACT_ROOT}" >&2
    exit 1
fi
TUNABLEOP_PROFILE_PATH="${ROOT_DIR}/vllm/platforms/tunable_profiles/gfx936_qwen3_5_27b_bf16_tn_m4096.csv"
if [[ ! -f "${TUNABLEOP_PROFILE_PATH}" ]]; then
    echo "missing current TunableOp profile: ${TUNABLEOP_PROFILE_PATH}" >&2
    exit 2
fi
CURRENT_TUNABLEOP_PROFILE_SHA256="$(sha256sum "${TUNABLEOP_PROFILE_PATH}" | awk '{print $1}')"
if [[ -n "${PRA_BACKEND_TUNABLEOP_PROFILE_SHA256_OVERRIDE}" ]]; then
    if [[ ! "${PRA_BACKEND_TUNABLEOP_PROFILE_SHA256_OVERRIDE}" =~ ^[0-9a-f]{64}$ ]] ||
        [[ "${PRA_BACKEND_TUNABLEOP_PROFILE_SHA256_OVERRIDE}" != "${CURRENT_TUNABLEOP_PROFILE_SHA256}" ]]; then
        echo "current TunableOp profile SHA-256 override does not match the source file" >&2
        exit 2
    fi
fi

for target_file_variable in \
    PRA_BACKEND_PERF_PROCESS_TARGETS_FILE \
    PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS_FILE; do
    target_file="${!target_file_variable}"
    if [[ -z "${target_file}" ]]; then
        continue
    fi
    if [[ ! -f "${target_file}" ]]; then
        echo "missing target file for ${target_file_variable}: ${target_file}" >&2
        exit 2
    fi
    target_file="$(readlink -f "${target_file}")"
    case "${target_file}" in
        "${RUNTIME_RUN_ROOT}/artifacts/R06/"*) ;;
        *)
            echo "${target_file_variable} must be a same-run R06 artifact" >&2
            exit 2
            ;;
    esac
    printf -v "${target_file_variable}" '%s' "${target_file}"
done
export PRA_BACKEND_PERF_PROCESS_TARGETS PRA_BACKEND_PERF_PROCESS_TARGETS_FILE
export PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS
export PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS_FILE

case "${OUTPUT_DIR}/" in
    "${RUNTIME_ARTIFACT_ROOT}/"*) ;;
    *)
        echo "OUTPUT_DIR must remain under RUNTIME_ARTIFACT_ROOT" >&2
        exit 2
        ;;
esac
if [[ -e "${OUTPUT_DIR}" ]] &&
    [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "refusing to reuse non-empty OUTPUT_DIR: ${OUTPUT_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CONTRACT_PATH}" ]]; then
    echo "missing frozen contract: ${CONTRACT_PATH}" >&2
    exit 1
fi
CONTRACT_PATH="$(readlink -f "${CONTRACT_PATH}")"
case "${CONTRACT_PATH}" in
    "${RUNTIME_RUN_ROOT}/artifacts/R01/"*) ;;
    *)
        echo "CONTRACT_PATH must be a same-run R01 artifact" >&2
        exit 2
        ;;
esac
LINEAGE_ARGS=()
if [[ -n "${FRESH_RUN_LINEAGE_MANIFEST}" ]]; then
    if [[ ! -f "${FRESH_RUN_LINEAGE_MANIFEST}" ]]; then
        echo "missing fresh-run lineage manifest: ${FRESH_RUN_LINEAGE_MANIFEST}" >&2
        exit 2
    fi
    FRESH_RUN_LINEAGE_MANIFEST="$(readlink -f "${FRESH_RUN_LINEAGE_MANIFEST}")"
    case "${FRESH_RUN_LINEAGE_MANIFEST}" in
        "${RUNTIME_RUN_ROOT}/artifacts/R06/"*) ;;
        *)
            echo "FRESH_RUN_LINEAGE_MANIFEST must be a same-run R06 artifact" >&2
            exit 2
            ;;
    esac
    LINEAGE_ARGS=(--lineage-manifest "${FRESH_RUN_LINEAGE_MANIFEST}")
fi

case "${WORKFLOW05_LIVE_UTILIZATION_MODE}" in
    disabled) ;;
    rsmi_se_snapshot)
        if [[ ! -f "${PRA_BACKEND_LIVE_UTIL_COLLECTOR}" ]]; then
            echo "missing live-utilization collector: ${PRA_BACKEND_LIVE_UTIL_COLLECTOR}" >&2
            exit 2
        fi
        PRA_BACKEND_LIVE_UTIL_COLLECTOR="$(readlink -f "${PRA_BACKEND_LIVE_UTIL_COLLECTOR}")"
        if [[ ! "${PRA_BACKEND_LIVE_UTIL_COLLECTOR_SHA256}" =~ ^[0-9a-f]{64}$ ]] ||
            [[ "$(sha256sum "${PRA_BACKEND_LIVE_UTIL_COLLECTOR}" | awk '{print $1}')" != "${PRA_BACKEND_LIVE_UTIL_COLLECTOR_SHA256}" ]]; then
            echo "live-utilization collector SHA-256 mismatch" >&2
            exit 2
        fi
        if [[ ! "${PRA_BACKEND_LIVE_UTIL_INTERVAL_US}" =~ ^[0-9]+$ ]] ||
            (( PRA_BACKEND_LIVE_UTIL_INTERVAL_US < 100 || PRA_BACKEND_LIVE_UTIL_INTERVAL_US >= 1000 )); then
            echo "live-utilization interval must be 100-999 microseconds" >&2
            exit 2
        fi
        ;;
    *)
        echo "unsupported WORKFLOW05_LIVE_UTILIZATION_MODE: ${WORKFLOW05_LIVE_UTILIZATION_MODE}" >&2
        exit 2
        ;;
esac

mkdir -p "${OUTPUT_DIR}/tmp"

source "${ROOT_DIR}/scripts/cscc_gfx936_env.sh"

CSCC_TUNABLEOP_PROFILE_SHA256="${VLLM_ROCM_TUNABLEOP_PROFILE_SHA256:-}"
if [[ -n "${PRA_BACKEND_TUNABLEOP_PROFILE_SHA256_OVERRIDE}" ]]; then
    export VLLM_ROCM_TUNABLEOP_PROFILE_SHA256="${PRA_BACKEND_TUNABLEOP_PROFILE_SHA256_OVERRIDE}"
fi
if [[ "${VLLM_ROCM_TUNABLEOP_PROFILE_SHA256:-}" != "${CURRENT_TUNABLEOP_PROFILE_SHA256}" ]]; then
    echo "active TunableOp profile SHA-256 does not match the current source file; set PRA_BACKEND_TUNABLEOP_PROFILE_SHA256_OVERRIDE explicitly" >&2
    exit 2
fi

export HIP_VISIBLE_DEVICES="${DCU_DEVICE}"
export CUDA_VISIBLE_DEVICES="${DCU_DEVICE}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PRA_BACKEND_PERF_LAYER_PROFILE=1
export TOKENIZERS_PARALLELISM=false
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

HIPPROF=/opt/dtk/bin/hipprof
HIPPROF_LLVM=/opt/dtk-26.04-DCC2602-0317/dcc/lib
export LD_LIBRARY_PATH="${HIPPROF_LLVM}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

PROFILE_SCRIPT="${ROOT_DIR}/scripts/perf_trace/profile_qwen_same_input_layer.py"
RAW_PREFIX="${OUTPUT_DIR}/${TAG}.hipprof"
RAW_DB="${RAW_PREFIX}.db"
EVENT_JSONL="${OUTPUT_DIR}/${TAG}.layer_events.runtime.jsonl"
RUN_METADATA="${OUTPUT_DIR}/full_request_profile_metadata.json"

{
    echo "root_dir=${ROOT_DIR}"
    echo "runtime_run_root=${RUNTIME_RUN_ROOT}"
    echo "runtime_artifact_root=${RUNTIME_ARTIFACT_ROOT}"
    echo "source_revision=$(git -C "${ROOT_DIR}" rev-parse HEAD)"
    echo "source_status_count=$(git -C "${ROOT_DIR}" status --porcelain | wc -l)"
    echo "config=${CONFIG}"
    echo "tag=${TAG}"
    echo "model_root=$(readlink -f "${MODEL_ROOT}")"
    echo "served_model_name=${SERVED_MODEL_NAME}"
    echo "physical_dcu=${DCU_DEVICE}"
    echo "hip_visible_devices=${HIP_VISIBLE_DEVICES}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "process_profile=${PRA_BACKEND_PERF_PROCESS_PROFILE}"
    echo "tunableop_profile_path=${TUNABLEOP_PROFILE_PATH}"
    echo "tunableop_profile_current_sha256=${CURRENT_TUNABLEOP_PROFILE_SHA256}"
    echo "tunableop_profile_cscc_declared_sha256=${CSCC_TUNABLEOP_PROFILE_SHA256}"
    echo "tunableop_profile_override_sha256=${PRA_BACKEND_TUNABLEOP_PROFILE_SHA256_OVERRIDE}"
    echo "tunableop_profile_active_sha256=${VLLM_ROCM_TUNABLEOP_PROFILE_SHA256}"
    echo "process_targets=${PRA_BACKEND_PERF_PROCESS_TARGETS}"
    echo "process_targets_file=${PRA_BACKEND_PERF_PROCESS_TARGETS_FILE}"
    if [[ -n "${PRA_BACKEND_PERF_PROCESS_TARGETS_FILE}" ]]; then
        echo "process_targets_file_sha256=$(sha256sum "${PRA_BACKEND_PERF_PROCESS_TARGETS_FILE}" | awk '{print $1}')"
        echo "process_targets_file_count=$(grep -cve '^[[:space:]]*$' "${PRA_BACKEND_PERF_PROCESS_TARGETS_FILE}")"
    fi
    echo "exact_process_range_filter=required"
    echo "exact_process_range_targets=${PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS}"
    echo "exact_process_range_targets_file=${PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS_FILE}"
    if [[ -n "${PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS_FILE}" ]]; then
        echo "exact_process_range_targets_file_sha256=$(sha256sum "${PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS_FILE}" | awk '{print $1}')"
        echo "exact_process_range_targets_file_count=$(grep -cve '^[[:space:]]*$' "${PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS_FILE}")"
    fi
    echo "live_utilization_mode=${WORKFLOW05_LIVE_UTILIZATION_MODE}"
    echo "live_utilization_collector=${PRA_BACKEND_LIVE_UTIL_COLLECTOR}"
    echo "live_utilization_collector_sha256=${PRA_BACKEND_LIVE_UTIL_COLLECTOR_SHA256}"
    echo "live_utilization_interval_us=${PRA_BACKEND_LIVE_UTIL_INTERVAL_US}"
    echo "layer_profile=${PRA_BACKEND_PERF_LAYER_PROFILE}"
    echo "max_new_tokens=${MAX_NEW_TOKENS}"
    echo "warmup_iters=${WARMUP_ITERS}"
    echo "hipprof=$(readlink -f "${HIPPROF}")"
    echo "hipprof_sha256=$(sha256sum "$(readlink -f "${HIPPROF}")" | awk '{print $1}')"
    echo "hipprof_llvm=${HIPPROF_LLVM}"
    echo "profile_script_sha256=$(sha256sum "${PROFILE_SCRIPT}" | awk '{print $1}')"
    echo "launcher_script_sha256=$(sha256sum "$0" | awk '{print $1}')"
    echo "contract_sha256_file=$(sha256sum "${CONTRACT_PATH}" | awk '{print $1}')"
    echo "fresh_run_lineage_manifest=${FRESH_RUN_LINEAGE_MANIFEST}"
    if [[ -n "${FRESH_RUN_LINEAGE_MANIFEST}" ]]; then
        echo "fresh_run_lineage_manifest_sha256=$(sha256sum "${FRESH_RUN_LINEAGE_MANIFEST}" | awk '{print $1}')"
    fi
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"${OUTPUT_DIR}/${TAG}.tool_provenance.txt"

LIVE_UTIL_PID=""
stop_live_utilization_collector() {
    if [[ -n "${LIVE_UTIL_PID}" ]] && kill -0 "${LIVE_UTIL_PID}" 2>/dev/null; then
        if [[ -n "${PRA_BACKEND_LIVE_UTIL_STOP_FILE:-}" ]] &&
            [[ ! -e "${PRA_BACKEND_LIVE_UTIL_STOP_FILE}" ]]; then
            touch "${PRA_BACKEND_LIVE_UTIL_STOP_FILE}"
        fi
        wait "${LIVE_UTIL_PID}" || true
    fi
}
trap stop_live_utilization_collector EXIT

if [[ "${WORKFLOW05_LIVE_UTILIZATION_MODE}" == "rsmi_se_snapshot" ]]; then
    PRA_BACKEND_LIVE_UTIL_READY_FILE="${OUTPUT_DIR}/live_utilization_ready.json"
    PRA_BACKEND_LIVE_UTIL_ARM_FILE="${OUTPUT_DIR}/${TAG}.live_util.arm"
    PRA_BACKEND_LIVE_UTIL_STOP_FILE="${OUTPUT_DIR}/${TAG}.live_util.stop"
    PRA_BACKEND_LIVE_UTIL_SAMPLES_FILE="${OUTPUT_DIR}/live_utilization_samples.jsonl"
    PRA_BACKEND_LIVE_UTIL_SUMMARY_FILE="${OUTPUT_DIR}/live_utilization_summary.json"
    export PRA_BACKEND_LIVE_UTIL_READY_FILE PRA_BACKEND_LIVE_UTIL_ARM_FILE
    export PRA_BACKEND_LIVE_UTIL_STOP_FILE PRA_BACKEND_LIVE_UTIL_SAMPLES_FILE
    export PRA_BACKEND_LIVE_UTIL_SUMMARY_FILE
    python3 "${PRA_BACKEND_LIVE_UTIL_COLLECTOR}" \
        --device "${DCU_DEVICE}" \
        --interval-us "${PRA_BACKEND_LIVE_UTIL_INTERVAL_US}" \
        --output-jsonl "${PRA_BACKEND_LIVE_UTIL_SAMPLES_FILE}" \
        --summary-json "${PRA_BACKEND_LIVE_UTIL_SUMMARY_FILE}" \
        --ready-file "${PRA_BACKEND_LIVE_UTIL_READY_FILE}" \
        --arm-file "${PRA_BACKEND_LIVE_UTIL_ARM_FILE}" \
        --stop-file "${PRA_BACKEND_LIVE_UTIL_STOP_FILE}" \
        >"${OUTPUT_DIR}/${TAG}.live_util.stdout.log" \
        2>"${OUTPUT_DIR}/${TAG}.live_util.stderr.log" &
    LIVE_UTIL_PID=$!
    for _attempt in $(seq 1 6000); do
        if [[ -s "${PRA_BACKEND_LIVE_UTIL_READY_FILE}" ]]; then
            break
        fi
        if ! kill -0 "${LIVE_UTIL_PID}" 2>/dev/null; then
            echo "live-utilization collector exited before ready" >&2
            wait "${LIVE_UTIL_PID}"
            exit 1
        fi
        sleep 0.01
    done
    if [[ ! -s "${PRA_BACKEND_LIVE_UTIL_READY_FILE}" ]]; then
        echo "live-utilization collector did not become ready" >&2
        exit 1
    fi
fi

DEVICE_PREFLIGHT="${OUTPUT_DIR}/device_preflight.json"
python3 - "${CONTRACT_PATH}" "${DEVICE_PREFLIGHT}" <<'PY'
import json
import subprocess
import sys
import time
from pathlib import Path

contract_path = Path(sys.argv[1]).resolve()
output_path = Path(sys.argv[2]).resolve()
if output_path.exists():
    raise SystemExit(f"refusing existing device preflight: {output_path}")
command = [
    "/opt/hyhal/bin/hy-smi",
    "-d",
    "1",
    "--showuniqueid",
    "--showproductname",
    "--showserial",
    "--showuse",
    "--showmemuse",
    "--json",
]
start_realtime_ns = time.time_ns()
start_monotonic_ns = time.monotonic_ns()
result = subprocess.run(
    command,
    check=True,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
end_monotonic_ns = time.monotonic_ns()
end_realtime_ns = time.time_ns()
observed = json.loads(result.stdout)
card = observed.get("card1")
if not isinstance(card, dict):
    raise SystemExit("hy-smi did not return physical card1")
contract = json.loads(contract_path.read_text(encoding="utf-8"))
device = contract.get("device", {})
if device.get("physical_device_id") != 1:
    raise SystemExit("measurement contract does not bind physical DCU 1")
if card.get("Unique ID") != device.get("unique_id"):
    raise SystemExit("physical DCU 1 Unique ID differs from the contract")
if card.get("Card Series") != device.get("device_name"):
    raise SystemExit("physical DCU 1 product differs from the contract")
hcu_use_pct = float(card["HCU use (%)"])
memory_use_pct = int(card["HCU memory use (%)"])
if hcu_use_pct != 0.0 or memory_use_pct != 0:
    raise SystemExit(
        "concurrent physical DCU 1 work detected: "
        f"HCU={hcu_use_pct}%, memory={memory_use_pct}%"
    )
payload = {
    "schema_version": 1,
    "status": "PASS",
    "physical_device_id": 1,
    "expected_unique_id": device.get("unique_id"),
    "expected_device_name": device.get("device_name"),
    "observed": card,
    "hcu_use_pct": hcu_use_pct,
    "memory_use_pct": memory_use_pct,
    "concurrent_gpu_work_detected": False,
    "observed_interval": {
        "start_realtime_ns": start_realtime_ns,
        "end_realtime_ns": end_realtime_ns,
        "start_monotonic_ns": start_monotonic_ns,
        "end_monotonic_ns": end_monotonic_ns,
    },
    "command": command,
}
output_path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

set -o pipefail
"${HIPPROF}" \
    --hiptx-trace \
    --hip-trace \
    --trace-args \
    --output-type 0 \
    --show-pid \
    --exit-cleanup \
    --flush-interval 1000 \
    --buffer-size 50000 \
    -d "${OUTPUT_DIR}/tmp" \
    -o "${RAW_PREFIX}" \
    python3 "${PROFILE_SCRIPT}" \
        --root-dir "${ROOT_DIR}" \
        --model-root "${MODEL_ROOT}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --dataset /home/testdata/16-32K_throughput.jsonl \
        --dataset-row 0 \
        --contract "${CONTRACT_PATH}" \
        "${LINEAGE_ARGS[@]}" \
        --tag "${TAG}" \
        --output-dir "${OUTPUT_DIR}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --warmup-iters "${WARMUP_ITERS}" \
        --expected-layers 64 \
    2>&1 | tee "${OUTPUT_DIR}/${TAG}.hipprof.log"

if [[ -n "${LIVE_UTIL_PID}" ]]; then
    wait "${LIVE_UTIL_PID}"
    LIVE_UTIL_PID=""
    python3 - "${PRA_BACKEND_LIVE_UTIL_SUMMARY_FILE}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("status") != "complete":
    raise SystemExit(f"live-utilization collector status is {summary.get('status')}")
if int(summary.get("successful_sample_count", 0)) < 3:
    raise SystemExit("live-utilization collector produced fewer than three samples")
PY
fi

test -s "${RUN_METADATA}"
test -s "${EVENT_JSONL}"
test -s "${RAW_DB}"

{
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "exit_code=0"
} >>"${OUTPUT_DIR}/${TAG}.tool_provenance.txt"
