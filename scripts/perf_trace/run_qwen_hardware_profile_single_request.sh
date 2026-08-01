#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 process-trace|pmc|pmc-read|pmc-write OUTPUT_DIR TAG" >&2
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ "$#" -ne 3 ]]; then
    usage
    exit 2
fi

PROFILE_KIND="$1"
OUTPUT_DIR="$2"
TAG="$3"

: "${ROOT_DIR:?set ROOT_DIR}"
: "${RUNTIME_ARTIFACT_ROOT:?set RUNTIME_ARTIFACT_ROOT}"
: "${CONTRACT_PATH:?set CONTRACT_PATH}"
: "${MODEL_ROOT:?set MODEL_ROOT}"
: "${SERVED_MODEL_NAME:?set SERVED_MODEL_NAME}"
: "${PRA_BACKEND_PERF_PROCESS_TARGETS:?set current representative event ids}"
: "${DCU_DEVICE:?set DCU_DEVICE}"

case "${PROFILE_KIND}" in
    process-trace|pmc|pmc-read|pmc-write) ;;
    *)
        echo "invalid profile kind: ${PROFILE_KIND}" >&2
        exit 2
        ;;
esac
if [[ "${DCU_DEVICE}" != "1" ]]; then
    echo "reviewed hardware replay policy requires physical DCU 1" >&2
    exit 2
fi
if [[ ! "${TAG}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "TAG contains unsupported characters" >&2
    exit 2
fi

ROOT_DIR="$(readlink -f "${ROOT_DIR}")"
RUNTIME_ARTIFACT_ROOT="$(readlink -f "${RUNTIME_ARTIFACT_ROOT}")"
OUTPUT_PARENT="$(readlink -f "$(dirname "${OUTPUT_DIR}")")"
OUTPUT_DIR="${OUTPUT_PARENT}/$(basename "${OUTPUT_DIR}")"
CONTRACT_PATH="$(readlink -f "${CONTRACT_PATH}")"
MODEL_ROOT="$(readlink -f "${MODEL_ROOT}")"

case "${OUTPUT_DIR}/" in
    "${RUNTIME_ARTIFACT_ROOT}/"*) ;;
    *)
        echo "OUTPUT_DIR must remain under RUNTIME_ARTIFACT_ROOT" >&2
        exit 2
        ;;
esac
if [[ "${OUTPUT_DIR}" == *"/perf_trace_bk/"* ]]; then
    echo "archived perf_trace_bk is not a live output root" >&2
    exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]] &&
    [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "refusing to reuse non-empty OUTPUT_DIR: ${OUTPUT_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CONTRACT_PATH}" ]]; then
    echo "missing frozen SAME_INPUT contract: ${CONTRACT_PATH}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}/raw" "${OUTPUT_DIR}/tmp"
source "${ROOT_DIR}/scripts/cscc_gfx936_env.sh"

export HIP_VISIBLE_DEVICES="${DCU_DEVICE}"
export CUDA_VISIBLE_DEVICES="${DCU_DEVICE}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PRA_BACKEND_PERF_LAYER_PROFILE=1
export PRA_BACKEND_PERF_PROCESS_PROFILE=1
export TOKENIZERS_PARALLELISM=false
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

HIPPROF=/opt/dtk/bin/hipprof
HIPPROF="$(readlink -f "${HIPPROF}")"
HIPPROF_LLVM=/opt/dtk-26.04-DCC2602-0317/dcc/lib
export LD_LIBRARY_PATH="${HIPPROF_LLVM}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

PROFILE_SCRIPT="${ROOT_DIR}/scripts/perf_trace/profile_qwen_same_input_layer.py"
SESSION_NAME="qwen_r04_${PROFILE_KIND//-/_}_$$_$(date -u +%s)"
export PRA_HIPPROF_SESSION_NAME="${SESSION_NAME}"
export PRA_HIPPROF_PROFILE_KIND="${PROFILE_KIND}"
export PRA_HIPPROF_BIN="${HIPPROF}"
export PRA_HIPPROF_SESSION_REQUIRED=1

HIPPROF_ARGS=(
    --session "${SESSION_NAME}"
    --trace-off
    --hiptx-trace
    --hip-trace
    --trace-args
    --follow-fork
    --exit-cleanup
    --flush-interval 1000
    --buffer-size 50000
    --show-pid
    --no-export
)
case "${PROFILE_KIND}" in
    process-trace) ;;
    pmc) HIPPROF_ARGS+=(--pmc --pmc-type 0 --pmc-off) ;;
    pmc-read) HIPPROF_ARGS+=(--pmc-read --pmc-type 0 --pmc-off) ;;
    pmc-write) HIPPROF_ARGS+=(--pmc-write --pmc-type 0 --pmc-off) ;;
esac

RAW_PREFIX="${OUTPUT_DIR}/raw/${TAG}"
EVENT_JSONL="${OUTPUT_DIR}/${TAG}.layer_events.runtime.jsonl"
RUN_METADATA="${OUTPUT_DIR}/${TAG}.json"
PROVENANCE="${OUTPUT_DIR}/tool_provenance.txt"

/opt/hyhal/bin/hy-smi \
    -d "${DCU_DEVICE}" \
    --showuniqueid \
    --showproductname \
    --showserial \
    --showuse \
    --showmemuse \
    --json >"${OUTPUT_DIR}/device_preflight.json"

{
    echo "profile_kind=${PROFILE_KIND}"
    echo "session_name=${SESSION_NAME}"
    echo "root_dir=${ROOT_DIR}"
    echo "source_revision=$(git -C "${ROOT_DIR}" rev-parse HEAD)"
    echo "source_branch=$(git -C "${ROOT_DIR}" branch --show-current)"
    echo "source_status_count=$(git -C "${ROOT_DIR}" status --porcelain | wc -l)"
    echo "contract_path=${CONTRACT_PATH}"
    echo "contract_file_sha256=$(sha256sum "${CONTRACT_PATH}" | awk '{print $1}')"
    echo "model_root=${MODEL_ROOT}"
    echo "served_model_name=${SERVED_MODEL_NAME}"
    echo "physical_dcu=${DCU_DEVICE}"
    echo "logical_dcu=0"
    echo "hip_visible_devices=${HIP_VISIBLE_DEVICES}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "process_profile=${PRA_BACKEND_PERF_PROCESS_PROFILE}"
    echo "process_targets=${PRA_BACKEND_PERF_PROCESS_TARGETS}"
    echo "layer_profile=${PRA_BACKEND_PERF_LAYER_PROFILE}"
    echo "max_new_tokens=32"
    echo "warmup_iters=1"
    echo "pmc_type=0"
    echo "hipprof_device_filter=none"
    echo "hipprof=${HIPPROF}"
    echo "hipprof_sha256=$(sha256sum "${HIPPROF}" | awk '{print $1}')"
    echo "hipprof_llvm=${HIPPROF_LLVM}"
    echo "hipprof_args=${HIPPROF_ARGS[*]}"
    echo "profile_script=${PROFILE_SCRIPT}"
    echo "profile_script_sha256=$(sha256sum "${PROFILE_SCRIPT}" | awk '{print $1}')"
    echo "runner_script_sha256=$(sha256sum "$0" | awk '{print $1}')"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"${PROVENANCE}"

set +e
set -o pipefail
"${HIPPROF}" "${HIPPROF_ARGS[@]}" \
    -d "${OUTPUT_DIR}/tmp" \
    -o "${RAW_PREFIX}" \
    python3 "${PROFILE_SCRIPT}" \
        --root-dir "${ROOT_DIR}" \
        --model-root "${MODEL_ROOT}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --dataset /home/testdata/16-32K_throughput.jsonl \
        --dataset-row 0 \
        --contract "${CONTRACT_PATH}" \
        --tag "${TAG}" \
        --output-dir "${OUTPUT_DIR}" \
        --max-new-tokens 32 \
        --warmup-iters 1 \
        --expected-layers 64 \
    2>&1 | tee "${OUTPUT_DIR}/hipprof.log"
PROFILE_RC="${PIPESTATUS[0]}"
set -e

{
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "exit_code=${PROFILE_RC}"
} >>"${PROVENANCE}"
echo "${PROFILE_RC}" >"${OUTPUT_DIR}/profile.exit_code"
if [[ "${PROFILE_RC}" -ne 0 ]]; then
    exit "${PROFILE_RC}"
fi

test -s "${RUN_METADATA}"
test -s "${EVENT_JSONL}"
RAW_DB="$(find "${OUTPUT_DIR}/raw" -maxdepth 1 -type f -name '*.db' -size +0c -print -quit)"
test -n "${RAW_DB}"
if [[ "${PROFILE_KIND}" != "process-trace" ]]; then
    RAW_METRICS="$(find "${OUTPUT_DIR}/raw" -maxdepth 1 -type f -name '*.txt' -size +0c -print -quit)"
    test -n "${RAW_METRICS}"
fi
