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
: "${PRA_BACKEND_PERF_PROCESS_TARGETS:?set PRA_BACKEND_PERF_PROCESS_TARGETS}"

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
if [[ -z "${PRA_BACKEND_PERF_PROCESS_TARGETS}" ]]; then
    echo "process target list must not be empty" >&2
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
OUTPUT_PARENT="$(readlink -f "$(dirname "${OUTPUT_DIR}")")"
OUTPUT_DIR="${OUTPUT_PARENT}/$(basename "${OUTPUT_DIR}")"

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

mkdir -p "${OUTPUT_DIR}/tmp"

source "${ROOT_DIR}/scripts/cscc_gfx936_env.sh"

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
RUN_METADATA="${OUTPUT_DIR}/${TAG}.json"

{
    echo "root_dir=${ROOT_DIR}"
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
    echo "process_targets=${PRA_BACKEND_PERF_PROCESS_TARGETS}"
    echo "layer_profile=${PRA_BACKEND_PERF_LAYER_PROFILE}"
    echo "max_new_tokens=${MAX_NEW_TOKENS}"
    echo "warmup_iters=${WARMUP_ITERS}"
    echo "hipprof=$(readlink -f "${HIPPROF}")"
    echo "hipprof_sha256=$(sha256sum "$(readlink -f "${HIPPROF}")" | awk '{print $1}')"
    echo "hipprof_llvm=${HIPPROF_LLVM}"
    echo "profile_script_sha256=$(sha256sum "${PROFILE_SCRIPT}" | awk '{print $1}')"
    echo "launcher_script_sha256=$(sha256sum "$0" | awk '{print $1}')"
    echo "contract_sha256_file=$(sha256sum "${CONTRACT_PATH}" | awk '{print $1}')"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"${OUTPUT_DIR}/${TAG}.tool_provenance.txt"

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
        --tag "${TAG}" \
        --output-dir "${OUTPUT_DIR}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --warmup-iters "${WARMUP_ITERS}" \
        --expected-layers 64 \
    2>&1 | tee "${OUTPUT_DIR}/${TAG}.hipprof.log"

test -s "${RUN_METADATA}"
test -s "${EVENT_JSONL}"
test -s "${RAW_DB}"

{
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "exit_code=0"
} >>"${OUTPUT_DIR}/${TAG}.tool_provenance.txt"
