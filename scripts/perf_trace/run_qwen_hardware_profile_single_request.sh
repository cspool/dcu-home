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
: "${DCU_DEVICE:?set DCU_DEVICE}"
LINEAGE_MANIFEST="${LINEAGE_MANIFEST:-}"
WORKFLOW05_R08_FRESH_LINEAGE_REQUIRED="${WORKFLOW05_R08_FRESH_LINEAGE_REQUIRED:-0}"

PRA_BACKEND_PERF_PROCESS_TARGETS="${PRA_BACKEND_PERF_PROCESS_TARGETS:-}"
PRA_BACKEND_PERF_PROCESS_TARGETS_FILE="${PRA_BACKEND_PERF_PROCESS_TARGETS_FILE:-}"
PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS="${PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS:-}"
PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS_FILE="${PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS_FILE:-}"
PRA_BACKEND_TUNABLEOP_PROFILE_SHA256_OVERRIDE="${PRA_BACKEND_TUNABLEOP_PROFILE_SHA256_OVERRIDE:-}"
WORKFLOW05_EXACT_PROCESS_FILTER_REQUIRED="${WORKFLOW05_EXACT_PROCESS_FILTER_REQUIRED:-0}"
WORKFLOW05_PMC_COLLECTION_POLICY="${WORKFLOW05_PMC_COLLECTION_POLICY:-complete_request_exact_post_attribution}"
PRA_HIPPROF_KERNEL_NAME_FILTER="${PRA_HIPPROF_KERNEL_NAME_FILTER:-}"
WORKFLOW05_PMC_CAPTURE_BATCH_ID="${WORKFLOW05_PMC_CAPTURE_BATCH_ID:-}"
WORKFLOW05_TARGET_SELECTION_PLAN="${WORKFLOW05_TARGET_SELECTION_PLAN:-}"
if [[ -z "${PRA_BACKEND_PERF_PROCESS_TARGETS}" ]] &&
    [[ -z "${PRA_BACKEND_PERF_PROCESS_TARGETS_FILE}" ]]; then
    echo "process target CSV or newline file must not be empty" >&2
    exit 2
fi
if [[ "${WORKFLOW05_EXACT_PROCESS_FILTER_REQUIRED}" == "1" ]] &&
    [[ -z "${PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS}" ]] &&
    [[ -z "${PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS_FILE}" ]]; then
    echo "Workflow05 hardware replay requires exact process ranges" >&2
    exit 2
fi
if [[ -n "${PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS}" ]]; then
    PROCESS_RANGE_PATTERN='^pra\.fx_process\.input[0-9]+_layer[0-9]+\.[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)?(,pra\.fx_process\.input[0-9]+_layer[0-9]+\.[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)?)*$'
    if [[ ! "${PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS}" =~ ${PROCESS_RANGE_PATTERN} ]]; then
        echo "exact process range target list has invalid syntax" >&2
        exit 2
    fi
fi
case "${WORKFLOW05_PMC_COLLECTION_POLICY}" in
    complete_request_exact_post_attribution)
        if [[ -n "${PRA_HIPPROF_KERNEL_NAME_FILTER}" ]]; then
            echo "complete-request collection must not set a kernel-name filter" >&2
            exit 2
        fi
        ;;
    bounded_family_superset_exact_post_attribution)
        if [[ "${PROFILE_KIND}" == "process-trace" ]]; then
            echo "bounded family-superset policy applies only to PMC modes" >&2
            exit 2
        fi
        if [[ "${WORKFLOW05_EXACT_PROCESS_FILTER_REQUIRED}" != "1" ]]; then
            echo "bounded family-superset collection requires exact process markers" >&2
            exit 2
        fi
        if [[ -z "${PRA_HIPPROF_KERNEL_NAME_FILTER}" ]]; then
            echo "bounded family-superset collection requires one literal kernel-name filter" >&2
            exit 2
        fi
        if [[ -z "${WORKFLOW05_PMC_CAPTURE_BATCH_ID}" ]] ||
            [[ ! "${WORKFLOW05_PMC_CAPTURE_BATCH_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]]; then
            echo "bounded family-superset collection requires a safe capture batch ID" >&2
            exit 2
        fi
        if [[ -z "${WORKFLOW05_TARGET_SELECTION_PLAN}" ]] ||
            [[ ! -f "${WORKFLOW05_TARGET_SELECTION_PLAN}" ]]; then
            echo "bounded family-superset collection requires a target selection plan" >&2
            exit 2
        fi
        ;;
    *)
        echo "invalid Workflow05 PMC collection policy: ${WORKFLOW05_PMC_COLLECTION_POLICY}" >&2
        exit 2
        ;;
esac
if [[ "${PRA_HIPPROF_KERNEL_NAME_FILTER}" == -* ]] ||
    [[ "${PRA_HIPPROF_KERNEL_NAME_FILTER}" == *$'\n'* ]] ||
    [[ "${PRA_HIPPROF_KERNEL_NAME_FILTER}" == *$'\r'* ]]; then
    echo "kernel-name filter contains unsupported characters" >&2
    exit 2
fi
if (( ${#PRA_HIPPROF_KERNEL_NAME_FILTER} > 4096 )); then
    echo "kernel-name filter is too long" >&2
    exit 2
fi
export WORKFLOW05_PMC_COLLECTION_POLICY PRA_HIPPROF_KERNEL_NAME_FILTER
export WORKFLOW05_PMC_CAPTURE_BATCH_ID WORKFLOW05_TARGET_SELECTION_PLAN

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
if [[ "${WORKFLOW05_R08_FRESH_LINEAGE_REQUIRED}" == "1" ]]; then
    if [[ -z "${LINEAGE_MANIFEST}" ]] || [[ ! -f "${LINEAGE_MANIFEST}" ]]; then
        echo "fresh R08 capture requires the same-run lineage manifest" >&2
        exit 2
    fi
    LINEAGE_MANIFEST="$(readlink -f "${LINEAGE_MANIFEST}")"
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
if [[ -n "${WORKFLOW05_TARGET_SELECTION_PLAN}" ]]; then
    WORKFLOW05_TARGET_SELECTION_PLAN="$(readlink -f "${WORKFLOW05_TARGET_SELECTION_PLAN}")"
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
        "${RUNTIME_ARTIFACT_ROOT}"/*) ;;
        *)
            echo "${target_file_variable} must remain under RUNTIME_ARTIFACT_ROOT" >&2
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
if [[ -n "${PRA_HIPPROF_KERNEL_NAME_FILTER}" ]]; then
    HIPPROF_ARGS+=(--kernel-name "${PRA_HIPPROF_KERNEL_NAME_FILTER}")
fi

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
    echo "lineage_manifest=${LINEAGE_MANIFEST}"
    if [[ -n "${LINEAGE_MANIFEST}" ]]; then
        echo "lineage_manifest_sha256=$(sha256sum "${LINEAGE_MANIFEST}" | awk '{print $1}')"
    fi
    echo "workflow05_r08_fresh_lineage_required=${WORKFLOW05_R08_FRESH_LINEAGE_REQUIRED}"
    echo "model_root=${MODEL_ROOT}"
    echo "served_model_name=${SERVED_MODEL_NAME}"
    echo "physical_dcu=${DCU_DEVICE}"
    echo "logical_dcu=0"
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
    echo "workflow05_exact_process_filter_required=${WORKFLOW05_EXACT_PROCESS_FILTER_REQUIRED}"
    echo "exact_process_range_targets=${PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS}"
    echo "exact_process_range_targets_file=${PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS_FILE}"
    if [[ -n "${PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS_FILE}" ]]; then
        echo "exact_process_range_targets_file_sha256=$(sha256sum "${PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS_FILE}" | awk '{print $1}')"
        echo "exact_process_range_targets_file_count=$(grep -cve '^[[:space:]]*$' "${PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS_FILE}")"
    fi
    echo "workflow05_pmc_collection_policy=${WORKFLOW05_PMC_COLLECTION_POLICY}"
    echo "kernel_name_filter=${PRA_HIPPROF_KERNEL_NAME_FILTER}"
    echo "kernel_name_filter_sha256=$(printf '%s' "${PRA_HIPPROF_KERNEL_NAME_FILTER}" | sha256sum | awk '{print $1}')"
    echo "workflow05_pmc_capture_batch_id=${WORKFLOW05_PMC_CAPTURE_BATCH_ID}"
    echo "workflow05_target_selection_plan=${WORKFLOW05_TARGET_SELECTION_PLAN}"
    if [[ -n "${WORKFLOW05_TARGET_SELECTION_PLAN}" ]]; then
        echo "workflow05_target_selection_plan_sha256=$(sha256sum "${WORKFLOW05_TARGET_SELECTION_PLAN}" | awk '{print $1}')"
    fi
    echo "collector_side_process_window_filter=false"
    echo "final_process_attribution_policy=same_replay_strict_hiptx_runtime_hipops"
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

PROFILE_LINEAGE_ARGS=()
if [[ -n "${LINEAGE_MANIFEST}" ]]; then
    PROFILE_LINEAGE_ARGS+=(--lineage-manifest "${LINEAGE_MANIFEST}")
fi

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
        "${PROFILE_LINEAGE_ARGS[@]}" \
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
