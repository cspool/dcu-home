#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="fa718036bdb9dfd80a872b86c8ac16c9d02bfd31"
SNAPSHOT_ROOT="67f44ab405d8efed30a42f04f6e74ae2e8370884"
PROFILE_REL="vllm/platforms/tunable_profiles/gfx936_qwen3_5_27b_bf16_tn_m4096.csv"
PROFILE_SHA256="169c7b11a0340d9e22405327b5e5667b2aa9e9e8d899bd59e10ca4fb7fb52030"
SOURCE_MANIFEST="evidence/manifests/repro_minimal_runtime.sha256"
RUNTIME_CHURN_LIMIT=600
WHEEL="${1:-}"

fail() {
    echo "verify_cscc_repro: ERROR: $*" >&2
    exit 1
}

pass() {
    echo "verify_cscc_repro: OK: $*"
}

cd "$ROOT"

git cat-file -e "$SNAPSHOT_ROOT^{commit}" 2>/dev/null || \
    fail "submission snapshot root is unavailable: $SNAPSHOT_ROOT"
git merge-base --is-ancestor "$SNAPSHOT_ROOT" HEAD || \
    fail "HEAD does not descend from the submission snapshot root"
if git cat-file -e "$BASELINE^{commit}" 2>/dev/null; then
    pass "official OpenDAS comparison object and submission ancestry"
else
    echo "verify_cscc_repro: NOTE: OpenDAS comparison object is absent; " \
        "tree validation uses the submission history only"
fi

required_files=(
    "$SOURCE_MANIFEST"
    scripts/bench_cscc_multi_request.sh
    scripts/build_cscc_wheel.sh
    scripts/cscc_gfx936_env.sh
    scripts/opencompass_accuracy_entry.py
    scripts/run_cscc_tp2_batch10_e2e.sh
    scripts/serve_cscc_dp2.sh
    scripts/serve_cscc_tp2_batch10.sh
    scripts/setup_cscc_tp2_batch10.sh
    "$PROFILE_REL"
    vllm/model_executor/layers/fla/ops/gfx936.py
    vllm/model_executor/models/qwen3_next.py
    vllm/platforms/rocm.py
    vllm/v1/attention/backends/rocm_aiter_unified_attn.py
    vllm/v1/attention/ops/rocm_aiter_unified_attention_gqa6.py
    vllm/v1/worker/gpu_model_runner.py
)
for path in "${required_files[@]}"; do
    git ls-files --error-unmatch "$path" >/dev/null 2>&1 || \
        fail "required tracked file is missing: $path"
done
pass "required optimization files are tracked"

if git cat-file -e "$BASELINE^{commit}" 2>/dev/null; then
    read -r runtime_additions runtime_deletions < <(
        git diff --numstat "$BASELINE" -- csrc setup.py vllm |
            awk '{ additions += $1; deletions += $2 }
                 END { print additions + 0, deletions + 0 }'
    )
    runtime_churn=$((runtime_additions + runtime_deletions))
    (( runtime_churn <= RUNTIME_CHURN_LIMIT )) || fail \
        "runtime source churn ${runtime_churn} exceeds ${RUNTIME_CHURN_LIMIT} (${runtime_additions} additions + ${runtime_deletions} deletions)"
    pass "runtime source churn ${runtime_churn}/${RUNTIME_CHURN_LIMIT} (${runtime_additions} additions + ${runtime_deletions} deletions)"
    while read -r path; do
        awk -v path="$path" '$2 == path { found = 1 } END { exit !found }' \
            "$SOURCE_MANIFEST" || fail "runtime file is absent from manifest: $path"
    done < <(git diff --name-only "$BASELINE" -- csrc setup.py vllm)
    pass "runtime source manifest coverage"
fi

sha256sum -c "$SOURCE_MANIFEST" >/dev/null || \
    fail "runtime source manifest mismatch"
pass "runtime source manifest"

actual_profile_sha="$(sha256sum "$PROFILE_REL" | awk '{print $1}')"
[[ "$actual_profile_sha" == "$PROFILE_SHA256" ]] || \
    fail "TunableOp profile SHA256 mismatch: $actual_profile_sha"

python3 - "$PROFILE_REL" <<'PY'
import csv
import sys
from pathlib import Path

rows = list(csv.reader(Path(sys.argv[1]).open(encoding="utf-8")))
validators = [row for row in rows if row and row[0] == "Validator"]
results = [row for row in rows if row and row[0] != "Validator"]
if len(validators) != 5 or len(results) != 5:
    raise SystemExit(
        f"expected 5 validators and 5 results, got {len(validators)} and "
        f"{len(results)}"
    )
if any(len(row) != 4 or row[0] != "GemmTunableOp_BFloat16_TN" for row in results):
    raise SystemExit("unexpected TunableOp result schema")
PY
pass "frozen TunableOp profile"

required_patterns=(
    "speculative_config is None"
    "fixed_gdn"
    "qwen35_packed_decode"
    "_mrope_scratch"
    "tensor.shape[1] == 10240"
    "dual_card_topology"
    "max_cudagraph_capture_size = 16"
)
for pattern in "${required_patterns[@]}"; do
    git grep -q -F "$pattern" -- csrc vllm || \
        fail "required source marker is missing: $pattern"
done

forbidden_patterns=(
    "LLMM1StridedSilu"
    "LLMM1Strided"
    "LLGemm1_strided_kernel"
    "rocm_gateup_swiglu"
    "VLLM_CSCC_DISABLE_GATEUP_SWIGLU_FUSION"
    "CSCC_DISABLE_DECODE_OUTPUT_GEMV_K17408"
    "LLGemm1_qwen35_output_kernel"
)
for pattern in "${forbidden_patterns[@]}"; do
    if git grep -q -F "$pattern" -- csrc vllm; then
        fail "rejected or stale experiment marker remains: $pattern"
    fi
done
pass "required paths present and rejected experiments absent"

git grep -q -F -- "--tensor-parallel-size 2" scripts/serve_cscc_tp2_batch10.sh || \
    fail "TP=2 launcher does not request tensor parallelism"
git grep -q -F -- "--max-model-len 32768" scripts/serve_cscc_tp2_batch10.sh || \
    fail "TP=2 launcher does not enforce the downloaded evaluator context limit"
if git grep -q -E -- "data-parallel|speculative|prefix.cach" \
    scripts/serve_cscc_tp2_batch10.sh; then
    fail "TP=2 launcher enables a prohibited serving mode"
fi
pass "TP=2 launcher topology and prohibited-mode checks"

VERIFY_TMP="$(mktemp -d "${TMPDIR:-/tmp}/vllm-cscc-verify.XXXXXX")"
cleanup() {
    rm -rf -- "$VERIFY_TMP"
}
trap cleanup EXIT

PYTHONPYCACHEPREFIX="$VERIFY_TMP/pycache" python3 -m py_compile \
    scripts/opencompass_accuracy_entry.py \
    vllm/_custom_ops.py \
    vllm/config/parallel.py \
    vllm/model_executor/layers/fla/ops/chunk.py \
    vllm/model_executor/layers/fla/ops/chunk_o.py \
    vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py \
    vllm/model_executor/layers/fla/ops/fused_recurrent.py \
    vllm/model_executor/layers/fla/ops/gfx936.py \
    vllm/model_executor/layers/fla/ops/solve_tril.py \
    vllm/model_executor/layers/fla/ops/utils.py \
    vllm/model_executor/layers/fla/ops/wy_fast.py \
    vllm/model_executor/layers/utils.py \
    vllm/model_executor/models/qwen3_5.py \
    vllm/model_executor/models/qwen3_next.py \
    vllm/platforms/rocm.py \
    vllm/v1/attention/backends/gdn_attn.py \
    vllm/v1/attention/backends/rocm_aiter_unified_attn.py \
    vllm/v1/attention/ops/rocm_aiter_unified_attention_gqa6.py \
    vllm/v1/engine/core.py \
    vllm/v1/worker/gpu_model_runner.py \
    vllm/v1/worker/gpu_worker.py
bash -n \
    scripts/bench_cscc_multi_request.sh \
    scripts/build_cscc_wheel.sh \
    scripts/cscc_gfx936_env.sh \
    scripts/run_cscc_tp2_batch10_e2e.sh \
    scripts/serve_cscc_dp2.sh \
    scripts/serve_cscc_tp2_batch10.sh \
    scripts/setup_cscc_tp2_batch10.sh
if git cat-file -e "$BASELINE^{commit}" 2>/dev/null; then
    format_base="$BASELINE"
else
    format_base="$SNAPSHOT_ROOT"
fi
git diff --check "$format_base" -- \
    csrc setup.py vllm scripts README.md BUILD.md ENVIRONMENT.md
pass "Python, shell, and patch-format checks"

if [[ -n "$WHEEL" ]]; then
    [[ -f "$WHEEL" ]] || fail "wheel not found: $WHEEL"
    wheel_files="$VERIFY_TMP/wheel-files.txt"
    unzip -Z1 "$WHEEL" >"$wheel_files"
    grep -q '/_rocm_C[^/]*\.so$' "$wheel_files" || \
        fail "wheel does not contain vllm/_rocm_C"
    grep -q "^vllm/platforms/tunable_profiles/$(basename "$PROFILE_REL")$" \
        "$wheel_files" || fail "wheel does not contain the frozen profile"
    for module in gfx936.py rocm_aiter_unified_attention_gqa6.py; do
        grep -q "/${module}$" "$wheel_files" || \
            fail "wheel does not contain ${module}"
    done
    if grep -Eq '(^|/)(__pycache__/|.*\.pyc$)' "$wheel_files"; then
        fail "wheel contains Python bytecode from a reused build tree"
    fi
    if grep -Eq 'rocm_gateup_swiglu|perf_trace' "$wheel_files"; then
        fail "wheel contains a removed experiment module"
    fi
    pass "clean wheel contents: $(sha256sum "$WHEEL" | awk '{print $1}')"
fi

echo "verify_cscc_repro: PASS"
