#!/usr/bin/env bash
# Required environment for the frozen gfx936 Qwen3.5 BF16 M=4096 profile.
# The loader accepts one TP1 replica or exactly two local TP1 DP replicas.

export VLLM_ROCM_TUNABLEOP_PROFILE=gfx936_qwen3_5_27b_bf16_tn_m4096
export VLLM_ROCM_TUNABLEOP_PROFILE_SHA256=169c7b11a0340d9e22405327b5e5667b2aa9e9e8d899bd59e10ca4fb7fb52030
profile_path="$(python3 - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("vllm")
assert spec is not None and spec.origin is not None
print(Path(spec.origin).parent / "platforms/tunable_profiles/"
      "gfx936_qwen3_5_27b_bf16_tn_m4096.csv")
PY
)"
if [[ "$(sha256sum "$profile_path" | cut -d' ' -f1)" != \
      "$VLLM_ROCM_TUNABLEOP_PROFILE_SHA256" ]]; then
    echo "error: invalid gfx936 TunableOp profile: $profile_path" >&2
    return 2 2>/dev/null || exit 2
fi
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=0
export PYTORCH_TUNABLEOP_RECORD_UNTUNED=0
export PYTORCH_TUNABLEOP_ROCBLAS_ENABLED=1
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
export PYTORCH_TUNABLEOP_FILENAME="$profile_path"

unset PYTORCH_TUNABLEOP_VERBOSE
unset PYTORCH_TUNABLEOP_VEROBSE
