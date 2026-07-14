"""Optional vLLM trace patch bootstrap.

This module is imported automatically by Python when this directory is on
PYTHONPATH. It only enables instrumentation when VLLM_TRACE_PATCH_ENABLE=1.
"""

import os
import sys


if os.environ.get("VLLM_TRACE_PATCH_ENABLE") == "1":
    try:
        import vllm_trace_patch

        vllm_trace_patch.apply_patches()
    except BaseException as exc:  # pragma: no cover - must not break vLLM boot.
        sys.stderr.write(f"[vllm-trace-patch] failed to apply patches: {exc!r}\n")
