"""Bootstrap selected-layer FX tracing for vLLM.

This module is imported automatically when this directory is on PYTHONPATH.
Instrumentation is opt-in via VLLM_SELECTED_LAYER_FX_ENABLE=1.
"""

import os
import sys


if os.environ.get("VLLM_SELECTED_LAYER_FX_ENABLE") == "1":
    try:
        import vllm_selected_layer_fx_patch

        vllm_selected_layer_fx_patch.apply_patches()
    except BaseException as exc:  # pragma: no cover - tracing must not break boot.
        sys.stderr.write(f"[vllm-selected-layer-fx] failed to apply patches: {exc!r}\n")
