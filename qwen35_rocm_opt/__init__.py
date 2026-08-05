# SPDX-License-Identifier: Apache-2.0
"""Portable Qwen3.5 ROCm optimization kernels.

The core package intentionally has no vLLM or SGLang imports. Framework
adapters pass host-owned kernels, metadata, and buffers through the public
functions in this package.
"""

__version__ = "0.1.0"
