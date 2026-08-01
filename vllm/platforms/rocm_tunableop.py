# SPDX-License-Identifier: Apache-2.0
"""Load the frozen gfx936 Qwen3.5 TunableOp profile."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Final

from vllm.logger import init_logger

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig

logger = init_logger(__name__)

_PROFILE_ENV: Final = "VLLM_ROCM_TUNABLEOP_PROFILE"
_PROFILE_SHA_ENV: Final = "VLLM_ROCM_TUNABLEOP_PROFILE_SHA256"
_PROFILE_NAME: Final = "gfx936_qwen3_5_27b_bf16_tn_m4096"
_PROFILE_FILENAME: Final = f"{_PROFILE_NAME}.csv"
_PROFILE_SHA256: Final = (
    "169c7b11a0340d9e22405327b5e5667b2aa9e9e8d899bd59e10ca4fb7fb52030"
)
_VALIDATORS_SHA256: Final = (
    "138092e6502e24f4aa30aae012da43451501521e489d9f754ea2df471d30af68"
)
_OPERATOR: Final = "GemmTunableOp_BFloat16_TN"
_EXPECTED_VALIDATORS: Final = {
    "PT_VERSION": "2.10.0",
    "HIP_VERSION": "603",
    "GCN_ARCH_NAME": "gfx936:sramecc+:xnack-",
    "ROCBLAS_VERSION": "4.3.0.ef408db7-dirty",
    "HIPBLASLT_VERSION": "1000-a6254b89-dirty",
}
_EXPECTED_SOLUTIONS: Final = {
    "tn_14336_4096_5120_ld_5120_5120_14336": "Gemm_Rocblas_20981",
    "tn_16384_4096_5120_ld_5120_5120_16384": "Gemm_Rocblas_20981",
    "tn_34816_4096_5120_ld_5120_5120_34816": "Gemm_Rocblas_20981",
    "tn_5120_4096_17408_ld_17408_17408_5120": "Gemm_Rocblas_20979",
    "tn_5120_4096_6144_ld_6144_6144_5120": "Gemm_Rocblas_20981",
}
_EXPECTED_RESULTS: Final = frozenset(
    (_OPERATOR, params, solution, 0.0)
    for params, solution in _EXPECTED_SOLUTIONS.items()
)


def _fail(message: str) -> RuntimeError:
    return RuntimeError(f"ROCm TunableOp profile rejected: {message}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile_path() -> Path:
    profile_dir = Path(__file__).with_name("tunable_profiles").resolve(strict=True)
    candidate = profile_dir / _PROFILE_FILENAME
    if candidate.is_symlink():
        raise _fail(f"profile path must not be a symlink: {candidate}")
    path = candidate.resolve(strict=True)
    if path.parent != profile_dir or not path.is_file():
        raise _fail(f"profile is not a regular wheel-internal file: {path}")
    return path


def _validate_scope(vllm_config: VllmConfig) -> None:
    import torch

    model = vllm_config.model_config
    scheduler = vllm_config.scheduler_config
    parallel = vllm_config.parallel_config
    actual = (
        tuple(model.architectures),
        model.dtype,
        scheduler.max_num_batched_tokens,
        parallel.tensor_parallel_size,
        parallel.pipeline_parallel_size,
        parallel.prefill_context_parallel_size,
        parallel.data_parallel_size,
    )
    expected = (("Qwen3_5ForConditionalGeneration",), torch.bfloat16, 4096, 1, 1, 1, 1)
    if actual != expected:
        raise _fail(f"workload scope mismatch: actual={actual!r}")


def _validate_environment(path: Path) -> None:
    expected_sha256 = os.environ.get(_PROFILE_SHA_ENV, "").lower()
    if expected_sha256 != _PROFILE_SHA256:
        raise _fail(
            f"{_PROFILE_SHA_ENV} mismatch: "
            f"expected={_PROFILE_SHA256}, actual={expected_sha256!r}"
        )

    allowed: dict[str, set[str | None]] = {
        "PYTORCH_TUNABLEOP_ENABLED": {None, "1"},
        "PYTORCH_TUNABLEOP_TUNING": {None, "0"},
        "PYTORCH_TUNABLEOP_RECORD_UNTUNED": {None, "0"},
        "PYTORCH_TUNABLEOP_FILENAME": {None, str(path)},
        "PYTORCH_TUNABLEOP_ROCBLAS_ENABLED": {None, "1"},
        "PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED": {"0"},
    }
    conflicts = {
        key: os.environ.get(key)
        for key, values in allowed.items()
        if os.environ.get(key) not in values
    }
    if "PYTORCH_TUNABLEOP_VEROBSE" in os.environ:
        conflicts["PYTORCH_TUNABLEOP_VEROBSE"] = os.environ[
            "PYTORCH_TUNABLEOP_VEROBSE"
        ]
    if conflicts:
        raise _fail(f"conflicting/cached environment: {conflicts!r}")


def _normalize_results(raw_results: object) -> tuple[tuple[str, str, str, float], ...]:
    try:
        return tuple(
            (str(operator), str(params), str(solution), float(elapsed_ms))
            for operator, params, solution, elapsed_ms in (
                raw_results  # type: ignore[union-attr]
            )
        )
    except (TypeError, ValueError) as exc:
        raise _fail(f"unexpected get_results payload: {raw_results!r}") from exc


def _assert_loaded_profile(path: Path) -> None:
    from torch.cuda import tunable

    results = _normalize_results(tunable.get_results())
    if len(results) != len(_EXPECTED_RESULTS) or set(results) != _EXPECTED_RESULTS:
        raise _fail(f"runtime result mismatch: {results!r}")
    validators = dict(tunable.get_validators())
    if validators != _EXPECTED_VALIDATORS:
        raise _fail(f"runtime validator mismatch: {validators!r}")
    if _sha256(path) != _PROFILE_SHA256:
        raise _fail("wheel-internal profile changed after initialization")


def _assert_api_state(path: Path) -> None:
    from torch.cuda import tunable

    if not tunable.is_enabled():
        raise _fail("TunableOp unexpectedly disabled")
    if tunable.tuning_is_enabled():
        raise _fail("online tuning unexpectedly enabled")
    if tunable.record_untuned_is_enabled():
        raise _fail("record-untuned unexpectedly enabled")
    _assert_loaded_profile(path)


def maybe_init_rocm_tunableop(
    vllm_config: VllmConfig, device: torch.device
) -> Path | None:
    """Initialize after device/distributed setup and before memory snapshot."""
    profile = os.environ.get(_PROFILE_ENV, "")
    if not profile:
        logger.info("VLLM_ROCM_TUNABLEOP_INIT status=disabled reason=opt_in_unset")
        return None

    try:
        if profile != _PROFILE_NAME:
            raise _fail(f"unknown profile name: {profile!r}")
        _validate_scope(vllm_config)
        path = _profile_path()
        _validate_environment(path)
        actual_sha256 = _sha256(path)
        if actual_sha256 != _PROFILE_SHA256:
            raise _fail(
                f"profile SHA256 mismatch: expected={_PROFILE_SHA256}, "
                f"actual={actual_sha256}"
            )

        from torch.cuda import tunable

        tunable.tuning_enable(False)
        tunable.record_untuned_enable(False)
        tunable.set_filename(str(path), insert_device_ordinal=False)
        _assert_loaded_profile(path)
        tunable.enable(True)

        _assert_api_state(path)
        logger.info(
            "VLLM_ROCM_TUNABLEOP_INIT status=ready device=%s file=%s "
            "file_sha256=%s validators_sha256=%s rows=5 logical_families=6 "
            "explicit_families=5 shared_explicit_families=2 enabled=1 tuning=0 "
            "record_untuned=0",
            device,
            path,
            _PROFILE_SHA256,
            _VALIDATORS_SHA256,
        )
        return path
    except Exception as exc:
        logger.exception(
            "VLLM_ROCM_TUNABLEOP_INIT status=error profile=%s reason=%s",
            profile,
            exc,
        )
        raise


def assert_rocm_tunableop_pre_capture(path: Path) -> None:
    """Fail closed immediately after kernel warmup and before graph capture."""
    _assert_api_state(path)
    logger.info(
        "VLLM_ROCM_TUNABLEOP_PRE_CAPTURE status=ready profile=%s rows=5 "
        "expected_hit_keys=5 shared_explicit_families=2 "
        "observed_dispatch=external_verbose_canary_required",
        _PROFILE_NAME,
    )
