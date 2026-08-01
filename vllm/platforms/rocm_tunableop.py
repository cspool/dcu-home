# SPDX-License-Identifier: Apache-2.0
"""Fail-closed ROCm TunableOp profile loader for one frozen workload."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
from dataclasses import dataclass
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
_OPERATOR: Final = "GemmTunableOp_BFloat16_TN"
_SOLUTION_RE: Final = re.compile(r"Gemm_Rocblas_[1-9][0-9]*\Z")
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")

_EXPECTED_SOLUTIONS: Final = {
    "tn_14336_4096_5120_ld_5120_5120_14336": "Gemm_Rocblas_20981",
    "tn_16384_4096_5120_ld_5120_5120_16384": "Gemm_Rocblas_20981",
    "tn_34816_4096_5120_ld_5120_5120_34816": "Gemm_Rocblas_20981",
    "tn_5120_4096_17408_ld_17408_17408_5120": "Gemm_Rocblas_20979",
    "tn_5120_4096_6144_ld_6144_6144_5120": "Gemm_Rocblas_20981",
}
_EXPECTED_PARAMS: Final = frozenset(_EXPECTED_SOLUTIONS)
_EXPECTED_VALIDATORS: Final = {
    "PT_VERSION": "2.10.0",
    "HIP_VERSION": "603",
    "GCN_ARCH_NAME": "gfx936:sramecc+:xnack-",
    "ROCBLAS_VERSION": "4.3.0.ef408db7-dirty",
    "HIPBLASLT_VERSION": "1000-a6254b89-dirty",
}


@dataclass(frozen=True)
class RocmTunableOpState:
    profile: str
    path: Path
    file_sha256: str
    validators_sha256: str
    results: tuple[tuple[str, str, str, float], ...]


@dataclass(frozen=True)
class _ParsedProfile:
    validators: dict[str, str]
    results: tuple[tuple[str, str, str, float], ...]


def _fail(message: str) -> RuntimeError:
    return RuntimeError(f"ROCm TunableOp profile rejected: {message}")


def _profile_path() -> Path:
    profile_dir = Path(__file__).with_name("tunable_profiles").resolve(strict=True)
    candidate = profile_dir / _PROFILE_FILENAME
    if candidate.is_symlink():
        raise _fail(f"profile path must not be a symlink: {candidate}")
    path = candidate.resolve(strict=True)
    if path.parent != profile_dir or not path.is_file() or not path.is_absolute():
        raise _fail(f"profile is not a regular wheel-internal file: {path}")
    return path


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validators_sha256(validators: dict[str, str]) -> str:
    payload = json.dumps(validators, sort_keys=True, separators=(",", ":")).encode()
    return _sha256(payload)


def _parse_profile(payload: bytes) -> _ParsedProfile:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _fail("profile is not UTF-8") from exc

    validators: dict[str, str] = {}
    results: list[tuple[str, str, str, float]] = []
    keys: set[tuple[str, str]] = set()
    for line_number, row in enumerate(csv.reader(io.StringIO(text)), start=1):
        if not row:
            continue
        if row[0] == "Validator":
            if len(row) != 3 or not row[1] or row[1] in validators:
                raise _fail(f"invalid/duplicate validator at line {line_number}")
            validators[row[1]] = row[2]
            continue
        if len(row) != 4:
            raise _fail(f"result row must have four fields at line {line_number}")
        operator, params, solution, raw_time = row
        key = (operator, params)
        if key in keys:
            raise _fail(f"duplicate result key at line {line_number}: {key}")
        keys.add(key)
        if operator != _OPERATOR or params not in _EXPECTED_PARAMS:
            raise _fail(f"unexpected result key at line {line_number}: {key}")
        if _SOLUTION_RE.fullmatch(solution) is None:
            raise _fail(f"unconfirmed/non-rocBLAS solution at line {line_number}")
        if solution != _EXPECTED_SOLUTIONS[params]:
            raise _fail(f"solution ID mismatch at line {line_number}: {solution}")
        try:
            elapsed_ms = float(raw_time)
        except ValueError as exc:
            raise _fail(f"invalid timing metadata at line {line_number}") from exc
        if elapsed_ms < 0:
            raise _fail(f"negative timing metadata at line {line_number}")
        results.append((operator, params, solution, elapsed_ms))

    if validators != _EXPECTED_VALIDATORS:
        raise _fail(f"validator table mismatch: {validators!r}")
    if len(results) != 5 or {row[1] for row in results} != _EXPECTED_PARAMS:
        raise _fail("profile must contain exactly the five confirmed explicit keys")
    return _ParsedProfile(validators=validators, results=tuple(results))


def _validate_scope(vllm_config: VllmConfig) -> None:
    import torch

    model = vllm_config.model_config
    scheduler = vllm_config.scheduler_config
    parallel = vllm_config.parallel_config
    actual = {
        "architectures": tuple(model.architectures),
        "dtype": model.dtype,
        "max_num_batched_tokens": scheduler.max_num_batched_tokens,
        "tensor_parallel_size": parallel.tensor_parallel_size,
        "pipeline_parallel_size": parallel.pipeline_parallel_size,
        "prefill_context_parallel_size": parallel.prefill_context_parallel_size,
        "data_parallel_size": parallel.data_parallel_size,
    }
    expected = {
        "architectures": ("Qwen3_5ForConditionalGeneration",),
        "dtype": torch.bfloat16,
        "max_num_batched_tokens": 4096,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "data_parallel_size": 1,
    }
    if actual != expected:
        raise _fail(f"workload scope mismatch: actual={actual!r}")


def _validate_environment(path: Path, file_sha256: str) -> None:
    expected_sha256 = os.environ.get(_PROFILE_SHA_ENV, "").lower()
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise _fail(f"{_PROFILE_SHA_ENV} must be one lowercase SHA256")
    if expected_sha256 != file_sha256:
        raise _fail(
            f"profile SHA256 mismatch: expected={expected_sha256}, actual={file_sha256}"
        )

    allowed: dict[str, set[str | None]] = {
        "PYTORCH_TUNABLEOP_ENABLED": {None, "1"},
        "PYTORCH_TUNABLEOP_TUNING": {None, "0"},
        "PYTORCH_TUNABLEOP_RECORD_UNTUNED": {None, "0"},
        "PYTORCH_TUNABLEOP_FILENAME": {None, str(path)},
        "PYTORCH_TUNABLEOP_ROCBLAS_ENABLED": {None, "1"},
        # This profile records rocBLAS only.  Disable unused hipBLASLt
        # candidates before the Tunable context reads and caches its env.
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
    normalized: list[tuple[str, str, str, float]] = []
    try:
        for row in raw_results:  # type: ignore[union-attr]
            if len(row) != 4:
                raise _fail(f"unexpected get_results row: {row!r}")
            normalized.append((str(row[0]), str(row[1]), str(row[2]), float(row[3])))
    except (TypeError, ValueError) as exc:
        raise _fail(f"unexpected get_results payload: {raw_results!r}") from exc
    return tuple(normalized)


def _assert_api_state(state: RocmTunableOpState) -> None:
    from torch.cuda import tunable

    if not tunable.is_enabled():
        raise _fail("TunableOp unexpectedly disabled")
    if tunable.tuning_is_enabled():
        raise _fail("online tuning unexpectedly enabled")
    if tunable.record_untuned_is_enabled():
        raise _fail("record-untuned unexpectedly enabled")
    api_results = _normalize_results(tunable.get_results())
    if set(api_results) != set(state.results) or len(api_results) != len(state.results):
        raise _fail("Tunable API results changed after initialization")
    api_validators = dict(tunable.get_validators())
    if api_validators != _EXPECTED_VALIDATORS:
        raise _fail(f"runtime validator drift: {api_validators!r}")
    if _sha256(state.path.read_bytes()) != state.file_sha256:
        raise _fail("wheel-internal profile changed after initialization")


def maybe_init_rocm_tunableop(
    vllm_config: VllmConfig, device: torch.device
) -> RocmTunableOpState | None:
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
        payload = path.read_bytes()
        file_sha256 = _sha256(payload)
        parsed = _parse_profile(payload)
        _validate_environment(path, file_sha256)

        # Environment values override Python setters and are cached.  The
        # checks above run before the first Tunable call in this helper.
        from torch.cuda import tunable

        tunable.tuning_enable(False)
        tunable.record_untuned_enable(False)
        tunable.set_filename(str(path), insert_device_ordinal=False)
        # get_results() creates the manager and performs its one lazy file read.
        # Do not call read_file() on this startup path.
        api_results = _normalize_results(tunable.get_results())
        api_validators = dict(tunable.get_validators())
        if api_validators != parsed.validators:
            raise _fail(f"runtime validator mismatch: {api_validators!r}")
        if set(api_results) != set(parsed.results) or len(api_results) != 5:
            raise _fail(f"runtime result mismatch: {api_results!r}")
        if _sha256(path.read_bytes()) != file_sha256:
            raise _fail("profile changed while TunableOp loaded it")

        tunable.enable(True)
        state = RocmTunableOpState(
            profile=profile,
            path=path,
            file_sha256=file_sha256,
            validators_sha256=_validators_sha256(parsed.validators),
            results=parsed.results,
        )
        _assert_api_state(state)
        logger.info(
            "VLLM_ROCM_TUNABLEOP_INIT status=ready device=%s file=%s "
            "file_sha256=%s validators_sha256=%s rows=5 logical_families=6 "
            "explicit_families=5 shared_explicit_families=2 enabled=1 tuning=0 "
            "record_untuned=0",
            device,
            path,
            state.file_sha256,
            state.validators_sha256,
        )
        return state
    except Exception as exc:
        logger.exception(
            "VLLM_ROCM_TUNABLEOP_INIT status=error profile=%s reason=%s",
            profile,
            exc,
        )
        raise


def assert_rocm_tunableop_pre_capture(state: RocmTunableOpState) -> None:
    """Fail closed immediately after kernel warmup and before graph capture."""
    _assert_api_state(state)
    logger.info(
        "VLLM_ROCM_TUNABLEOP_PRE_CAPTURE status=ready profile=%s rows=5 "
        "expected_hit_keys=5 shared_explicit_families=2 "
        "observed_dispatch=external_verbose_canary_required",
        state.profile,
    )
