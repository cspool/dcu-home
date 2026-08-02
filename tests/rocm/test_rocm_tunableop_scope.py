# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms.rocm import _is_validated_qwen35_data_parallel_topology
from vllm.platforms.rocm_tunableop import _validate_scope


def _config(
    *,
    tp: int = 1,
    local_dp: int = 1,
    original_dp: int = 1,
    original_local_dp: int = 1,
    dp_index: int = 0,
    local_dp_rank: int = 0,
    max_tokens: int = 4096,
):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=["Qwen3_5ForConditionalGeneration"],
            dtype=torch.bfloat16,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_tokens),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            data_parallel_size=local_dp,
            data_parallel_size_original=original_dp,
            data_parallel_size_local_original=original_local_dp,
            data_parallel_index=dp_index,
            data_parallel_rank_local=local_dp_rank,
        ),
    )


@pytest.mark.parametrize(
    ("original_dp", "original_local_dp", "dp_index", "local_dp_rank"),
    [
        (1, 1, 0, 0),
        (2, 2, 0, 0),
        (2, 2, 1, 1),
    ],
)
def test_supported_dp_topologies_are_accepted(
    original_dp: int,
    original_local_dp: int,
    dp_index: int,
    local_dp_rank: int,
):
    topology = _validate_scope(
        _config(
            original_dp=original_dp,
            original_local_dp=original_local_dp,
            dp_index=dp_index,
            local_dp_rank=local_dp_rank,
        )
    )
    assert topology == (
        original_dp,
        original_local_dp,
        dp_index,
        local_dp_rank,
    )


@pytest.mark.parametrize(
    ("tp", "local_dp", "max_tokens"),
    [
        (2, 1, 4096),
        (1, 2, 4096),
        (1, 1, 8192),
    ],
)
def test_non_tp1_local_dp1_scopes_fail_closed(
    tp: int,
    local_dp: int,
    max_tokens: int,
):
    with pytest.raises(RuntimeError, match="workload scope mismatch"):
        _validate_scope(
            _config(tp=tp, local_dp=local_dp, max_tokens=max_tokens)
        )


@pytest.mark.parametrize(
    ("original_dp", "original_local_dp", "dp_index", "local_dp_rank"),
    [
        (2, 1, 0, 0),
        (4, 4, 0, 0),
        (1, 1, 1, 1),
        (2, 2, 0, 1),
        (2, 2, 1, 0),
        (2, 2, 2, 2),
    ],
)
def test_unsupported_dp_topologies_fail_closed(
    original_dp: int,
    original_local_dp: int,
    dp_index: int,
    local_dp_rank: int,
):
    with pytest.raises(RuntimeError, match="workload DP topology mismatch"):
        _validate_scope(
            _config(
                original_dp=original_dp,
                original_local_dp=original_local_dp,
                dp_index=dp_index,
                local_dp_rank=local_dp_rank,
            )
        )


def _global_dp_config(**overrides):
    values = {
        "data_parallel_size": 2,
        "data_parallel_size_local": 2,
        "data_parallel_rank": 0,
        "data_parallel_rank_local": None,
        "data_parallel_backend": "mp",
        "data_parallel_external_lb": False,
        "data_parallel_hybrid_lb": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_static_shape_accepts_internal_local_dp2():
    assert _is_validated_qwen35_data_parallel_topology(_global_dp_config())


@pytest.mark.parametrize(
    "overrides",
    [
        {"data_parallel_size_local": 1},
        {"data_parallel_rank": 1},
        {"data_parallel_rank_local": 0},
        {"data_parallel_backend": "ray"},
        {"data_parallel_external_lb": True},
        {"data_parallel_hybrid_lb": True},
        {"data_parallel_size": 4, "data_parallel_size_local": 4},
    ],
)
def test_static_shape_rejects_unvalidated_dp2_topology(overrides):
    assert not _is_validated_qwen35_data_parallel_topology(
        _global_dp_config(**overrides)
    )
