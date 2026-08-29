#!/usr/bin/env python3
"""Run one deterministic Qwen3.5 request with opt-in layer HIPTX ranges."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


PROCESS_RANGE_RE = re.compile(
    r"^pra\.fx_process\."
    r"(?P<event>input\d+_layer\d+)\."
    r"(?P<process>[A-Za-z0-9_]+)"
    r"(?:\.(?P<fragment>[A-Za-z0-9_]+))?$"
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _parse_csv_set(name: str) -> set[str]:
    raw = os.environ.get(name, "")
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if len(values) != len(set(values)):
        raise RuntimeError(f"{name} contains duplicate values")
    return set(values)


def _parse_target_set(name: str, file_name: str) -> tuple[set[str], dict[str, Any]]:
    """Load a bounded target set from CSV text and/or a newline file.

    Large full-request process plans exceed a safe environment-variable size,
    so the file transport is the required path for that mode.  Combining both
    transports is supported only when their union has no duplicate.
    """
    inline = _parse_csv_set(name)
    file_value = os.environ.get(file_name, "").strip()
    from_file: list[str] = []
    resolved_file: Path | None = None
    if file_value:
        resolved_file = Path(file_value).expanduser().resolve()
        if not resolved_file.is_file():
            raise RuntimeError(f"{file_name} is not a file: {resolved_file}")
        for line_number, raw_line in enumerate(
            resolved_file.read_text(encoding="utf-8").splitlines(), 1
        ):
            value = raw_line.strip()
            if not value or value.startswith("#"):
                continue
            if "," in value:
                raise RuntimeError(
                    f"{file_name}:{line_number} must contain one target per line"
                )
            from_file.append(value)
        if len(from_file) != len(set(from_file)):
            raise RuntimeError(f"{file_name} contains duplicate values")
    duplicate = inline.intersection(from_file)
    if duplicate:
        raise RuntimeError(
            f"{name} and {file_name} overlap: {sorted(duplicate)[:8]}"
        )
    values = set(inline).union(from_file)
    provenance = {
        "inline_count": len(inline),
        "file_count": len(from_file),
        "total_count": len(values),
        "file": str(resolved_file) if resolved_file is not None else None,
        "file_sha256": (
            _sha256_bytes(resolved_file.read_bytes())
            if resolved_file is not None
            else None
        ),
    }
    return values, provenance


@contextmanager
def _live_utilization_window() -> Any:
    """Arm and stop the launcher-supervised live-utilization sidecar."""
    names = {
        "ready": "PRA_BACKEND_LIVE_UTIL_READY_FILE",
        "arm": "PRA_BACKEND_LIVE_UTIL_ARM_FILE",
        "stop": "PRA_BACKEND_LIVE_UTIL_STOP_FILE",
        "samples": "PRA_BACKEND_LIVE_UTIL_SAMPLES_FILE",
        "summary": "PRA_BACKEND_LIVE_UTIL_SUMMARY_FILE",
    }
    values = {key: os.environ.get(name, "").strip() for key, name in names.items()}
    configured = [key for key, value in values.items() if value]
    if not configured:
        yield {"enabled": False}
        return
    if len(configured) != len(names):
        raise RuntimeError(
            "live utilization sidecar requires all control/output paths; "
            f"configured={configured}"
        )
    paths = {key: Path(value).expanduser().resolve() for key, value in values.items()}
    if not paths["ready"].is_file():
        raise RuntimeError(f"live utilization sidecar is not ready: {paths['ready']}")
    ready = _load_json(paths["ready"])
    if ready.get("status") != "ready_waiting_for_arm":
        raise RuntimeError("live utilization ready record has an invalid status")
    for key in ("arm", "stop", "samples", "summary"):
        if paths[key].exists():
            raise RuntimeError(f"fresh live utilization path already exists: {paths[key]}")
    paths["arm"].parent.mkdir(parents=True, exist_ok=True)
    arm_realtime_ns = time.time_ns()
    arm_monotonic_ns = time.monotonic_ns()
    paths["arm"].touch(exist_ok=False)
    state = {
        "enabled": True,
        "ready": str(paths["ready"]),
        "arm": str(paths["arm"]),
        "stop": str(paths["stop"]),
        "samples": str(paths["samples"]),
        "summary": str(paths["summary"]),
        "arm_realtime_ns": arm_realtime_ns,
        "arm_monotonic_ns": arm_monotonic_ns,
        "ready_payload": ready,
    }
    try:
        yield state
    finally:
        state["stop_realtime_ns"] = time.time_ns()
        state["stop_monotonic_ns"] = time.monotonic_ns()
        paths["stop"].touch(exist_ok=False)


@contextmanager
def _optional_hipprof_session() -> Any:
    """Arm a parent hipprof session only around the measured request."""
    session_name = os.environ.get("PRA_HIPPROF_SESSION_NAME", "").strip()
    profile_kind = os.environ.get("PRA_HIPPROF_PROFILE_KIND", "").strip()
    hipprof = os.environ.get("PRA_HIPPROF_BIN", "/opt/dtk/bin/hipprof")
    if not session_name:
        if os.environ.get("PRA_HIPPROF_SESSION_REQUIRED") == "1":
            raise RuntimeError("required hipprof session name is missing")
        yield {
            "enabled": False,
            "session_name": "",
            "profile_kind": profile_kind,
            "hipprof": hipprof,
        }
        return
    if not all(
        character.isalnum() or character in "_.-"
        for character in session_name
    ):
        raise RuntimeError("hipprof session name contains unsafe characters")
    start = subprocess.run(
        [hipprof, "--session-client", session_name, "--start"],
        check=False,
        text=True,
        capture_output=True,
    )
    if start.returncode != 0:
        raise RuntimeError(
            "failed to start hipprof session: "
            f"stdout={start.stdout!r}, stderr={start.stderr!r}"
        )
    body_error: BaseException | None = None
    try:
        yield {
            "enabled": True,
            "session_name": session_name,
            "profile_kind": profile_kind,
            "hipprof": str(Path(hipprof).resolve()),
            "start_returncode": start.returncode,
        }
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        stop = subprocess.run(
            [hipprof, "--session-client", session_name, "--stop"],
            check=False,
            text=True,
            capture_output=True,
        )
        flush = subprocess.run(
            [hipprof, "--session-client", session_name, "--flush"],
            check=False,
            text=True,
            capture_output=True,
        )
        if body_error is None and (stop.returncode != 0 or flush.returncode != 0):
            raise RuntimeError(
                "failed to stop/flush hipprof session: "
                f"stop=({stop.returncode},{stop.stdout!r},{stop.stderr!r}), "
                f"flush=({flush.returncode},{flush.stdout!r},{flush.stderr!r})"
            )


def _activate_current_build(root_dir: Path) -> dict[str, str]:
    """Expose the current checkout's already-built extension modules."""
    import vllm

    candidates = sorted(root_dir.glob("build/lib.*-cpython-*/vllm"))
    for candidate in candidates:
        if (candidate / "_C.abi3.so").is_file():
            candidate_str = str(candidate.resolve())
            if candidate_str not in vllm.__path__:
                vllm.__path__.append(candidate_str)
            break
    else:
        raise RuntimeError("no current-checkout vllm/_C.abi3.so build artifact")

    import vllm._C
    import vllm._rocm_C

    return {
        "vllm_python": str(Path(vllm.__file__).resolve()),
        "vllm_C": str(Path(vllm._C.__file__).resolve()),
        "vllm_rocm_C": str(Path(vllm._rocm_C.__file__).resolve()),
    }


class ProcessRangeRecorder:
    """Install opt-in process ranges around the current eager operators."""

    def __init__(
        self,
        *,
        targets: set[str],
        exact_range_targets: set[str] | None = None,
    ) -> None:
        if not targets:
            raise RuntimeError("process profiling requires at least one target event")
        self.targets = set(targets)
        self.exact_range_targets = (
            set(exact_range_targets)
            if exact_range_targets is not None
            else None
        )
        if self.exact_range_targets is not None:
            if not self.exact_range_targets:
                raise RuntimeError("exact process range target list is empty")
            invalid = sorted(
                name
                for name in self.exact_range_targets
                if PROCESS_RANGE_RE.fullmatch(name) is None
            )
            if invalid:
                raise RuntimeError(
                    f"invalid exact process range targets: {invalid}"
                )
            range_events = {
                PROCESS_RANGE_RE.fullmatch(name).group("event")
                for name in self.exact_range_targets
            }
            if range_events != self.targets:
                raise RuntimeError(
                    "event and exact process range targets differ: "
                    f"events={sorted(self.targets)}, "
                    f"range_events={sorted(range_events)}"
                )
        self._state = threading.local()
        self.seen_targets: set[str] = set()
        self.expected_ranges: set[str] = set()
        self.emitted_ranges: set[str] = set()

    @staticmethod
    def _range_name(
        event_id: str,
        process_id: str,
        fragment_id: str | None = None,
    ) -> str:
        name = f"pra.fx_process.{event_id}.{process_id}"
        if fragment_id:
            name += f".{fragment_id}"
        return name

    def expected_names(self, event_id: str, layer_type: str) -> list[str]:
        common = [
            ("inputs", None),
            ("input_rmsnorm", None),
            ("qkv_projection", None),
        ]
        if layer_type == "linear_attention":
            attention_stages = [
                ("gdn_recurrent_core", None),
                ("gdn_gated_rmsnorm", None),
                ("output_projection", "part01_mixer_out_proj"),
            ]
        elif layer_type == "full_attention":
            attention_stages = [
                ("rope", None),
                ("kv_cache_attention", None),
                ("attention_output", None),
                ("output_projection", "part01_attention_o_proj"),
            ]
        else:
            raise RuntimeError(f"unsupported layer type: {layer_type}")
        tail = [
            (
                "output_projection__post_attention_rmsnorm_fused",
                "part02_shared_fusion",
            ),
            ("mlp", None),
            ("layer_output", None),
        ]
        names = [
            self._range_name(event_id, process_id, fragment_id)
            for process_id, fragment_id in common + attention_stages + tail
        ]
        if self.exact_range_targets is not None:
            names = [
                name for name in names if name in self.exact_range_targets
            ]
        return names

    def selected(self, event_id: str) -> bool:
        return event_id in self.targets

    @contextmanager
    def activate(self, event_id: str, layer_type: str):
        previous_event_id = getattr(self._state, "event_id", None)
        self._state.event_id = event_id
        self.seen_targets.add(event_id)
        self.expected_ranges.update(self.expected_names(event_id, layer_type))
        try:
            yield
        finally:
            self._state.event_id = previous_event_id

    @contextmanager
    def range(
        self,
        process_id: str,
        fragment_id: str | None = None,
    ):
        event_id = getattr(self._state, "event_id", None)
        if event_id is None:
            raise RuntimeError("process range entered outside a selected layer")
        name = self._range_name(event_id, process_id, fragment_id)
        if (
            self.exact_range_targets is not None
            and name not in self.exact_range_targets
        ):
            yield
            return

        import torch

        self.emitted_ranges.add(name)
        with torch.cuda.nvtx.range(name):
            yield

    def forward_decoder_layer(
        self,
        layer: Any,
        hidden_states: Any,
        residual: Any,
        positions: Any,
    ):
        """Mirror Qwen3NextDecoderLayer.forward with range-only additions."""
        import torch

        with self.range("inputs"):
            pass
        with self.range("input_rmsnorm"):
            if residual is None:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
            else:
                hidden_states, residual = layer.input_layernorm(
                    hidden_states, residual
                )
            self_attention_output = torch.empty_like(hidden_states)

        if layer.layer_type == "linear_attention":
            layer.linear_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
            )
        elif layer.layer_type == "full_attention":
            layer.self_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")
        hidden_states = self_attention_output

        if layer.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    layer.attn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    layer.attn_layer_scale.to(hidden_states.dtype) + 1
                )

        with self.range(
            "output_projection__post_attention_rmsnorm_fused",
            "part02_shared_fusion",
        ):
            hidden_states, residual = layer.post_attention_layernorm(
                hidden_states, residual
            )
        with self.range("mlp"):
            hidden_states = layer.mlp(hidden_states)

        if layer.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    layer.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                assert len(hidden_states.shape) == len(
                    layer.ffn_layer_scale.shape
                ), (
                    f"shape must be the same {len(hidden_states.shape)}, "
                    f"{len(layer.ffn_layer_scale.shape)}"
                )
                hidden_states = hidden_states * (
                    layer.ffn_layer_scale.to(hidden_states.dtype) + 1
                )

        with self.range("layer_output"):
            pass
        return hidden_states, residual

    def install(self) -> None:
        import torch
        from einops import rearrange
        from vllm.model_executor.models.qwen3_5 import (
            Qwen3_5GatedDeltaNet,
            _can_use_qwen35_gdn_strided_z_rmsnorm,
            _qwen35_gdn_strided_z_rmsnorm,
        )
        from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

        recorder = self

        original_gdn = Qwen3_5GatedDeltaNet.forward
        if getattr(original_gdn, "_pra_same_input_process_wrapped", False):
            raise RuntimeError("Qwen3_5GatedDeltaNet.forward already wrapped")

        @functools.wraps(original_gdn)
        def wrapped_gdn(
            module: Any,
            hidden_states: Any,
            output: Any,
        ):
            if getattr(recorder._state, "event_id", None) is None:
                return original_gdn(module, hidden_states, output)

            num_tokens = hidden_states.size(0)
            with recorder.range("qkv_projection"):
                mixed_qkvz, _ = module.in_proj_qkvz(hidden_states)
                qkv_size = (
                    module.key_dim * 2 + module.value_dim
                ) // module.tp_size
                z_size = module.value_dim // module.tp_size
                mixed_qkv, z = mixed_qkvz.split(
                    [qkv_size, z_size], dim=-1
                )
                z = z.reshape(z.size(0), -1, module.head_v_dim)
                ba, _ = module.in_proj_ba(hidden_states)
                b, a = ba.chunk(2, dim=-1)
                b = b.contiguous()
                a = a.contiguous()

            with recorder.range("gdn_recurrent_core"):
                # Keep this allocation identical to the current
                # Qwen3_5GatedDeltaNet.forward implementation.  The process
                # wrapper may add ranges only; it must not restore the older
                # zero-filled allocation or introduce an extra device kernel.
                core_attn_out = torch.empty(
                    (
                        num_tokens,
                        module.num_v_heads // module.tp_size,
                        module.head_v_dim,
                    ),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                torch.ops.vllm.gdn_attention_core(
                    mixed_qkv,
                    b,
                    a,
                    core_attn_out,
                    module.prefix,
                )

            with recorder.range("gdn_gated_rmsnorm"):
                if _can_use_qwen35_gdn_strided_z_rmsnorm(
                    core_attn_out, z, module.norm.weight
                ):
                    core_attn_out = _qwen35_gdn_strided_z_rmsnorm(
                        core_attn_out,
                        z,
                        module.norm.weight,
                        module.norm.eps,
                    )
                else:
                    z_shape_og = z.shape
                    core_attn_out = core_attn_out.reshape(
                        -1, core_attn_out.shape[-1]
                    )
                    z = z.reshape(-1, z.shape[-1])
                    core_attn_out = module.norm(core_attn_out, z)
                    core_attn_out = core_attn_out.reshape(z_shape_og)
                core_attn_out = rearrange(
                    core_attn_out, "... h d -> ... (h d)"
                )

            with recorder.range(
                "output_projection", "part01_mixer_out_proj"
            ):
                output[:num_tokens], _ = module.out_proj(core_attn_out)

        wrapped_gdn._pra_same_input_process_wrapped = True
        Qwen3_5GatedDeltaNet.forward = wrapped_gdn

        original_attention = Qwen3NextAttention.forward
        if getattr(
            original_attention, "_pra_same_input_process_wrapped", False
        ):
            raise RuntimeError("Qwen3NextAttention.forward already wrapped")

        @functools.wraps(original_attention)
        def wrapped_attention(
            module: Any,
            positions: Any,
            output: Any,
            hidden_states: Any,
        ):
            if getattr(recorder._state, "event_id", None) is None:
                return original_attention(
                    module,
                    positions=positions,
                    output=output,
                    hidden_states=hidden_states,
                )

            with recorder.range("qkv_projection"):
                qkv, _ = module.qkv_proj(hidden_states)
                if module.attn_output_gate:
                    q_gate, k, v = qkv.split(
                        [
                            module.q_size * 2,
                            module.kv_size,
                            module.kv_size,
                        ],
                        dim=-1,
                    )
                    orig_shape = q_gate.shape[:-1]
                    q_gate = q_gate.view(
                        *orig_shape, module.num_heads, -1
                    )
                    q, gate = torch.chunk(q_gate, 2, dim=-1)
                    q = q.reshape(*orig_shape, -1)
                    gate = gate.reshape(*orig_shape, -1)
                else:
                    q, k, v = qkv.split(
                        [module.q_size, module.kv_size, module.kv_size],
                        dim=-1,
                    )
                    gate = None
                q = module.q_norm(
                    q.view(-1, module.num_heads, module.head_dim)
                ).view(-1, module.num_heads * module.head_dim)
                k = module.k_norm(
                    k.view(-1, module.num_kv_heads, module.head_dim)
                ).view(-1, module.num_kv_heads * module.head_dim)

            with recorder.range("rope"):
                q, k = module.rotary_emb(positions, q, k)
            with recorder.range("kv_cache_attention"):
                attn_output = module.attn(q, k, v)
            with recorder.range("attention_output"):
                if module.attn_output_gate:
                    assert gate is not None
                    gate = torch.sigmoid(gate)
                    attn_output = attn_output * gate
            with recorder.range(
                "output_projection", "part01_attention_o_proj"
            ):
                output[:], _ = module.o_proj(attn_output)

        wrapped_attention._pra_same_input_process_wrapped = True
        Qwen3NextAttention.forward = wrapped_attention

    def assert_complete(self) -> None:
        missing = sorted(self.targets - self.seen_targets)
        extra = sorted(self.seen_targets - self.targets)
        if missing or extra:
            raise RuntimeError(
                f"process target coverage mismatch: missing={missing}, extra={extra}"
            )
        if not self.expected_ranges:
            raise RuntimeError("no process ranges were expected")
        if self.exact_range_targets is not None:
            missing_ranges = sorted(
                self.exact_range_targets - self.emitted_ranges
            )
            extra_ranges = sorted(
                self.emitted_ranges - self.exact_range_targets
            )
            if missing_ranges or extra_ranges:
                raise RuntimeError(
                    "exact process range coverage mismatch: "
                    f"missing={missing_ranges}, extra={extra_ranges}"
                )


class LayerRangeRecorder:
    """Attach deterministic layer ranges without changing model operators."""

    def __init__(
        self,
        *,
        contract_id: str,
        contract_sha256: str,
        event_path: Path,
        expected_layers: int,
        process_recorder: ProcessRangeRecorder | None = None,
    ) -> None:
        self.contract_id = contract_id
        self.contract_sha256 = contract_sha256
        self.event_path = event_path
        self.expected_layers = expected_layers
        self.process_recorder = process_recorder
        self.enabled = False
        self.forward_id = 0
        self.total_tokens = 0
        self.current: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            if self.current is not None:
                raise RuntimeError("cannot reset during a layer forward")
            self.forward_id = 0
            self.total_tokens = 0
            if self.event_path.exists():
                raise RuntimeError(f"refusing to overwrite {self.event_path}")

    def _begin_layer(
        self,
        layer_idx: int,
        q_len: int,
        workload_type: str,
    ) -> dict[str, Any]:
        with self._lock:
            if layer_idx == 0:
                if self.current is not None:
                    raise RuntimeError("new forward began before prior forward completed")
                self.forward_id += 1
                phase = "prefill_chunk" if q_len > 1 else "decode"
                self.current = {
                    "forward_id": self.forward_id,
                    "q_len": q_len,
                    "past_len": self.total_tokens,
                    "kv_len": self.total_tokens + q_len,
                    "phase": phase,
                    "next_layer": 0,
                }
            if self.current is None:
                raise RuntimeError(f"layer {layer_idx} observed without layer 0")
            if layer_idx != self.current["next_layer"]:
                raise RuntimeError(
                    f"layer order mismatch: expected {self.current['next_layer']}, "
                    f"observed {layer_idx}"
                )
            if q_len != self.current["q_len"]:
                raise RuntimeError("q_len changed within one model forward")

            event = {
                "contract_id": self.contract_id,
                "contract_sha256": self.contract_sha256,
                "forward_id": self.current["forward_id"],
                "layer_idx": layer_idx,
                # Workflow 04/05 stable identities use the 1-based
                # input/forward occurrence. Preserve the raw within-range
                # occurrence separately for exact historical-plan joins.
                "occurrence": self.current["forward_id"],
                "source_range_occurrence": 0,
                "phase": self.current["phase"],
                "q_len": self.current["q_len"],
                "past_len": self.current["past_len"],
                "kv_len": self.current["kv_len"],
                "workload_type": workload_type,
            }
            event["occurrence_key"] = (
                f"{self.contract_id}:forward={event['forward_id']}:"
                f"layer={layer_idx}:occurrence={event['occurrence']}"
            )
            event["range_name"] = (
                f"pra.layer.input{event['forward_id']}_layer{layer_idx}."
                f"{event['phase']}.{workload_type}"
            )
            event_id = f"input{event['forward_id']}_layer{layer_idx}"
            event["event_id"] = event_id
            if (
                self.process_recorder is not None
                and self.process_recorder.selected(event_id)
            ):
                event["expected_process_range_names"] = (
                    self.process_recorder.expected_names(
                        event_id, workload_type
                    )
                )
            else:
                event["expected_process_range_names"] = []
            return event

    def _finish_layer(self, event: dict[str, Any]) -> None:
        with self._lock:
            assert self.current is not None
            self.current["next_layer"] += 1
            if event["layer_idx"] == self.expected_layers - 1:
                if self.current["next_layer"] != self.expected_layers:
                    raise RuntimeError("incomplete layer sequence")
                self.total_tokens += self.current["q_len"]
                self.current = None

        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        with self.event_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )

    def install(self) -> None:
        import torch
        from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer

        original = Qwen3_5DecoderLayer.forward
        if getattr(original, "_pra_same_input_layer_wrapped", False):
            raise RuntimeError("Qwen3_5DecoderLayer.forward already wrapped")
        recorder = self

        @functools.wraps(original)
        def wrapped(layer: Any, *args: Any, **kwargs: Any):
            if not recorder.enabled:
                return original(layer, *args, **kwargs)

            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[0]
            if hidden_states is None:
                raise RuntimeError("layer hidden_states unavailable")
            q_len = int(hidden_states.shape[0])
            layer_idx = int(layer.layer_idx)
            workload_type = str(layer.layer_type)
            event = recorder._begin_layer(layer_idx, q_len, workload_type)
            host_start_ns = time.perf_counter_ns()
            with torch.cuda.nvtx.range(event["range_name"]):
                process_recorder = recorder.process_recorder
                if (
                    process_recorder is not None
                    and process_recorder.selected(event["event_id"])
                ):
                    hidden_states = kwargs.get(
                        "hidden_states", args[0] if args else None
                    )
                    residual = kwargs.get(
                        "residual", args[1] if len(args) > 1 else None
                    )
                    positions = kwargs.get(
                        "positions", args[2] if len(args) > 2 else None
                    )
                    if hidden_states is None:
                        raise RuntimeError(
                            "selected process layer lacks hidden_states"
                        )
                    with process_recorder.activate(
                        event["event_id"], workload_type
                    ):
                        result = process_recorder.forward_decoder_layer(
                            layer,
                            hidden_states,
                            residual,
                            positions,
                        )
                else:
                    result = original(layer, *args, **kwargs)
            event.update(
                {
                    "host_start_perf_ns": host_start_ns,
                    "host_end_perf_ns": time.perf_counter_ns(),
                    "pid": os.getpid(),
                    "tid": threading.get_native_id(),
                }
            )
            recorder._finish_layer(event)
            return result

        wrapped._pra_same_input_layer_wrapped = True
        Qwen3_5DecoderLayer.forward = wrapped

    def assert_complete(self) -> None:
        if self.current is not None:
            raise RuntimeError("last profiled forward is incomplete")
        if self.forward_id < 1:
            raise RuntimeError("no profiled model forward was recorded")


def _request_result(output: Any) -> dict[str, Any]:
    if len(output) != 1 or len(output[0].outputs) != 1:
        raise RuntimeError("expected exactly one request and one completion")
    request = output[0]
    completion = request.outputs[0]
    prompt_ids = list(request.prompt_token_ids or [])
    output_ids = list(completion.token_ids)
    text = str(completion.text)
    return {
        "prompt_token_count": len(prompt_ids),
        "prompt_token_ids_sha256": _canonical_json_sha256(prompt_ids),
        "output_token_count": len(output_ids),
        "output_token_ids_sha256": _canonical_json_sha256(output_ids),
        "output_text": text,
        "output_text_sha256": _sha256_bytes(text.encode("utf-8")),
        "finish_reason": completion.finish_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-row", type=int, default=0)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--lineage-manifest",
        type=Path,
        help=(
            "Optional same-run lineage manifest. Required by the fresh R07 "
            "workflow and recorded in the generated request metadata."
        ),
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--warmup-iters", type=int, required=True)
    parser.add_argument("--expected-layers", type=int, default=64)
    args = parser.parse_args()

    process_profile_flag = os.environ.get(
        "PRA_BACKEND_PERF_PROCESS_PROFILE"
    )
    if process_profile_flag not in {"0", "1"}:
        raise RuntimeError(
            "PRA_BACKEND_PERF_PROCESS_PROFILE must equal 0 or 1"
        )
    if os.environ.get("PRA_BACKEND_PERF_LAYER_PROFILE") != "1":
        raise RuntimeError("PRA_BACKEND_PERF_LAYER_PROFILE must equal 1")
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise RuntimeError("VLLM_ENABLE_V1_MULTIPROCESSING must equal 0")
    if args.max_new_tokens != 32:
        raise RuntimeError("MAX_NEW_TOKENS contract requires 32")
    if args.warmup_iters != 1:
        raise RuntimeError("WARMUP_ITERS contract requires 1")

    root_dir = args.root_dir.resolve()
    output_dir = args.output_dir.resolve()
    model_root = args.model_root.resolve()
    contract = _load_json(args.contract)
    contract_payload = dict(contract)
    recorded_contract_sha256 = str(contract_payload.pop("contract_sha256"))
    computed_contract_sha256 = _canonical_json_sha256(contract_payload)
    if recorded_contract_sha256 != computed_contract_sha256:
        raise RuntimeError("contract_sha256 mismatch")
    lineage: dict[str, Any] | None = None
    lineage_record: dict[str, Any] | None = None
    if args.lineage_manifest is not None:
        lineage_path = args.lineage_manifest.expanduser().resolve()
        if not lineage_path.is_file():
            raise RuntimeError(f"lineage manifest is missing: {lineage_path}")
        lineage = _load_json(lineage_path)
        if (
            lineage.get("schema_version") != 1
            or lineage.get("status") != "PASS"
            or lineage.get("evidence_source_policy") != "current_run_only"
            or lineage.get("source_change_policy")
            != "stage_trace_instrumentation_allowed"
            or lineage.get("source_hash_equality_required") is not False
            or lineage.get("semantic_contract_id") != contract.get("contract_id")
            or not isinstance(lineage.get("lineage_id"), str)
            or not lineage.get("lineage_id")
        ):
            raise RuntimeError(
                "lineage manifest is incompatible with this fresh request"
            )
        lineage_record = {
            "path": str(lineage_path),
            "sha256": _sha256_bytes(lineage_path.read_bytes()),
        }

    event_path = output_dir / f"{args.tag}.layer_events.runtime.jsonl"
    result_path = output_dir / f"{args.tag}.json"
    if event_path.exists() or result_path.exists():
        raise RuntimeError("fresh output files already exist")

    raw_lines = args.dataset.read_bytes().splitlines()
    if args.dataset_row < 0 or args.dataset_row >= len(raw_lines):
        raise RuntimeError("dataset row is out of range")
    raw_row = raw_lines[args.dataset_row]
    row = json.loads(raw_row)
    prompt = row.get("prompt")
    if not isinstance(prompt, str):
        raise RuntimeError("dataset row must contain a string prompt")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_root),
        trust_remote_code=True,
    )
    messages = [{"role": "user", "content": prompt}]
    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    rendered_ids = tokenizer(
        rendered_prompt,
        add_special_tokens=False,
    ).input_ids
    observed_prompt = {
        "dataset_row_raw_sha256": _sha256_bytes(raw_row),
        "prompt_text_sha256": _sha256_bytes(prompt.encode("utf-8")),
        "rendered_prompt_sha256": _sha256_bytes(
            rendered_prompt.encode("utf-8")
        ),
        "rendered_prompt_token_count": len(rendered_ids),
        "rendered_prompt_token_ids_sha256": _canonical_json_sha256(rendered_ids),
    }
    expected_prompt = contract.get("same_input", {}).get("prompt", {})
    for key, value in observed_prompt.items():
        if expected_prompt.get(key) != value:
            raise RuntimeError(
                f"prompt contract mismatch for {key}: "
                f"{expected_prompt.get(key)!r} != {value!r}"
            )

    build_binding = _activate_current_build(root_dir)

    import torch
    from vllm import LLM, SamplingParams

    process_targets, process_target_provenance = _parse_target_set(
        "PRA_BACKEND_PERF_PROCESS_TARGETS",
        "PRA_BACKEND_PERF_PROCESS_TARGETS_FILE",
    )
    exact_range_targets, exact_range_target_provenance = _parse_target_set(
        "PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS",
        "PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS_FILE",
    )
    if process_profile_flag == "0" and process_targets:
        raise RuntimeError(
            "process targets must be empty when process profiling is off"
        )
    if process_profile_flag == "0" and exact_range_targets:
        raise RuntimeError(
            "exact process range targets must be empty when profiling is off"
        )
    process_recorder = (
        ProcessRangeRecorder(
            targets=process_targets,
            exact_range_targets=(exact_range_targets or None),
        )
        if process_profile_flag == "1"
        else None
    )
    if process_recorder is not None:
        process_recorder.install()

    recorder = LayerRangeRecorder(
        contract_id=str(contract["contract_id"]),
        contract_sha256=recorded_contract_sha256,
        event_path=event_path,
        expected_layers=args.expected_layers,
        process_recorder=process_recorder,
    )
    recorder.install()

    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        seed=0,
        ignore_eos=False,
        max_tokens=args.max_new_tokens,
    )
    llm = LLM(
        model=str(model_root),
        served_model_name=args.served_model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_num_seqs=128,
        max_num_batched_tokens=4096,
        max_model_len=32768,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        seed=0,
        attention_config={
            "backend": "ROCM_AITER_UNIFIED_ATTN",
            "use_prefill_decode_attention": False,
        },
        disable_log_stats=True,
    )

    warmup_results: list[dict[str, Any]] = []
    recorder.enabled = False
    for _ in range(args.warmup_iters):
        warmup_output = llm.chat(
            messages,
            sampling_params=sampling_params,
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        torch.cuda.synchronize()
        warmup_results.append(_request_result(warmup_output))

    recorder.reset()
    torch.cuda.synchronize()
    request_marker = (
        "pra.request."
        f"contract_{recorded_contract_sha256[:16]}.tag_{args.tag}"
    )
    with _optional_hipprof_session() as hipprof_session:
        with _live_utilization_window() as live_utilization_window:
            recorder.enabled = True
            request_start_realtime_ns = time.time_ns()
            request_start_monotonic_ns = time.monotonic_ns()
            request_start_perf_ns = time.perf_counter_ns()
            with torch.cuda.nvtx.range(request_marker):
                measured_output = llm.chat(
                    messages,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                    chat_template_kwargs={"enable_thinking": False},
                )
                torch.cuda.synchronize()
            request_end_perf_ns = time.perf_counter_ns()
            request_end_monotonic_ns = time.monotonic_ns()
            request_end_realtime_ns = time.time_ns()
            recorder.enabled = False
    recorder.assert_complete()
    if process_recorder is not None:
        process_recorder.assert_complete()
    measured_result = _request_result(measured_output)

    for warmup in warmup_results:
        comparable_keys = (
            "prompt_token_count",
            "prompt_token_ids_sha256",
            "output_token_count",
            "output_token_ids_sha256",
            "output_text_sha256",
            "finish_reason",
        )
        if any(warmup[key] != measured_result[key] for key in comparable_keys):
            raise RuntimeError("warmup and measured request outputs differ")
    if measured_result["prompt_token_count"] != len(rendered_ids):
        raise RuntimeError("vLLM prompt token count differs from frozen rendering")
    if not 0 < measured_result["output_token_count"] <= args.max_new_tokens:
        raise RuntimeError("invalid measured output token count")
    expected_output = contract.get("same_input", {}).get(
        "expected_output", {}
    )
    if expected_output:
        for key in (
            "prompt_token_count",
            "prompt_token_ids_sha256",
            "output_token_count",
            "output_token_ids_sha256",
            "output_text_sha256",
            "finish_reason",
        ):
            if expected_output.get(key) != measured_result.get(key):
                raise RuntimeError(
                    f"frozen SAME_INPUT output mismatch for {key}: "
                    f"{expected_output.get(key)!r} != "
                    f"{measured_result.get(key)!r}"
                )

    metadata = {
        "schema_version": 1,
        "status": "profile_complete_analysis_pending",
        "tag": args.tag,
        "contract_id": contract["contract_id"],
        "contract_sha256": recorded_contract_sha256,
        "lineage_id": lineage.get("lineage_id") if lineage is not None else None,
        "fresh_run_lineage_manifest": lineage_record,
        "contract_relation": contract.get("contract_relation"),
        "parent_contract": contract.get("parent_contract"),
        "source_root": str(root_dir),
        "model_root": str(model_root),
        "served_model_name": args.served_model_name,
        "process_profile": (
            "on" if process_profile_flag == "1" else "off"
        ),
        "layer_marker_namespace": "pra.layer.inputN_layerM.<phase>.<workload_type>",
        "process_marker_namespace": (
            "pra.fx_process.inputN_layerM.<process_id>[.<fragment_id>]"
        ),
        "process_targets": sorted(process_targets),
        "process_target_transport": process_target_provenance,
        "exact_process_range_filter_enabled": bool(exact_range_targets),
        "exact_process_range_targets": sorted(exact_range_targets),
        "exact_process_range_target_transport": exact_range_target_provenance,
        "emitted_process_ranges": (
            sorted(process_recorder.emitted_ranges)
            if process_recorder is not None
            else []
        ),
        "expected_process_ranges": (
            sorted(process_recorder.expected_ranges)
            if process_recorder is not None
            else []
        ),
        "expected_process_range_count": (
            len(process_recorder.expected_ranges)
            if process_recorder is not None
            else 0
        ),
        "request_marker": request_marker,
        "expected_layers": args.expected_layers,
        "observed_forwards": recorder.forward_id,
        "observed_layer_events": recorder.forward_id * args.expected_layers,
        "warmup_iters": args.warmup_iters,
        "max_new_tokens": args.max_new_tokens,
        "request_synchronized_latency_ms": (
            request_end_perf_ns - request_start_perf_ns
        )
        / 1e6,
        "request_start_realtime_ns": request_start_realtime_ns,
        "request_end_realtime_ns": request_end_realtime_ns,
        "request_start_monotonic_ns": request_start_monotonic_ns,
        "request_end_monotonic_ns": request_end_monotonic_ns,
        "live_utilization_window": live_utilization_window,
        "profiler_session_control": hipprof_session,
        "request_synchronized_latency_is_replay_distorted": (
            hipprof_session.get("profile_kind")
            in {"pmc", "pmc-read", "pmc-write"}
        ),
        "same_input": observed_prompt,
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "seed": 0,
            "ignore_eos": False,
            "max_tokens": args.max_new_tokens,
            "enable_thinking": False,
        },
        "warmup_results": warmup_results,
        "measured_result": measured_result,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_hip": torch.version.hip,
            "device_name": torch.cuda.get_device_name(0),
            "logical_device": 0,
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            **build_binding,
        },
    }
    result_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "profile_complete_analysis_pending",
                "result": str(result_path),
                "events": str(event_path),
                "forwards": recorder.forward_id,
                "layers": recorder.forward_id * args.expected_layers,
                "request_synchronized_latency_ms": metadata[
                    "request_synchronized_latency_ms"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
