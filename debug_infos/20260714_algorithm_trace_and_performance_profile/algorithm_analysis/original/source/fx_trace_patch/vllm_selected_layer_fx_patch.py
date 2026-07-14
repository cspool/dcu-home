"""Selected-layer FX tracing patches for Qwen3.5 on vLLM.

The patch mirrors the VisiPrune FX workflow in a vLLM server setting:

1. Runtime generation continues through the normal eager decoder layer output.
2. Selected decoder-layer inputs are cloned at layer entry.
3. After the real layer forward returns, the cloned fixed input is replayed
   through ``make_fx`` while the vLLM forward context is still active.
4. Per-event FX artifacts and run-level manifests are written immediately.

This intentionally records FX DAG evidence separately from the existing
algorithm-level patch trace evidence.
"""

from __future__ import annotations

import atexit
import csv
import functools
import inspect
import json
import os
import shutil
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.fx as fx
from torch.fx.experimental.proxy_tensor import make_fx


_PATCHED = False
_STATE = threading.local()
_LOCK = threading.RLock()
_EVENT_INDEX = 0
_LAYER_EVENT_COUNT = 0
_FX_SAMPLE_COUNT = 0
_FX_TRACE_COUNT = 0
_FX_TRACE_ERROR_COUNT = 0
_CAPTURED_EVENTS: set[tuple[int, int]] = set()
_CAPTURED_BY_LAYER: Counter[int] = Counter()
_PATCH_ERRORS: list[dict[str, Any]] = []


LAYER_EVENT_FIELDS = [
    "event_id",
    "context",
    "pid",
    "engine_step_id",
    "forward_id",
    "phase",
    "total_num_scheduled_tokens",
    "layer_id",
    "layer_type",
    "q_len",
    "hidden_shape_in",
    "residual_shape_in",
    "positions_shape",
    "hidden_shape_out",
    "residual_shape_out",
    "matched",
    "fx_sampled",
    "fx_traced",
    "fx_trace_status",
    "fx_node_count",
    "trace_dir",
    "error",
    "duration_ns",
]

TRACE_MANIFEST_FIELDS = [
    "event_id",
    "context",
    "pid",
    "engine_step_id",
    "forward_id",
    "phase",
    "total_num_scheduled_tokens",
    "layer_id",
    "layer_type",
    "q_len",
    "status",
    "node_count",
    "trace_dir",
    "specialization",
    "input_binding",
    "error",
    "save_errors",
    "duration_ns",
]


def _enabled() -> bool:
    return os.environ.get("VLLM_SELECTED_LAYER_FX_ENABLE") == "1"


def _trace_dir() -> Path:
    return Path(os.environ.get("VLLM_SELECTED_LAYER_FX_DIR") or os.getcwd())


def _context() -> str | None:
    return os.environ.get("VLLM_SELECTED_LAYER_FX_CONTEXT")


def _arm_file() -> str | None:
    return os.environ.get("VLLM_SELECTED_LAYER_FX_ARM_FILE")


def _is_armed() -> bool:
    arm = _arm_file()
    return not arm or os.path.exists(arm)


def _event_log_path() -> Path:
    return _trace_dir() / f"events.{os.getpid()}.jsonl"


def _layer_events_path() -> Path:
    return _trace_dir() / "fx_layer_events.csv"


def _manifest_path() -> Path:
    return _trace_dir() / "fx_layer_trace_manifest.csv"


def _metadata_path() -> Path:
    return _trace_dir() / "run_metadata.json"


def _metadata_score(metadata: Mapping[str, Any]) -> int:
    score = 0
    for key in (
        "observed_layer_event_count",
        "fx_sample_count",
        "fx_trace_count",
        "fx_trace_error_count",
    ):
        score += _safe_int(metadata.get(key)) or 0
    captured = metadata.get("captured_events")
    if isinstance(captured, list):
        score += len(captured)
    return score


def _now_ns() -> int:
    return time.monotonic_ns()


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _shape(value: Any) -> list[int] | None:
    if not torch.is_tensor(value):
        return None
    try:
        return [int(item) for item in value.shape]
    except Exception:
        return None


def _shape_json(value: Any) -> str:
    return json.dumps(_shape(value), separators=(",", ":"))


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if torch.is_tensor(value):
        return {
            "type": "Tensor",
            "shape": _shape(value),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _compact_json(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)


def _write_event(event_type: str, **payload: Any) -> None:
    global _EVENT_INDEX
    if not _enabled():
        return
    try:
        _trace_dir().mkdir(parents=True, exist_ok=True)
        with _LOCK:
            _EVENT_INDEX += 1
            event = {
                "event_index": _EVENT_INDEX,
                "event_type": event_type,
                "ts_ns": _now_ns(),
                "pid": os.getpid(),
                "ppid": os.getppid(),
                "context": _context(),
                "argv0": os.path.basename(sys.argv[0]) if sys.argv else None,
                **payload,
            }
            with _event_log_path().open("a", encoding="utf-8") as handle:
                handle.write(_compact_json(event) + "\n")
    except BaseException as exc:  # pragma: no cover - tracing is best effort.
        try:
            sys.stderr.write(f"[vllm-selected-layer-fx] event write failed: {exc!r}\n")
        except Exception:
            pass


def _append_csv(path: Path, fields: list[str], row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    normalized = {field: row.get(field, "") for field in fields}
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(normalized)


def _parse_csv_targets(value: str) -> tuple[set[int], set[tuple[int, int]]]:
    any_layers: set[int] = set()
    event_keys: set[tuple[int, int]] = set()
    for raw in (value or "").split(","):
        item = raw.strip()
        if not item:
            continue
        if item.startswith("input") and "_layer" in item:
            left, right = item.split("_layer", 1)
            event_keys.add((int(left.removeprefix("input")), int(right)))
            continue
        if ":" in item:
            forward, layer = item.split(":", 1)
            event_keys.add((int(forward), int(layer)))
            continue
        any_layers.add(int(item))
    return any_layers, event_keys


def _target_sets() -> tuple[set[int], set[tuple[int, int]]]:
    return _parse_csv_targets(os.environ.get("VLLM_SELECTED_LAYER_FX_TARGETS", ""))


def _phase_set() -> set[str]:
    value = os.environ.get("VLLM_SELECTED_LAYER_FX_PHASES", "prefill_chunk")
    phases = {item.strip() for item in value.split(",") if item.strip()}
    return phases or {"prefill_chunk"}


def _max_samples_per_layer() -> int:
    return int(os.environ.get("VLLM_SELECTED_LAYER_FX_MAX_SAMPLES_PER_LAYER", "0") or "0")


def _trace_options() -> dict[str, Any]:
    options = {
        "tracing_mode": os.environ.get("VLLM_SELECTED_LAYER_FX_TRACING_MODE", "fake"),
        "pre_dispatch": os.environ.get("VLLM_SELECTED_LAYER_FX_PRE_DISPATCH") == "1",
        "record_module_stack": os.environ.get("VLLM_SELECTED_LAYER_FX_NO_MODULE_STACK") != "1",
        "record_stack_traces": os.environ.get("VLLM_SELECTED_LAYER_FX_RECORD_STACK_TRACES") == "1",
        "_allow_non_fake_inputs": os.environ.get("VLLM_SELECTED_LAYER_FX_ALLOW_NON_FAKE_INPUTS", "1") == "1",
    }
    signature = inspect.signature(make_fx)
    return {key: value for key, value in options.items() if key in signature.parameters}


def _current_step_id() -> int | None:
    return getattr(_STATE, "engine_step_id", None)


def _current_forward_id() -> int | None:
    return getattr(_STATE, "forward_id", None)


def _current_phase() -> str | None:
    return getattr(_STATE, "phase", None)


def _current_total_scheduled_tokens() -> int | None:
    return getattr(_STATE, "total_num_scheduled_tokens", None)


def _set_forward_context(
    *,
    engine_step_id: int | None,
    forward_id: int | None,
    phase: str | None,
    total_num_scheduled_tokens: int | None,
) -> tuple[int | None, int | None, str | None, int | None]:
    previous = (
        _current_step_id(),
        _current_forward_id(),
        _current_phase(),
        _current_total_scheduled_tokens(),
    )
    _STATE.engine_step_id = engine_step_id
    _STATE.forward_id = forward_id
    _STATE.phase = phase
    _STATE.total_num_scheduled_tokens = total_num_scheduled_tokens
    return previous


def _restore_forward_context(previous: tuple[int | None, int | None, str | None, int | None]) -> None:
    (
        _STATE.engine_step_id,
        _STATE.forward_id,
        _STATE.phase,
        _STATE.total_num_scheduled_tokens,
    ) = previous


def _next_id(obj: Any, attr: str) -> int:
    value = int(getattr(obj, attr, 0)) + 1
    setattr(obj, attr, value)
    return value


def _phase_from_scheduler_output(output: Any) -> str:
    total = _safe_int(getattr(output, "total_num_scheduled_tokens", None))
    if total is None:
        token_map = getattr(output, "num_scheduled_tokens", {}) or {}
        try:
            total = sum(int(value) for value in token_map.values())
        except Exception:
            total = None
    if total is None:
        return "unknown"
    if total == 0:
        return "empty"
    if total == 1:
        return "decode"
    return "prefill_chunk"


def _wrap_method(cls: Any, name: str, wrapper_factory: Any) -> None:
    original = getattr(cls, name, None)
    if original is None or getattr(original, "_vllm_selected_layer_fx_patched", False):
        return
    wrapped = wrapper_factory(original)
    wrapped._vllm_selected_layer_fx_patched = True
    setattr(cls, name, wrapped)


def _tensor_summary(value: Any) -> dict[str, Any] | None:
    if not torch.is_tensor(value):
        return None
    return {
        "shape": _shape(value),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "requires_grad": bool(value.requires_grad),
    }


def _describe_value(value: Any, depth: int = 0) -> Any:
    if torch.is_tensor(value):
        return _tensor_summary(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if depth >= 3:
        return type(value).__name__
    if isinstance(value, tuple):
        return {"tuple": [_describe_value(item, depth + 1) for item in value]}
    if isinstance(value, list):
        return [_describe_value(item, depth + 1) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _describe_value(item, depth + 1) for key, item in value.items()}
    return type(value).__name__


def _snapshot_tensor(value: torch.Tensor) -> torch.Tensor:
    snapshot = value.detach().clone(memory_format=torch.preserve_format)
    snapshot.requires_grad_(False)
    return snapshot


def _snapshot_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return _snapshot_tensor(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return tuple(_snapshot_value(item) for item in value)
    if isinstance(value, list):
        return [_snapshot_value(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _snapshot_value(item) for key, item in value.items()}
    return value


def _snapshot_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    return _snapshot_value(args), _snapshot_value(kwargs)


def _short_repr(value: Any, limit: int = 500) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<{type(value).__name__}>"
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _serialize_node(index: int, node: fx.Node) -> dict[str, Any]:
    meta = {}
    tensor_meta = node.meta.get("tensor_meta") if hasattr(node, "meta") else None
    if tensor_meta is not None:
        meta["tensor_meta"] = _jsonable(tensor_meta)
    if "val" in node.meta:
        meta["val"] = _describe_value(node.meta["val"])
    return {
        "index": index,
        "name": node.name,
        "op": node.op,
        "target": str(node.target),
        "args": _short_repr(node.args),
        "kwargs": _short_repr(dict(node.kwargs)),
        "users": sorted(user.name for user in node.users),
        "meta": meta,
    }


def _meta_like_tensor(value: torch.Tensor) -> torch.Tensor:
    return torch.empty_strided(
        tuple(value.shape),
        tuple(value.stride()),
        dtype=value.dtype,
        device="meta",
    )


def _strip_graph_module_tensor_data(graph_module: fx.GraphModule) -> dict[str, Any]:
    stripped = {
        "enabled": True,
        "parameters": 0,
        "buffers": 0,
        "tensor_attributes": 0,
        "bytes_avoided_estimate": 0,
    }
    for module in graph_module.modules():
        parameters = getattr(module, "_parameters", {})
        for name, parameter in list(parameters.items()):
            if parameter is None or not torch.is_tensor(parameter):
                continue
            stripped["parameters"] += 1
            stripped["bytes_avoided_estimate"] += parameter.numel() * parameter.element_size()
            parameters[name] = torch.nn.Parameter(
                _meta_like_tensor(parameter),
                requires_grad=bool(getattr(parameter, "requires_grad", False)),
            )
        buffers = getattr(module, "_buffers", {})
        for name, buffer in list(buffers.items()):
            if buffer is None or not torch.is_tensor(buffer):
                continue
            stripped["buffers"] += 1
            stripped["bytes_avoided_estimate"] += buffer.numel() * buffer.element_size()
            buffers[name] = _meta_like_tensor(buffer)

    for name, value in list(vars(graph_module).items()):
        if torch.is_tensor(value) and value.device.type != "meta":
            stripped["tensor_attributes"] += 1
            stripped["bytes_avoided_estimate"] += value.numel() * value.element_size()
            setattr(graph_module, name, _meta_like_tensor(value))
    return stripped


def _make_fx_positional_call(
    original_forward: Any,
    layer: Any,
    call_args: tuple[Any, ...],
    call_kwargs: dict[str, Any],
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
    keyword_names = tuple(call_kwargs)
    positional_count = len(call_args)
    flat_args = tuple(call_args) + tuple(call_kwargs[name] for name in keyword_names)

    def target(*flat_call_args: Any) -> Any:
        original_args = flat_call_args[:positional_count]
        original_kwargs = {
            name: flat_call_args[positional_count + index]
            for index, name in enumerate(keyword_names)
        }
        return original_forward(layer, *original_args, **original_kwargs)

    binding = {
        "positional_arg_count": positional_count,
        "keyword_names": list(keyword_names),
        "flat_inputs": [
            {"flat_index": index, "source": "arg", "source_index": index}
            for index in range(positional_count)
        ]
        + [
            {
                "flat_index": positional_count + index,
                "source": "kwarg",
                "source_name": name,
            }
            for index, name in enumerate(keyword_names)
        ],
    }
    return target, flat_args, binding


def _write_trace_outputs(
    trace_dir: Path,
    graph_module: fx.GraphModule,
    metadata: dict[str, Any],
) -> list[str]:
    save_errors: list[str] = []
    trace_dir.mkdir(parents=True, exist_ok=True)
    nodes = [_serialize_node(index, node) for index, node in enumerate(graph_module.graph.nodes)]
    (trace_dir / "fx_graph.py").write_text(graph_module.code + "\n", encoding="utf-8")
    (trace_dir / "fx_graph.txt").write_text(str(graph_module.graph) + "\n", encoding="utf-8")
    (trace_dir / "fx_nodes.json").write_text(
        json.dumps(nodes, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    tensor_storage = {"enabled": False, "reason": "disabled"}
    if os.environ.get("VLLM_SELECTED_LAYER_FX_STRIP_TENSOR_DATA_FOR_SAVE", "1") == "1":
        tensor_storage = _strip_graph_module_tensor_data(graph_module)

    try:
        torch.save(graph_module, trace_dir / "fx_graph_module.pt")
    except BaseException as exc:
        save_errors.append(f"torch.save: {exc!r}")

    graph_module_dir = trace_dir / "fx_graph_module"
    try:
        if graph_module_dir.exists():
            shutil.rmtree(graph_module_dir)
        graph_module.to_folder(graph_module_dir, module_name="FxLayerGraphModule")
    except BaseException as exc:
        save_errors.append(f"to_folder: {exc!r}")
        graph_module_dir.mkdir(parents=True, exist_ok=True)
        (graph_module_dir / "to_folder_error.txt").write_text(repr(exc) + "\n", encoding="utf-8")

    metadata = {
        **metadata,
        "node_count": len(nodes),
        "save_errors": save_errors,
        "graph_module_tensor_storage": tensor_storage,
        "outputs": {
            "fx_graph_py": str(trace_dir / "fx_graph.py"),
            "fx_graph_txt": str(trace_dir / "fx_graph.txt"),
            "fx_nodes_json": str(trace_dir / "fx_nodes.json"),
            "fx_graph_module_pt": str(trace_dir / "fx_graph_module.pt"),
            "fx_graph_module_dir": str(graph_module_dir),
        },
    }
    (trace_dir / "fx_trace_metadata.json").write_text(
        json.dumps(_jsonable(metadata), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return save_errors


@contextmanager
def _temporarily_unwrap_compiled_layernorms(layer: Any):
    """Restore vLLM RMSNorm instance methods that forward_cuda torch.compiles.

    GemmaRMSNorm.forward_cuda compiles ``_forward_static_no_residual`` and
    ``_forward_static_with_residual`` onto the instance after the first runtime
    call. make_fx cannot symbolically trace through those Dynamo wrappers, so
    replay uses the original class functions and restores the instance state
    afterwards.
    """

    restores: list[tuple[Any, str, Any, bool]] = []
    try:
        for module in layer.modules():
            for name in ("_forward_static_no_residual", "_forward_static_with_residual"):
                class_value = getattr(module.__class__, name, None)
                if class_value is None:
                    continue
                try:
                    current = getattr(module, name)
                except Exception:
                    continue
                if current is class_value:
                    continue
                restores.append((module, name, current, name in getattr(module, "__dict__", {})))
                setattr(module, name, class_value)
            if hasattr(module, "_is_compiled"):
                restores.append((module, "_is_compiled", getattr(module, "_is_compiled"), "_is_compiled" in getattr(module, "__dict__", {})))
                setattr(module, "_is_compiled", True)
        yield
    finally:
        for module, name, value, existed in reversed(restores):
            if existed:
                setattr(module, name, value)
            else:
                try:
                    delattr(module, name)
                except AttributeError:
                    pass


@contextmanager
def _temporarily_use_default_unquantized_gemm():
    """Avoid ROCm custom GEMM dispatch while tracing with ProxyTensor."""

    restores: list[tuple[Any, str, Any]] = []
    try:
        def fx_default_unquantized_gemm(
            layer: torch.nn.Module,
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            return torch.nn.functional.linear(x, weight, bias)

        def default_dispatch() -> Any:
            return fx_default_unquantized_gemm

        module_names = [
            "vllm.model_executor.layers.utils",
            "vllm.model_executor.layers.linear",
            "vllm.model_executor.layers.vocab_parallel_embedding",
            "vllm.model_executor.layers.quantization.compressed_tensors.transform.module",
        ]
        for module_name in module_names:
            try:
                module = __import__(module_name, fromlist=["_"])
            except Exception:
                continue
            if hasattr(module, "dispatch_unquantized_gemm"):
                restores.append((module, "dispatch_unquantized_gemm", getattr(module, "dispatch_unquantized_gemm")))
                setattr(module, "dispatch_unquantized_gemm", default_dispatch)
        yield
    finally:
        for module, name, value in reversed(restores):
            setattr(module, name, value)


@contextmanager
def _temporarily_plain_parameters(layer: Any):
    """Replace vLLM parameter subclasses with plain Parameters for FX replay."""

    restores: list[tuple[Any, str, Any]] = []
    try:
        for module in layer.modules():
            parameters = getattr(module, "_parameters", None)
            if not isinstance(parameters, dict):
                continue
            for name, parameter in list(parameters.items()):
                if parameter is None:
                    continue
                if type(parameter) is torch.nn.Parameter:
                    continue
                if not torch.is_tensor(parameter):
                    continue
                plain = torch.nn.Parameter(
                    parameter.detach(),
                    requires_grad=bool(getattr(parameter, "requires_grad", False)),
                )
                restores.append((module, name, parameter))
                parameters[name] = plain
        yield
    finally:
        for module, name, parameter in reversed(restores):
            module._parameters[name] = parameter


@contextmanager
def _temporarily_use_native_rotary(layer: Any):
    """Route rotary embedding custom ops to their Python native path."""

    restores: list[tuple[Any, str, Any]] = []
    try:
        for module in layer.modules():
            if not hasattr(module, "forward_native"):
                continue
            name = module.__class__.__name__.lower()
            module_path = module.__class__.__module__.lower()
            if "rotary" not in name and "rotary_embedding" not in module_path:
                continue
            if hasattr(module, "_forward_method"):
                restores.append((module, "_forward_method", getattr(module, "_forward_method")))
                setattr(module, "_forward_method", module.forward_native)
        yield
    finally:
        for module, name, value in reversed(restores):
            setattr(module, name, value)


def _write_run_metadata() -> None:
    if not _enabled():
        return
    any_layers, event_keys = _target_sets()
    metadata = {
        "analysis_type": "vllm_qwen35_selected_layer_fx_trace",
        "trace_strategy": "runtime_eager_layer_forward_then_inline_fixed_input_make_fx",
        "evidence_boundary": {
            "runtime_sampling": "selected decoder-layer inputs are cloned at runtime layer entry",
            "offline_fx_dag": "cloned fixed inputs are replayed with make_fx; generation output comes from eager layer forward",
        },
        "context": _context(),
        "pid": os.getpid(),
        "targets": os.environ.get("VLLM_SELECTED_LAYER_FX_TARGETS", ""),
        "target_any_forward_layers": sorted(any_layers),
        "target_event_keys": [f"input{forward}_layer{layer}" for forward, layer in sorted(event_keys)],
        "target_phases": sorted(_phase_set()),
        "selected_layer_rationale": os.environ.get("VLLM_SELECTED_LAYER_FX_RATIONALE", ""),
        "layerwise_source": os.environ.get("VLLM_SELECTED_LAYER_FX_LAYERWISE_SOURCE", ""),
        "max_samples_per_layer": _max_samples_per_layer(),
        "tracing_options": _trace_options(),
        "observed_layer_event_count": _LAYER_EVENT_COUNT,
        "fx_sample_count": _FX_SAMPLE_COUNT,
        "fx_trace_count": _FX_TRACE_COUNT,
        "fx_trace_error_count": _FX_TRACE_ERROR_COUNT,
        "captured_events": [f"input{forward}_layer{layer}" for forward, layer in sorted(_CAPTURED_EVENTS)],
        "captured_by_layer": {str(key): value for key, value in sorted(_CAPTURED_BY_LAYER.items())},
        "patch_errors": _PATCH_ERRORS,
        "outputs": {
            "fx_layer_events": str(_layer_events_path()),
            "fx_layer_trace_manifest": str(_manifest_path()),
            "run_metadata": str(_metadata_path()),
        },
        "expected_event_outputs": [
            "fx_graph.py",
            "fx_graph.txt",
            "fx_nodes.json",
            "fx_graph_module.pt",
            "fx_graph_module/",
            "fx_trace_metadata.json",
        ],
    }
    try:
        _trace_dir().mkdir(parents=True, exist_ok=True)
        with _LOCK:
            path = _metadata_path()
            if path.exists():
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    existing = {}
                if isinstance(existing, Mapping) and _metadata_score(existing) > _metadata_score(metadata):
                    return
            tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            tmp_path.write_text(
                json.dumps(_jsonable(metadata), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            tmp_path.replace(path)
    except BaseException as exc:
        _write_event("patch_error", where="write_run_metadata", error=repr(exc))


def _should_trace(forward_id: int, layer_id: int, phase: str) -> bool:
    if phase not in _phase_set() and "*" not in _phase_set():
        return False
    any_layers, event_keys = _target_sets()
    if not any_layers and not event_keys:
        return False
    if event_keys:
        matched = (forward_id, layer_id) in event_keys
    else:
        matched = layer_id in any_layers
    if not matched:
        return False
    max_per_layer = _max_samples_per_layer()
    if max_per_layer > 0 and _CAPTURED_BY_LAYER[layer_id] >= max_per_layer:
        return False
    return (forward_id, layer_id) not in _CAPTURED_EVENTS


def _run_fx_trace(
    *,
    layer: Any,
    original_forward: Any,
    sample_args: tuple[Any, ...],
    sample_kwargs: dict[str, Any],
    layer_row: dict[str, Any],
) -> None:
    global _FX_SAMPLE_COUNT, _FX_TRACE_COUNT, _FX_TRACE_ERROR_COUNT

    event_id = layer_row["event_id"]
    trace_dir = _trace_dir() / "traces" / event_id
    manifest_row = {
        key: layer_row.get(key, "")
        for key in (
            "event_id",
            "context",
            "pid",
            "engine_step_id",
            "forward_id",
            "phase",
            "total_num_scheduled_tokens",
            "layer_id",
            "layer_type",
            "q_len",
        )
    }
    manifest_row["trace_dir"] = str(trace_dir)
    _FX_SAMPLE_COUNT += 1

    t0 = _now_ns()
    try:
        target, flat_args, input_binding = _make_fx_positional_call(
            original_forward,
            layer,
            sample_args,
            sample_kwargs,
        )
        specialization = {
            "kind": "vllm_decoder_layer_fixed_runtime_input",
            "layer_class": f"{layer.__class__.__module__}.{layer.__class__.__qualname__}",
            "original_forward": f"{original_forward.__module__}.{original_forward.__qualname__}",
            "layer_idx": _safe_int(getattr(layer, "layer_idx", None)),
            "layer_type": getattr(layer, "layer_type", None),
            "runtime_inputs": {
                "args": _describe_value(sample_args),
                "kwargs": _describe_value(sample_kwargs),
            },
            "note": "FX replay runs after the eager layer output is produced, while vLLM forward context is still active.",
        }
        with (
            torch.inference_mode(),
            _temporarily_unwrap_compiled_layernorms(layer),
            _temporarily_plain_parameters(layer),
            _temporarily_use_native_rotary(layer),
            _temporarily_use_default_unquantized_gemm(),
        ):
            graph_module = make_fx(target, **_trace_options())(*flat_args)
        save_errors = _write_trace_outputs(
            trace_dir,
            graph_module,
            {
                "target": f"{event_id}:layer{layer_row['layer_id']}.forward",
                "context": _context(),
                "event_id": event_id,
                "engine_step_id": layer_row.get("engine_step_id"),
                "forward_id": layer_row.get("forward_id"),
                "phase": layer_row.get("phase"),
                "q_len": layer_row.get("q_len"),
                "input_binding": input_binding,
                "specialization": specialization,
                "tracing_options": _trace_options(),
            },
        )
        node_count = len(list(graph_module.graph.nodes))
        manifest_row.update(
            {
                "status": "ok",
                "node_count": node_count,
                "specialization": json.dumps(_jsonable(specialization), ensure_ascii=False, sort_keys=True),
                "input_binding": json.dumps(_jsonable(input_binding), ensure_ascii=False, sort_keys=True),
                "save_errors": json.dumps(save_errors, ensure_ascii=False),
                "duration_ns": _now_ns() - t0,
            }
        )
        layer_row["fx_traced"] = True
        layer_row["fx_trace_status"] = "ok"
        layer_row["fx_node_count"] = node_count
        layer_row["trace_dir"] = str(trace_dir)
        _FX_TRACE_COUNT += 1
        _write_event(
            "fx_trace_ok",
            event_id=event_id,
            layer_id=layer_row.get("layer_id"),
            node_count=node_count,
            trace_dir=str(trace_dir),
            duration_ns=manifest_row["duration_ns"],
            save_errors=save_errors,
        )
    except BaseException as exc:
        error = repr(exc)
        manifest_row.update(
            {
                "status": "error",
                "node_count": "",
                "error": error,
                "duration_ns": _now_ns() - t0,
            }
        )
        layer_row["fx_traced"] = False
        layer_row["fx_trace_status"] = "error"
        layer_row["error"] = error
        layer_row["trace_dir"] = str(trace_dir)
        _FX_TRACE_ERROR_COUNT += 1
        _write_event(
            "fx_trace_error",
            event_id=event_id,
            layer_id=layer_row.get("layer_id"),
            error=error,
            traceback=traceback.format_exc(),
        )
        if os.environ.get("VLLM_SELECTED_LAYER_FX_STRICT") == "1":
            raise
    finally:
        with _LOCK:
            _append_csv(_manifest_path(), TRACE_MANIFEST_FIELDS, manifest_row)
            _write_run_metadata()


def _patch_engine() -> None:
    from vllm.v1.engine.core import EngineCore

    def step_wrapper(original: Any) -> Any:
        @functools.wraps(original)
        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            step_id = _next_id(self, "_selected_layer_fx_engine_step_id")
            previous = _set_forward_context(
                engine_step_id=step_id,
                forward_id=_current_forward_id(),
                phase=_current_phase(),
                total_num_scheduled_tokens=_current_total_scheduled_tokens(),
            )
            try:
                return original(self, *args, **kwargs)
            finally:
                _restore_forward_context(previous)

        return wrapped

    _wrap_method(EngineCore, "step", step_wrapper)
    _wrap_method(EngineCore, "step_with_batch_queue", step_wrapper)


def _patch_scheduler() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    def schedule_wrapper(original: Any) -> Any:
        @functools.wraps(original)
        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            output = original(self, *args, **kwargs)
            step_id = _current_step_id()
            if step_id is None:
                step_id = _next_id(self, "_selected_layer_fx_scheduler_step_id")
            setattr(output, "_selected_layer_fx_engine_step_id", step_id)
            setattr(output, "_selected_layer_fx_phase", _phase_from_scheduler_output(output))
            return output

        return wrapped

    _wrap_method(Scheduler, "schedule", schedule_wrapper)


def _patch_model_runner() -> None:
    try:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    except Exception:
        return

    def execute_wrapper(original: Any) -> Any:
        @functools.wraps(original)
        def wrapped(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
            forward_id = _next_id(self, "_selected_layer_fx_forward_id")
            step_id = getattr(scheduler_output, "_selected_layer_fx_engine_step_id", _current_step_id())
            phase = getattr(scheduler_output, "_selected_layer_fx_phase", _phase_from_scheduler_output(scheduler_output))
            total = _safe_int(getattr(scheduler_output, "total_num_scheduled_tokens", None))
            previous = _set_forward_context(
                engine_step_id=step_id,
                forward_id=forward_id,
                phase=phase,
                total_num_scheduled_tokens=total,
            )
            try:
                return original(self, scheduler_output, *args, **kwargs)
            finally:
                _restore_forward_context(previous)

        return wrapped

    _wrap_method(GPUModelRunner, "execute_model", execute_wrapper)


def _patch_v1_model_runner() -> None:
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception:
        return

    def execute_wrapper(original: Any) -> Any:
        @functools.wraps(original)
        def wrapped(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
            forward_id = _next_id(self, "_selected_layer_fx_forward_id")
            step_id = getattr(scheduler_output, "_selected_layer_fx_engine_step_id", _current_step_id())
            phase = getattr(scheduler_output, "_selected_layer_fx_phase", _phase_from_scheduler_output(scheduler_output))
            total = _safe_int(getattr(scheduler_output, "total_num_scheduled_tokens", None))
            previous = _set_forward_context(
                engine_step_id=step_id,
                forward_id=forward_id,
                phase=phase,
                total_num_scheduled_tokens=total,
            )
            try:
                return original(self, scheduler_output, *args, **kwargs)
            finally:
                _restore_forward_context(previous)

        return wrapped

    _wrap_method(GPUModelRunner, "execute_model", execute_wrapper)


def _patch_qwen35_decoder_layer() -> None:
    from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer

    def forward_wrapper(original: Any) -> Any:
        @functools.wraps(original)
        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not _is_armed():
                return original(self, *args, **kwargs)

            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[0]
            residual = kwargs.get("residual")
            if residual is None and len(args) > 1:
                residual = args[1]
            positions = kwargs.get("positions")
            if positions is None and len(args) > 2:
                positions = args[2]

            layer_id = _safe_int(getattr(self, "layer_idx", None))
            forward_id = _safe_int(_current_forward_id()) or -1
            phase = _current_phase() or "unknown"
            q_len = None
            if torch.is_tensor(hidden_states) and hidden_states.dim() >= 1:
                q_len = int(hidden_states.shape[0])
                if phase == "unknown":
                    phase = "prefill_chunk" if q_len > 1 else "decode"
            event_id = f"input{forward_id}_layer{layer_id}"
            t0 = _now_ns()
            matched = layer_id is not None and _should_trace(forward_id, layer_id, phase)

            sample_args: tuple[Any, ...] | None = None
            sample_kwargs: dict[str, Any] | None = None
            if matched:
                sample_args, sample_kwargs = _snapshot_inputs(tuple(args), dict(kwargs))
                _CAPTURED_EVENTS.add((forward_id, layer_id))
                _CAPTURED_BY_LAYER[layer_id] += 1

            output = original(self, *args, **kwargs)

            hidden_out = None
            residual_out = None
            if isinstance(output, tuple):
                if output:
                    hidden_out = output[0]
                if len(output) > 1:
                    residual_out = output[1]
            else:
                hidden_out = output

            layer_row = {
                "event_id": event_id,
                "context": _context(),
                "pid": os.getpid(),
                "engine_step_id": _current_step_id(),
                "forward_id": forward_id,
                "phase": phase,
                "total_num_scheduled_tokens": _current_total_scheduled_tokens(),
                "layer_id": layer_id,
                "layer_type": getattr(self, "layer_type", None),
                "q_len": q_len,
                "hidden_shape_in": _shape_json(hidden_states),
                "residual_shape_in": _shape_json(residual),
                "positions_shape": _shape_json(positions),
                "hidden_shape_out": _shape_json(hidden_out),
                "residual_shape_out": _shape_json(residual_out),
                "matched": bool(matched),
                "fx_sampled": bool(matched),
                "fx_traced": False,
                "fx_trace_status": "",
                "fx_node_count": "",
                "trace_dir": "",
                "error": "",
                "duration_ns": _now_ns() - t0,
            }

            if matched and sample_args is not None and sample_kwargs is not None:
                _run_fx_trace(
                    layer=self,
                    original_forward=original,
                    sample_args=sample_args,
                    sample_kwargs=sample_kwargs,
                    layer_row=layer_row,
                )

            global _LAYER_EVENT_COUNT
            with _LOCK:
                _LAYER_EVENT_COUNT += 1
                _append_csv(_layer_events_path(), LAYER_EVENT_FIELDS, layer_row)
                _write_run_metadata()
            return output

        return wrapped

    _wrap_method(Qwen3_5DecoderLayer, "forward", forward_wrapper)


def apply_patches() -> None:
    global _PATCHED
    if _PATCHED or not _enabled():
        return
    _PATCHED = True

    _trace_dir().mkdir(parents=True, exist_ok=True)
    for patch_fn in (
        _patch_engine,
        _patch_scheduler,
        _patch_model_runner,
        _patch_v1_model_runner,
        _patch_qwen35_decoder_layer,
    ):
        try:
            patch_fn()
        except BaseException:
            error = {"patch": patch_fn.__name__, "traceback": traceback.format_exc()}
            _PATCH_ERRORS.append(error)
            _write_event("patch_error", **error)

    _write_event(
        "patch_loaded",
        patches=[
            "engine",
            "scheduler",
            "model_runner",
            "v1_model_runner",
            "qwen35_decoder_layer",
        ],
        targets=os.environ.get("VLLM_SELECTED_LAYER_FX_TARGETS", ""),
        phases=sorted(_phase_set()),
        trace_options=_trace_options(),
        errors=_PATCH_ERRORS,
    )
    _write_run_metadata()
    atexit.register(_write_run_metadata)
