"""Runtime monkey patches for algorithm-level vLLM inference traces.

The patches record small metadata summaries at semantic runtime boundaries:
engine iteration, scheduler decisions, model execution, batch construction,
attention backend calls, KV cache allocation, sampling, and output updates.
No tensor contents are copied from device memory.
"""

from __future__ import annotations

import functools
import json
import os
import sys
import threading
import time
import traceback
from collections import Counter
from typing import Any


_PATCHED = False
_STATE = threading.local()
_WRITE_LOCK = threading.Lock()
_EVENT_INDEX = 0


def _enabled() -> bool:
    return os.environ.get("VLLM_TRACE_PATCH_ENABLE") == "1"


def _trace_dir() -> str:
    return os.environ.get("VLLM_TRACE_PATCH_DIR") or os.getcwd()


def _arm_file() -> str | None:
    return os.environ.get("VLLM_TRACE_PATCH_ARM_FILE")


def _is_armed(event_type: str) -> bool:
    if event_type in {"patch_loaded", "patch_error"}:
        return True
    arm = _arm_file()
    return not arm or os.path.exists(arm)


def _event_path() -> str:
    return os.path.join(_trace_dir(), f"events.{os.getpid()}.jsonl")


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_len(value: Any) -> int | None:
    try:
        return len(value)
    except Exception:
        return None


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(x) for x in tuple(shape)]
    except Exception:
        return None


def _tensor_summary(value: Any) -> dict[str, Any] | None:
    shape = _shape(value)
    if shape is None:
        return None
    out: dict[str, Any] = {"shape": shape}
    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)
    if dtype is not None:
        out["dtype"] = str(dtype)
    if device is not None:
        out["device"] = str(device)
    return out


def _small_list(value: Any, limit: int = 16) -> dict[str, Any] | list[Any] | None:
    if value is None:
        return None
    try:
        if hasattr(value, "tolist"):
            value = value.tolist()
        else:
            value = list(value)
    except Exception:
        return None
    if len(value) <= limit:
        return value
    return {"len": len(value), "head": value[: limit // 2], "tail": value[-limit // 2 :]}


def _block_counts(blocks: Any) -> list[int] | None:
    try:
        raw_blocks = blocks.blocks
    except Exception:
        raw_blocks = blocks
    try:
        return [len(group) for group in raw_blocks]
    except Exception:
        return None


def _block_id_counts(block_ids: Any) -> list[int] | None:
    try:
        return [len(group) for group in block_ids]
    except Exception:
        return None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "name"):
        try:
            return str(value.name)
        except Exception:
            pass
    return str(value)


def _write_event(event_type: str, **payload: Any) -> None:
    global _EVENT_INDEX
    if not _enabled() or not _is_armed(event_type):
        return
    try:
        os.makedirs(_trace_dir(), exist_ok=True)
        with _WRITE_LOCK:
            _EVENT_INDEX += 1
            event = {
                "event_index": _EVENT_INDEX,
                "event_type": event_type,
                "ts_ns": time.monotonic_ns(),
                "pid": os.getpid(),
                "ppid": os.getppid(),
                "context": os.environ.get("VLLM_TRACE_PATCH_CONTEXT"),
                "argv0": os.path.basename(sys.argv[0]) if sys.argv else None,
                **payload,
            }
            line = json.dumps(_jsonable(event), separators=(",", ":"), sort_keys=True)
            with open(_event_path(), "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except BaseException as exc:  # pragma: no cover - tracing must be best effort.
        try:
            sys.stderr.write(f"[vllm-trace-patch] write failed: {exc!r}\n")
        except Exception:
            pass


def _next_id(obj: Any, attr: str) -> int:
    value = getattr(obj, attr, 0) + 1
    setattr(obj, attr, value)
    return value


def _current_step_id() -> int | None:
    return getattr(_STATE, "engine_step_id", None)


def _current_forward_id() -> int | None:
    return getattr(_STATE, "forward_id", None)


def _set_current_step_id(value: int | None) -> None:
    _STATE.engine_step_id = value


def _set_current_forward_id(value: int | None) -> None:
    _STATE.forward_id = value


def _request_summary(req: Any, scheduled_tokens: int | None = None) -> dict[str, Any]:
    num_computed_after = _safe_int(getattr(req, "num_computed_tokens", None))
    num_computed_before = (
        num_computed_after - scheduled_tokens
        if num_computed_after is not None and scheduled_tokens is not None
        else None
    )
    prompt_len = _safe_int(getattr(req, "num_prompt_tokens", None))
    output_len = _safe_int(getattr(req, "num_output_tokens", None))
    total_tokens = _safe_int(getattr(req, "num_tokens", None))
    if prompt_len is not None and num_computed_before is not None:
        if num_computed_before < prompt_len:
            phase = "prefill_chunk"
        elif scheduled_tokens == 1:
            phase = "decode"
        else:
            phase = "decode_or_spec"
    else:
        phase = "unknown"
    return {
        "request_id": getattr(req, "request_id", None),
        "phase": phase,
        "scheduled_tokens": scheduled_tokens,
        "prompt_len": prompt_len,
        "num_computed_before": num_computed_before,
        "num_computed_after": num_computed_after,
        "num_output_tokens": output_len,
        "num_tokens": total_tokens,
        "max_tokens": _safe_int(getattr(req, "max_tokens", None)),
        "status": str(getattr(req, "status", None)),
        "is_prefill_chunk": bool(getattr(req, "is_prefill_chunk", False)),
        "num_cached_tokens": _safe_int(getattr(req, "num_cached_tokens", None)),
    }


def _phase_from_requests(requests: list[dict[str, Any]]) -> str:
    phases = {req.get("phase") for req in requests if req.get("phase")}
    if not phases:
        return "empty"
    if len(phases) == 1:
        return str(next(iter(phases)))
    return "mixed"


def _scheduler_output_summary(scheduler: Any, output: Any) -> dict[str, Any]:
    num_scheduled = dict(getattr(output, "num_scheduled_tokens", {}) or {})
    requests = []
    for req_id, scheduled in num_scheduled.items():
        req = getattr(scheduler, "requests", {}).get(req_id)
        if req is None:
            requests.append(
                {
                    "request_id": req_id,
                    "phase": "missing",
                    "scheduled_tokens": _safe_int(scheduled),
                }
            )
        else:
            requests.append(_request_summary(req, _safe_int(scheduled)))
    return {
        "phase": _phase_from_requests(requests),
        "num_scheduled_tokens": {str(k): _safe_int(v) for k, v in num_scheduled.items()},
        "total_num_scheduled_tokens": _safe_int(
            getattr(output, "total_num_scheduled_tokens", None)
        ),
        "scheduled_new_req_count": _safe_len(getattr(output, "scheduled_new_reqs", None)),
        "scheduled_cached_req_count": _safe_len(
            getattr(getattr(output, "scheduled_cached_reqs", None), "req_ids", None)
        ),
        "finished_req_count": _safe_len(getattr(output, "finished_req_ids", None)),
        "preempted_req_count": _safe_len(getattr(output, "preempted_req_ids", None)),
        "running_count": _safe_len(getattr(scheduler, "running", None)),
        "waiting_count": _safe_len(getattr(scheduler, "waiting", None)),
        "num_common_prefix_blocks": _small_list(
            getattr(output, "num_common_prefix_blocks", None)
        ),
        "new_block_ids_to_zero_count": _safe_len(
            getattr(output, "new_block_ids_to_zero", None)
        ),
        "requests": requests,
    }


def _input_batch_summary(input_batch: Any) -> dict[str, Any]:
    return {
        "req_ids": list(getattr(input_batch, "req_ids", []) or []),
        "num_reqs": _safe_int(getattr(input_batch, "num_reqs", None)),
        "num_reqs_after_padding": _safe_int(
            getattr(input_batch, "num_reqs_after_padding", None)
        ),
        "num_tokens": _safe_int(getattr(input_batch, "num_tokens", None)),
        "num_tokens_after_padding": _safe_int(
            getattr(input_batch, "num_tokens_after_padding", None)
        ),
        "num_scheduled_tokens": _small_list(
            getattr(input_batch, "num_scheduled_tokens", None)
        ),
        "query_start_loc": _small_list(getattr(input_batch, "query_start_loc_np", None)),
        "cu_num_logits": _small_list(getattr(input_batch, "cu_num_logits_np", None)),
        "input_ids": _tensor_summary(getattr(input_batch, "input_ids", None)),
        "positions": _tensor_summary(getattr(input_batch, "positions", None)),
        "seq_lens": _tensor_summary(getattr(input_batch, "seq_lens", None)),
    }


def _v1_input_batch_summary(input_batch: Any, num_scheduled_tokens: Any = None) -> dict[str, Any]:
    return {
        "req_ids": list(getattr(input_batch, "req_ids", []) or []),
        "num_reqs": _safe_int(getattr(input_batch, "num_reqs", None)),
        "num_scheduled_tokens": _small_list(num_scheduled_tokens),
        "num_computed_tokens_cpu": _small_list(
            getattr(input_batch, "num_computed_tokens_cpu", None)
        ),
        "num_prompt_tokens": _small_list(getattr(input_batch, "num_prompt_tokens", None)),
        "num_tokens_no_spec": _small_list(getattr(input_batch, "num_tokens_no_spec", None)),
    }


def _attn_metadata_summary(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {"metadata": None}
    return {
        "metadata_cls": metadata.__class__.__name__,
        "num_actual_tokens": _safe_int(getattr(metadata, "num_actual_tokens", None)),
        "max_query_len": _safe_int(getattr(metadata, "max_query_len", None)),
        "max_seq_len": _safe_int(getattr(metadata, "max_seq_len", None)),
        "query_start_loc": _tensor_summary(getattr(metadata, "query_start_loc", None)),
        "seq_lens": _tensor_summary(getattr(metadata, "seq_lens", None)),
        "block_table": _tensor_summary(getattr(metadata, "block_table", None)),
        "slot_mapping": _tensor_summary(getattr(metadata, "slot_mapping", None)),
        "use_cascade": bool(getattr(metadata, "use_cascade", False)),
        "common_prefix_len": _safe_int(getattr(metadata, "common_prefix_len", None)),
    }


def _wrap_method(cls: Any, name: str, wrapper_factory: Any) -> None:
    original = getattr(cls, name, None)
    if original is None or getattr(original, "_vllm_trace_patched", False):
        return
    wrapped = wrapper_factory(original)
    wrapped._vllm_trace_patched = True
    setattr(cls, name, wrapped)


def _patch_engine() -> None:
    from vllm.v1.engine.core import EngineCore

    def step_wrapper(original: Any) -> Any:
        @functools.wraps(original)
        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            step_id = _next_id(self, "_trace_engine_step_id")
            previous = _current_step_id()
            _set_current_step_id(step_id)
            t0 = time.monotonic_ns()
            _write_event("engine_step_begin", engine_step_id=step_id)
            try:
                result = original(self, *args, **kwargs)
                outputs, model_executed = result
                output_count = 0
                finished_count = 0
                for eco in (outputs or {}).values():
                    eco_outputs = getattr(eco, "outputs", []) or []
                    output_count += len(eco_outputs)
                    finished_count += sum(1 for out in eco_outputs if getattr(out, "finished", False))
                _write_event(
                    "engine_step_end",
                    engine_step_id=step_id,
                    duration_ns=time.monotonic_ns() - t0,
                    model_executed=bool(model_executed),
                    output_count=output_count,
                    finished_output_count=finished_count,
                )
                return result
            finally:
                _set_current_step_id(previous)

        return wrapped

    _wrap_method(EngineCore, "step", step_wrapper)
    _wrap_method(EngineCore, "step_with_batch_queue", step_wrapper)


def _patch_scheduler() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    def schedule_wrapper(original: Any) -> Any:
        @functools.wraps(original)
        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            step_id = _current_step_id()
            t0 = time.monotonic_ns()
            output = original(self, *args, **kwargs)
            try:
                if step_id is None:
                    step_id = _next_id(self, "_trace_scheduler_step_id")
                setattr(output, "_trace_engine_step_id", step_id)
                summary = _scheduler_output_summary(self, output)
                _write_event(
                    "scheduler_step",
                    engine_step_id=step_id,
                    duration_ns=time.monotonic_ns() - t0,
                    **summary,
                )
            except BaseException:
                _write_event("patch_error", where="scheduler_step", traceback=traceback.format_exc())
            return output

        return wrapped

    def update_wrapper(original: Any) -> Any:
        @functools.wraps(original)
        def wrapped(self: Any, scheduler_output: Any, model_runner_output: Any, *args: Any, **kwargs: Any) -> Any:
            step_id = getattr(scheduler_output, "_trace_engine_step_id", _current_step_id())
            t0 = time.monotonic_ns()
            outputs = original(self, scheduler_output, model_runner_output, *args, **kwargs)
            try:
                output_summaries = []
                for client_index, eco in (outputs or {}).items():
                    for out in getattr(eco, "outputs", []) or []:
                        token_ids = getattr(out, "new_token_ids", []) or []
                        output_summaries.append(
                            {
                                "client_index": client_index,
                                "request_id": getattr(out, "request_id", None),
                                "new_token_count": len(token_ids),
                                "new_token_head": token_ids[:8],
                                "finish_reason": str(getattr(out, "finish_reason", None)),
                                "stop_reason": getattr(out, "stop_reason", None),
                                "finished": bool(getattr(out, "finished", False)),
                                "num_cached_tokens": _safe_int(
                                    getattr(out, "num_cached_tokens", None)
                                ),
                            }
                        )
                _write_event(
                    "scheduler_update_output",
                    engine_step_id=step_id,
                    duration_ns=time.monotonic_ns() - t0,
                    output_count=len(output_summaries),
                    outputs=output_summaries,
                )
            except BaseException:
                _write_event("patch_error", where="scheduler_update_output", traceback=traceback.format_exc())
            return outputs

        return wrapped

    _wrap_method(Scheduler, "schedule", schedule_wrapper)
    _wrap_method(Scheduler, "update_from_output", update_wrapper)


def _patch_model_runner() -> None:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    def execute_wrapper(original: Any) -> Any:
        @functools.wraps(original)
        def wrapped(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
            forward_id = _next_id(self, "_trace_forward_id")
            previous_forward = _current_forward_id()
            previous_step = _current_step_id()
            step_id = getattr(scheduler_output, "_trace_engine_step_id", previous_step)
            setattr(scheduler_output, "_trace_forward_id", forward_id)
            _set_current_forward_id(forward_id)
            _set_current_step_id(step_id)
            t0 = time.monotonic_ns()
            _write_event(
                "model_execute_begin",
                engine_step_id=step_id,
                forward_id=forward_id,
                total_num_scheduled_tokens=_safe_int(
                    getattr(scheduler_output, "total_num_scheduled_tokens", None)
                ),
                num_scheduled_tokens=dict(
                    getattr(scheduler_output, "num_scheduled_tokens", {}) or {}
                ),
            )
            try:
                result = original(self, scheduler_output, *args, **kwargs)
                _write_event(
                    "model_execute_end",
                    engine_step_id=step_id,
                    forward_id=forward_id,
                    duration_ns=time.monotonic_ns() - t0,
                    result_type=result.__class__.__name__ if result is not None else None,
                )
                return result
            finally:
                _set_current_forward_id(previous_forward)
                _set_current_step_id(previous_step)

        return wrapped

    def prepare_inputs_wrapper(original: Any) -> Any:
        @functools.wraps(original)
        def wrapped(self: Any, scheduler_output: Any, batch_desc: Any, *args: Any, **kwargs: Any) -> Any:
            input_batch = original(self, scheduler_output, batch_desc, *args, **kwargs)
            try:
                _write_event(
                    "batch_constructed",
                    engine_step_id=getattr(scheduler_output, "_trace_engine_step_id", _current_step_id()),
                    forward_id=getattr(scheduler_output, "_trace_forward_id", _current_forward_id()),
                    batch_desc={
                        "cg_mode": str(getattr(batch_desc, "cg_mode", None)),
                        "num_tokens": _safe_int(getattr(batch_desc, "num_tokens", None)),
                        "num_reqs": _safe_int(getattr(batch_desc, "num_reqs", None)),
                    },
                    **_input_batch_summary(input_batch),
                )
            except BaseException:
                _write_event("patch_error", where="batch_constructed", traceback=traceback.format_exc())
            return input_batch

        return wrapped

    def prepare_attn_wrapper(original: Any) -> Any:
        @functools.wraps(original)
        def wrapped(self: Any, input_batch: Any, *args: Any, **kwargs: Any) -> Any:
            result = original(self, input_batch, *args, **kwargs)
            try:
                block_tables, slot_mappings = result
                _write_event(
                    "attention_batch_metadata",
                    engine_step_id=_current_step_id(),
                    forward_id=_current_forward_id(),
                    req_ids=list(getattr(input_batch, "req_ids", []) or []),
                    block_tables=[_tensor_summary(x) for x in block_tables],
                    slot_mappings=_tensor_summary(slot_mappings),
                )
            except BaseException:
                _write_event("patch_error", where="attention_batch_metadata", traceback=traceback.format_exc())
            return result

        return wrapped

    def sample_wrapper(original: Any) -> Any:
        @functools.wraps(original)
        def wrapped(self: Any, hidden_states: Any, input_batch: Any, grammar_output: Any, *args: Any, **kwargs: Any) -> Any:
            t0 = time.monotonic_ns()
            result = original(self, hidden_states, input_batch, grammar_output, *args, **kwargs)
            try:
                sampler_output, num_sampled, num_rejected = result
                _write_event(
                    "model_sample",
                    engine_step_id=_current_step_id(),
                    forward_id=_current_forward_id(),
                    duration_ns=time.monotonic_ns() - t0,
                    req_ids=list(getattr(input_batch, "req_ids", []) or []),
                    hidden_states=_tensor_summary(hidden_states),
                    sampled_token_ids=_tensor_summary(
                        getattr(sampler_output, "sampled_token_ids", None)
                    ),
                    num_sampled=_tensor_summary(num_sampled),
                    num_rejected=_tensor_summary(num_rejected),
                )
            except BaseException:
                _write_event("patch_error", where="model_sample", traceback=traceback.format_exc())
            return result

        return wrapped

    def sample_tokens_wrapper(original: Any) -> Any:
        @functools.wraps(original)
        def wrapped(self: Any, grammar_output: Any, *args: Any, **kwargs: Any) -> Any:
            state = getattr(self, "execute_model_state", None)
            input_batch = getattr(state, "input_batch", None)
            t0 = time.monotonic_ns()
            result = original(self, grammar_output, *args, **kwargs)
            _write_event(
                "sample_tokens",
                engine_step_id=_current_step_id(),
                forward_id=_current_forward_id(),
                duration_ns=time.monotonic_ns() - t0,
                req_ids=list(getattr(input_batch, "req_ids", []) or []) if input_batch else [],
                result_type=result.__class__.__name__ if result is not None else None,
            )
            return result

        return wrapped

    _wrap_method(GPUModelRunner, "execute_model", execute_wrapper)
    _wrap_method(GPUModelRunner, "prepare_inputs", prepare_inputs_wrapper)
    _wrap_method(GPUModelRunner, "prepare_attn", prepare_attn_wrapper)
    _wrap_method(GPUModelRunner, "sample", sample_wrapper)
    _wrap_method(GPUModelRunner, "sample_tokens", sample_tokens_wrapper)


def _patch_v1_model_runner() -> None:
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception:
        return

    def execute_wrapper(original: Any) -> Any:
        @functools.wraps(original)
        def wrapped(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
            forward_id = _next_id(self, "_trace_forward_id")
            previous_forward = _current_forward_id()
            previous_step = _current_step_id()
            step_id = getattr(scheduler_output, "_trace_engine_step_id", previous_step)
            setattr(scheduler_output, "_trace_forward_id", forward_id)
            _set_current_forward_id(forward_id)
            _set_current_step_id(step_id)
            t0 = time.monotonic_ns()
            _write_event(
                "model_execute_begin",
                engine_step_id=step_id,
                forward_id=forward_id,
                runner="v1",
                total_num_scheduled_tokens=_safe_int(
                    getattr(scheduler_output, "total_num_scheduled_tokens", None)
                ),
                num_scheduled_tokens=dict(
                    getattr(scheduler_output, "num_scheduled_tokens", {}) or {}
                ),
            )
            try:
                result = original(self, scheduler_output, *args, **kwargs)
                _write_event(
                    "model_execute_end",
                    engine_step_id=step_id,
                    forward_id=forward_id,
                    runner="v1",
                    duration_ns=time.monotonic_ns() - t0,
                    result_type=result.__class__.__name__ if result is not None else None,
                    input_batch=_v1_input_batch_summary(getattr(self, "input_batch", None)),
                )
                return result
            finally:
                _set_current_forward_id(previous_forward)
                _set_current_step_id(previous_step)

        return wrapped

    def prepare_inputs_wrapper(original: Any) -> Any:
        @functools.wraps(original)
        def wrapped(self: Any, scheduler_output: Any, num_scheduled_tokens: Any, *args: Any, **kwargs: Any) -> Any:
            result = original(self, scheduler_output, num_scheduled_tokens, *args, **kwargs)
            try:
                _write_event(
                    "batch_constructed",
                    engine_step_id=getattr(scheduler_output, "_trace_engine_step_id", _current_step_id()),
                    forward_id=getattr(scheduler_output, "_trace_forward_id", _current_forward_id()),
                    runner="v1",
                    total_num_scheduled_tokens=_safe_int(
                        getattr(scheduler_output, "total_num_scheduled_tokens", None)
                    ),
                    **_v1_input_batch_summary(
                        getattr(self, "input_batch", None), num_scheduled_tokens
                    ),
                )
            except BaseException:
                _write_event("patch_error", where="v1_batch_constructed", traceback=traceback.format_exc())
            return result

        return wrapped

    def build_attn_wrapper(original: Any) -> Any:
        @functools.wraps(original)
        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = original(self, *args, **kwargs)
            try:
                attn_metadata = result[0] if isinstance(result, tuple) else result
                if isinstance(attn_metadata, dict):
                    layer_count = len(attn_metadata)
                    metadata_types = sorted(
                        {value.__class__.__name__ for value in attn_metadata.values()}
                    )
                elif isinstance(attn_metadata, list):
                    layer_count = sum(len(x) for x in attn_metadata if isinstance(x, dict))
                    metadata_types = sorted(
                        {
                            value.__class__.__name__
                            for item in attn_metadata
                            if isinstance(item, dict)
                            for value in item.values()
                        }
                    )
                else:
                    layer_count = None
                    metadata_types = [attn_metadata.__class__.__name__]
                _write_event(
                    "attention_batch_metadata",
                    engine_step_id=_current_step_id(),
                    forward_id=_current_forward_id(),
                    runner="v1",
                    layer_metadata_count=layer_count,
                    metadata_types=metadata_types,
                )
            except BaseException:
                _write_event("patch_error", where="v1_attention_batch_metadata", traceback=traceback.format_exc())
            return result

        return wrapped

    def sample_tokens_wrapper(original: Any) -> Any:
        @functools.wraps(original)
        def wrapped(self: Any, grammar_output: Any, *args: Any, **kwargs: Any) -> Any:
            state = getattr(self, "execute_model_state", None)
            scheduler_output = getattr(state, "scheduler_output", None)
            if scheduler_output is None and isinstance(state, tuple) and len(state) > 0:
                scheduler_output = state[0]
            step_id = getattr(scheduler_output, "_trace_engine_step_id", _current_step_id())
            forward_id = getattr(scheduler_output, "_trace_forward_id", _current_forward_id())
            previous_forward = _current_forward_id()
            previous_step = _current_step_id()
            _set_current_forward_id(forward_id)
            _set_current_step_id(step_id)
            t0 = time.monotonic_ns()
            try:
                result = original(self, grammar_output, *args, **kwargs)
                _write_event(
                    "sample_tokens",
                    engine_step_id=step_id,
                    forward_id=forward_id,
                    runner="v1",
                    duration_ns=time.monotonic_ns() - t0,
                    input_batch=_v1_input_batch_summary(getattr(self, "input_batch", None)),
                    result_type=result.__class__.__name__ if result is not None else None,
                )
                return result
            finally:
                _set_current_forward_id(previous_forward)
                _set_current_step_id(previous_step)

        return wrapped

    _wrap_method(GPUModelRunner, "execute_model", execute_wrapper)
    _wrap_method(GPUModelRunner, "_prepare_inputs", prepare_inputs_wrapper)
    _wrap_method(GPUModelRunner, "_build_attention_metadata", build_attn_wrapper)
    _wrap_method(GPUModelRunner, "sample_tokens", sample_tokens_wrapper)


def _patch_attention() -> None:
    def backend_wrapper(original: Any) -> Any:
        @functools.wraps(original)
        def wrapped(self: Any, layer: Any, query: Any, key: Any, value: Any, kv_cache: Any, attn_metadata: Any, *args: Any, **kwargs: Any) -> Any:
            t0 = time.monotonic_ns()
            try:
                _write_event(
                    "attention_forward_begin",
                    engine_step_id=_current_step_id(),
                    forward_id=_current_forward_id(),
                    backend=self.__class__.__name__,
                    layer_name=getattr(layer, "layer_name", None),
                    query=_tensor_summary(query),
                    key=_tensor_summary(key),
                    value=_tensor_summary(value),
                    kv_cache=_tensor_summary(kv_cache),
                    **_attn_metadata_summary(attn_metadata),
                )
            except BaseException:
                _write_event("patch_error", where="attention_forward_begin", traceback=traceback.format_exc())
            result = original(self, layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs)
            _write_event(
                "attention_forward_end",
                engine_step_id=_current_step_id(),
                forward_id=_current_forward_id(),
                backend=self.__class__.__name__,
                layer_name=getattr(layer, "layer_name", None),
                duration_ns=time.monotonic_ns() - t0,
                result=_tensor_summary(result),
            )
            return result

        return wrapped

    try:
        from vllm.v1.attention.backends.rocm_attn import RocmAttentionImpl

        _wrap_method(RocmAttentionImpl, "forward", backend_wrapper)
    except Exception:
        pass
    try:
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

        _wrap_method(TritonAttentionImpl, "forward", backend_wrapper)
    except Exception:
        pass


def _patch_kv_cache() -> None:
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    def get_computed_wrapper(original: Any) -> Any:
        @functools.wraps(original)
        def wrapped(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
            result = original(self, request, *args, **kwargs)
            try:
                blocks, num_tokens = result
                _write_event(
                    "kv_get_computed_blocks",
                    engine_step_id=_current_step_id(),
                    request_id=getattr(request, "request_id", None),
                    prompt_len=_safe_int(getattr(request, "num_prompt_tokens", None)),
                    num_tokens=_safe_int(getattr(request, "num_tokens", None)),
                    computed_tokens=_safe_int(num_tokens),
                    block_counts=_block_counts(blocks),
                    cache_usage=getattr(self, "usage", None),
                )
            except BaseException:
                _write_event("patch_error", where="kv_get_computed_blocks", traceback=traceback.format_exc())
            return result

        return wrapped

    def allocate_wrapper(original: Any) -> Any:
        @functools.wraps(original)
        def wrapped(self: Any, request: Any, num_new_tokens: int, *args: Any, **kwargs: Any) -> Any:
            free_before = None
            try:
                free_before = self.block_pool.get_num_free_blocks()
            except Exception:
                pass
            t0 = time.monotonic_ns()
            result = original(self, request, num_new_tokens, *args, **kwargs)
            _write_event(
                "kv_allocate_slots",
                engine_step_id=_current_step_id(),
                request_id=getattr(request, "request_id", None),
                duration_ns=time.monotonic_ns() - t0,
                num_new_tokens=_safe_int(num_new_tokens),
                num_new_computed_tokens=_safe_int(kwargs.get("num_new_computed_tokens", 0)),
                num_external_computed_tokens=_safe_int(kwargs.get("num_external_computed_tokens", 0)),
                num_lookahead_tokens=_safe_int(kwargs.get("num_lookahead_tokens", 0)),
                delay_cache_blocks=bool(kwargs.get("delay_cache_blocks", False)),
                allocated=result is not None,
                block_counts=_block_counts(result),
                free_blocks_before=free_before,
                free_blocks_after=(
                    self.block_pool.get_num_free_blocks()
                    if hasattr(self, "block_pool")
                    else None
                ),
                cache_usage=getattr(self, "usage", None),
            )
            return result

        return wrapped

    def take_new_wrapper(original: Any) -> Any:
        @functools.wraps(original)
        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            ids = original(self, *args, **kwargs)
            _write_event(
                "kv_take_new_block_ids",
                engine_step_id=_current_step_id(),
                count=len(ids or []),
                head=list(ids[:8]) if ids else [],
            )
            return ids

        return wrapped

    _wrap_method(KVCacheManager, "get_computed_blocks", get_computed_wrapper)
    _wrap_method(KVCacheManager, "allocate_slots", allocate_wrapper)
    _wrap_method(KVCacheManager, "take_new_block_ids", take_new_wrapper)


def _patch_sampler() -> None:
    def v2_call_wrapper(original: Any) -> Any:
        @functools.wraps(original)
        def wrapped(self: Any, logits: Any, input_batch: Any, *args: Any, **kwargs: Any) -> Any:
            t0 = time.monotonic_ns()
            result = original(self, logits, input_batch, *args, **kwargs)
            _write_event(
                "sampler_call",
                engine_step_id=_current_step_id(),
                forward_id=_current_forward_id(),
                duration_ns=time.monotonic_ns() - t0,
                req_ids=list(getattr(input_batch, "req_ids", []) or []),
                logits=_tensor_summary(logits),
                sampled_token_ids=_tensor_summary(getattr(result, "sampled_token_ids", None)),
                num_sampled=_tensor_summary(getattr(result, "num_sampled", None)),
            )
            return result

        return wrapped

    def v1_forward_wrapper(original: Any) -> Any:
        @functools.wraps(original)
        def wrapped(self: Any, logits: Any, *args: Any, **kwargs: Any) -> Any:
            t0 = time.monotonic_ns()
            result = original(self, logits, *args, **kwargs)
            _write_event(
                "sampler_call",
                engine_step_id=_current_step_id(),
                forward_id=_current_forward_id(),
                duration_ns=time.monotonic_ns() - t0,
                runner="v1",
                logits=_tensor_summary(logits),
                sampled_token_ids=_tensor_summary(getattr(result, "sampled_token_ids", None)),
            )
            return result

        return wrapped

    try:
        from vllm.v1.worker.gpu.sample.sampler import Sampler as V2Sampler

        _wrap_method(V2Sampler, "__call__", v2_call_wrapper)
    except Exception:
        pass

    try:
        from vllm.v1.sample.sampler import Sampler as V1Sampler

        _wrap_method(V1Sampler, "forward", v1_forward_wrapper)
    except Exception:
        pass


def _patch_qwen35() -> None:
    try:
        from vllm.model_executor.models.qwen3_5 import Qwen3_5GatedDeltaNet
    except Exception:
        return

    def forward_wrapper(original: Any) -> Any:
        @functools.wraps(original)
        def wrapped(self: Any, hidden_states: Any, output: Any, *args: Any, **kwargs: Any) -> Any:
            t0 = time.monotonic_ns()
            result = original(self, hidden_states, output, *args, **kwargs)
            _write_event(
                "qwen35_gdn_forward",
                engine_step_id=_current_step_id(),
                forward_id=_current_forward_id(),
                prefix=getattr(self, "prefix", None),
                duration_ns=time.monotonic_ns() - t0,
                hidden_states=_tensor_summary(hidden_states),
                output=_tensor_summary(output),
            )
            return result

        return wrapped

    _wrap_method(Qwen3_5GatedDeltaNet, "forward", forward_wrapper)


def apply_patches() -> None:
    global _PATCHED
    if _PATCHED or not _enabled():
        return
    _PATCHED = True
    errors = []
    for patch_fn in (
        _patch_engine,
        _patch_scheduler,
        _patch_model_runner,
        _patch_v1_model_runner,
        _patch_attention,
        _patch_kv_cache,
        _patch_sampler,
        _patch_qwen35,
    ):
        try:
            patch_fn()
        except BaseException:
            errors.append({"patch": patch_fn.__name__, "traceback": traceback.format_exc()})
    _write_event(
        "patch_loaded",
        patches=[
            "engine",
            "scheduler",
            "model_runner",
            "v1_model_runner",
            "attention",
            "kv_cache",
            "sampler",
            "qwen35",
        ],
        errors=errors,
    )
