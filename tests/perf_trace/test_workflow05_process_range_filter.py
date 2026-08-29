from __future__ import annotations

import builtins
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


PROFILE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts/perf_trace/profile_qwen_same_input_layer.py"
)
SPEC = importlib.util.spec_from_file_location("workflow05_profile", PROFILE_PATH)
assert SPEC is not None and SPEC.loader is not None
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)


class _FakeRange:
    def __init__(self, names: list[str], name: str) -> None:
        self.names = names
        self.name = name

    def __enter__(self) -> None:
        self.names.append(self.name)

    def __exit__(self, *_args: object) -> None:
        return None


class ProcessRangeFilterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.emitted: list[str] = []
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                nvtx=types.SimpleNamespace(
                    range=lambda name: _FakeRange(self.emitted, name)
                )
            )
        )
        self.torch_patch = mock.patch.dict(sys.modules, {"torch": fake_torch})
        self.torch_patch.start()

    def tearDown(self) -> None:
        self.torch_patch.stop()

    def test_exact_filter_emits_only_planned_process_and_fragment(self) -> None:
        event = "input5_layer6"
        targets = {
            f"pra.fx_process.{event}.qkv_projection",
            (
                f"pra.fx_process.{event}.output_projection."
                "part01_mixer_out_proj"
            ),
        }
        recorder = PROFILE.ProcessRangeRecorder(
            targets={event}, exact_range_targets=targets
        )
        with recorder.activate(event, "linear_attention"):
            with recorder.range("input_rmsnorm"):
                pass
            with recorder.range("qkv_projection"):
                pass
            with recorder.range(
                "output_projection", "part01_mixer_out_proj"
            ):
                pass
        recorder.assert_complete()
        self.assertEqual(set(self.emitted), targets)
        self.assertEqual(recorder.expected_ranges, targets)

    def test_unselected_range_is_a_noop_without_importing_torch(self) -> None:
        event = "input1_layer7"
        target = f"pra.fx_process.{event}.qkv_projection"
        recorder = PROFILE.ProcessRangeRecorder(
            targets={event}, exact_range_targets={target}
        )
        original_import = builtins.__import__

        def guarded_import(name: str, *args: object, **kwargs: object):
            if name == "torch":
                raise AssertionError("unselected range imported torch")
            return original_import(name, *args, **kwargs)

        with recorder.activate(event, "linear_attention"):
            with mock.patch("builtins.__import__", side_effect=guarded_import):
                with recorder.range("mlp"):
                    pass
        self.assertEqual(self.emitted, [])

    def test_legacy_event_filter_keeps_all_expected_ranges(self) -> None:
        event = "input2_layer3"
        recorder = PROFILE.ProcessRangeRecorder(targets={event})
        self.assertEqual(
            len(recorder.expected_names(event, "full_attention")), 10
        )

    def test_exact_target_event_set_must_match_event_filter(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "event and exact"):
            PROFILE.ProcessRangeRecorder(
                targets={"input1_layer0"},
                exact_range_targets={
                    "pra.fx_process.input2_layer0.qkv_projection"
                },
            )

    def test_runtime_event_preserves_workflow05_occurrence_identity(self) -> None:
        recorder = PROFILE.LayerRangeRecorder(
            contract_id="child-contract",
            contract_sha256="a" * 64,
            event_path=Path("unused-events.jsonl"),
            expected_layers=64,
        )

        event = recorder._begin_layer(0, 1, "linear_attention")

        self.assertEqual(event["forward_id"], 1)
        self.assertEqual(event["occurrence"], 1)
        self.assertEqual(event["source_range_occurrence"], 0)
        self.assertTrue(event["occurrence_key"].endswith("occurrence=1"))

    def test_large_target_transport_reads_one_target_per_line(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            target_file = Path(raw_root) / "targets.txt"
            target_file.write_text(
                "input1_layer0\ninput1_layer1\n",
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {
                    "TEST_TARGETS": "input1_layer2",
                    "TEST_TARGETS_FILE": str(target_file),
                },
                clear=False,
            ):
                values, provenance = PROFILE._parse_target_set(
                    "TEST_TARGETS", "TEST_TARGETS_FILE"
                )
            self.assertEqual(
                values,
                {"input1_layer0", "input1_layer1", "input1_layer2"},
            )
            self.assertEqual(provenance["file_count"], 2)
            self.assertEqual(provenance["total_count"], 3)
            self.assertEqual(len(provenance["file_sha256"]), 64)

    def test_live_sidecar_window_arms_and_stops_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            ready = root / "ready.json"
            ready.write_text(
                json.dumps({"status": "ready_waiting_for_arm"}) + "\n",
                encoding="utf-8",
            )
            paths = {
                "PRA_BACKEND_LIVE_UTIL_READY_FILE": str(ready),
                "PRA_BACKEND_LIVE_UTIL_ARM_FILE": str(root / "arm"),
                "PRA_BACKEND_LIVE_UTIL_STOP_FILE": str(root / "stop"),
                "PRA_BACKEND_LIVE_UTIL_SAMPLES_FILE": str(root / "samples.jsonl"),
                "PRA_BACKEND_LIVE_UTIL_SUMMARY_FILE": str(root / "summary.json"),
            }
            with mock.patch.dict(os.environ, paths, clear=False):
                with PROFILE._live_utilization_window() as state:
                    self.assertTrue(state["enabled"])
                    self.assertTrue((root / "arm").exists())
                    self.assertFalse((root / "stop").exists())
            self.assertTrue((root / "stop").exists())


if __name__ == "__main__":
    unittest.main()
