from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPORTER = ROOT / "scripts/perf_trace/export_workflow05_perfetto_trace.py"
PROBE = ROOT / "scripts/perf_trace/probe_workflow05_open_source_trace.py"
LAUNCHER = ROOT / "scripts/perf_trace/generate_workflow05_perfetto_ui_launcher.py"


class Workflow05OpenSourceTraceTest(unittest.TestCase):
    def make_trace(self, root: Path) -> Path:
        source = root / "process.csv"
        with source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "stable_key",
                    "process",
                    "lane",
                    "begin_ns",
                    "end_ns",
                    "evidence",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "stable_key": "a",
                        "process": "qkv",
                        "lane": "host",
                        "begin_ns": 1_000,
                        "end_ns": 5_000,
                        "evidence": "observed",
                    },
                    {
                        "stable_key": "b",
                        "process": "mlp",
                        "lane": "host",
                        "begin_ns": 3_000,
                        "end_ns": 7_000,
                        "evidence": "observed",
                    },
                ]
            )
        spec = root / "spec.json"
        spec.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "clock": {
                        "domain": "workflow05_child_non_replay",
                        "origin_ns": 1_000,
                        "parent_clock_mergeable": False,
                    },
                    "tracks": [
                        {
                            "track_group": "observed_process",
                            "category": "workflow05.observed_process",
                            "source_csv": str(source),
                            "name_columns": ["process"],
                            "lane_columns": ["lane"],
                            "row_key_columns": ["stable_key"],
                            "args_columns": ["stable_key"],
                            "begin_column": "begin_ns",
                            "end_column": "end_ns",
                            "timestamp_unit": "ns",
                            "evidence_column": "evidence",
                            "timing_semantics_value": "observed_non_replay",
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        trace = root / "workflow05.json"
        subprocess.run(
            [
                sys.executable,
                str(EXPORTER),
                "--track-spec",
                str(spec),
                "--output-trace",
                str(trace),
                "--output-manifest",
                str(root / "export_manifest.json"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return trace

    def test_export_allocates_separate_lanes_for_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            trace = self.make_trace(root)
            payload = json.loads(trace.read_text())
            intervals = [event for event in payload["traceEvents"] if event["ph"] == "X"]
            self.assertEqual(len(intervals), 2)
            self.assertEqual(len({event["tid"] for event in intervals}), 2)
            self.assertTrue(
                all(event["args"]["evidence_class"] == "observed" for event in intervals)
            )
            manifest = json.loads((root / "export_manifest.json").read_text())
            self.assertEqual(manifest["dropped_source_row_count"], 0)
            self.assertFalse(manifest["perfetto_parse_verified"])

    def test_official_cli_pass_prevents_custom_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            trace = self.make_trace(root)
            fake = root / "trace_processor"
            fake.write_text(
                "#!/bin/sh\n"
                "if [ \"$1\" = \"--help\" ]; then\n"
                "  echo 'Commands: query interactive server'\n"
                "  exit 0\n"
                "fi\n"
                "if [ \"$1\" = \"query\" ]; then\n"
                "  echo 'table_name row_count slice 2 counter 0 track 3'\n"
                "  exit 0\n"
                "fi\n"
                "exit 2\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            manifest_path = root / "probe.json"
            subprocess.run(
                [
                    sys.executable,
                    str(PROBE),
                    "--trace",
                    str(trace),
                    "--output-manifest",
                    str(manifest_path),
                    "--attempt-id",
                    "test-cli-pass",
                    "--expected-format",
                    "chrome-json",
                    "--trace-processor-bin",
                    str(fake),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(
                manifest["selected_backend"], "perfetto_trace_processor_cli"
            )
            self.assertTrue(manifest["open_source_processor_reused"])
            self.assertFalse(manifest["custom_timeline_fallback_required"])

    def test_missing_official_parser_selects_labeled_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            trace = self.make_trace(root)
            manifest_path = root / "probe.json"
            environment = dict(os.environ)
            environment["PATH"] = "/path/that/does/not/exist"
            subprocess.run(
                [
                    sys.executable,
                    str(PROBE),
                    "--trace",
                    str(trace),
                    "--output-manifest",
                    str(manifest_path),
                    "--attempt-id",
                    "test-fallback",
                    "--expected-format",
                    "chrome-json",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(
                manifest["selected_backend"], "custom_plotly_timeline_fallback"
            )
            self.assertFalse(manifest["open_source_processor_reused"])
            self.assertTrue(manifest["fallback_label_required"])
            self.assertEqual(
                manifest["trace"]["structural_check"]["status"], "pass"
            )

    def test_official_cli_must_pass_native_trace_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            trace = self.make_trace(root)
            expectations = {
                "required_category_min_counts": {"HIP": 2, "HIPOPS": 1, "HIPTX": 1},
                "required_slice_names": ["pra.fx_process.input1_layer0.mlp"],
                "required_track_name_substrings": ["Runtime API", "Stream on Device"],
                "required_arg_key_suffixes": ["BeginNs", "EndNs", "index"],
                "minimum_flow_count": 1,
            }
            native_manifest = root / "native.json"
            native_manifest.write_text(
                json.dumps(
                    {
                        "windows": [
                            {
                                "selection_rank": 1,
                                "stable_key": "test/1/0/1/mlp",
                                "attempts": [
                                    {
                                        "format": "chrome-json",
                                        "status": "pass",
                                        "trace_files": [
                                            {
                                                "path": str(trace),
                                                "sha256": hashlib.sha256(
                                                    trace.read_bytes()
                                                ).hexdigest(),
                                                "semantic_expectations": expectations,
                                            }
                                        ],
                                    }
                                ],
                            }
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            fake = root / "trace_processor"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import sys\n"
                "if sys.argv[1] == '--help':\n"
                " print('Commands: query interactive server')\n"
                "elif sys.argv[1] == 'query' and 'WF05|' in sys.argv[3]:\n"
                " print('WF05|category|HIP|2')\n"
                " print('WF05|category|HIPOPS|1')\n"
                " print('WF05|category|HIPTX|1')\n"
                " print('WF05|flow|all|1')\n"
                " print('WF05|track|Runtime API|1')\n"
                " print('WF05|track|Stream on Device 1|1')\n"
                " print('WF05|arg_key|debug.BeginNs|3')\n"
                " print('WF05|arg_key|debug.EndNs|3')\n"
                " print('WF05|arg_key|debug.index|3')\n"
                " print('WF05|hiptx_name|pra.fx_process.input1_layer0.mlp|1')\n"
                "elif sys.argv[1] == 'query':\n"
                " print('table_name row_count slice 4 counter 0 track 3')\n"
                "else:\n"
                " raise SystemExit(2)\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            output = root / "probe-semantic.json"
            subprocess.run(
                [
                    sys.executable,
                    str(PROBE),
                    "--trace",
                    str(trace),
                    "--output-manifest",
                    str(output),
                    "--attempt-id",
                    "semantic-pass",
                    "--expected-format",
                    "chrome-json",
                    "--trace-processor-bin",
                    str(fake),
                    "--native-export-manifest",
                    str(native_manifest),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads(output.read_text())
            self.assertEqual(
                manifest["selected_backend"], "perfetto_trace_processor_cli"
            )
            cli = manifest["attempts"][1]
            self.assertEqual(cli["semantic_validation"]["status"], "pass")
            self.assertTrue(all(cli["semantic_validation"]["checks"].values()))

    def test_launcher_uses_official_ui_message_interface_only(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            trace = self.make_trace(root)
            attempts = root / "attempts.json"
            ui_url = "http://127.0.0.1:9001/#!/?mode=embedded"
            environment = dict(os.environ)
            environment["PATH"] = "/path/that/does/not/exist"
            subprocess.run(
                [
                    sys.executable,
                    str(PROBE),
                    "--trace",
                    str(trace),
                    "--output-manifest",
                    str(attempts),
                    "--attempt-id",
                    "launcher-test",
                    "--expected-format",
                    "chrome-json",
                    "--perfetto-ui-url",
                    ui_url,
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            output_html = root / "SELECTED_PROCESS_TRACE_PERFETTO.html"
            output_manifest = root / "launcher_manifest.json"
            subprocess.run(
                [
                    sys.executable,
                    str(LAUNCHER),
                    "--trace",
                    str(trace),
                    "--attempt-manifest",
                    str(attempts),
                    "--output-html",
                    str(output_html),
                    "--output-manifest",
                    str(output_manifest),
                    "--perfetto-ui-url",
                    ui_url,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            page = output_html.read_text()
            self.assertIn("postMessage('PING'", page)
            self.assertIn("event.data === 'PONG'", page)
            self.assertIn("response.arrayBuffer()", page)
            self.assertIn("http://127.0.0.1:9001", page)
            self.assertNotIn("Plotly", page)
            manifest = json.loads(output_manifest.read_text())
            self.assertFalse(manifest["custom_timeline_renderer_present"])
            self.assertEqual(
                manifest["browser_validation_status"],
                "pending_runtime_handshake_and_ui_parse",
            )

    def test_managed_download_wrapper_is_not_invoked_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            trace = self.make_trace(root)
            sentinel = root / "invoked"
            wrapper = root / "trace_processor"
            wrapper.write_text(
                "#!/bin/sh\n"
                "# TRACE_PROCESSOR_SHELL_MANIFEST perfetto-luci-artifacts\n"
                "# download_or_get_cached\n"
                f"touch {sentinel}\n"
                "exit 9\n",
                encoding="utf-8",
            )
            wrapper.chmod(0o755)
            output = root / "probe-wrapper.json"
            subprocess.run(
                [
                    sys.executable,
                    str(PROBE),
                    "--trace",
                    str(trace),
                    "--output-manifest",
                    str(output),
                    "--attempt-id",
                    "managed-wrapper-skip",
                    "--expected-format",
                    "chrome-json",
                    "--trace-processor-bin",
                    str(wrapper),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertFalse(sentinel.exists())
            manifest = json.loads(output.read_text())
            cli = next(
                item
                for item in manifest["attempts"]
                if item["backend"] == "perfetto_trace_processor_cli"
            )
            self.assertEqual(cli["status"], "skipped")
            self.assertTrue(cli["managed_download_wrapper"])


if __name__ == "__main__":
    unittest.main()
