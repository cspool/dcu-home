from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ANALYZER = ROOT / "scripts/perf_trace/analyze_qwen_hipprof_pmc.py"


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class BoundedFamilySupersetTest(unittest.TestCase):
    def test_extra_family_match_is_discarded_after_exact_join(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            analysis = root / "analysis"
            analysis.mkdir()
            write_csv(
                analysis / "kernels.csv",
                [
                    "kernel_id",
                    "source_db",
                    "config_key",
                    "pid",
                    "runtime_index",
                    "begin_ns",
                    "end_ns",
                    "duration_ns",
                    "kernel_name",
                ],
                [
                    {
                        "kernel_id": 1,
                        "source_db": "same.db",
                        "config_key": "cfg",
                        "pid": 7,
                        "runtime_index": 1,
                        "begin_ns": 10,
                        "end_ns": 11,
                        "duration_ns": 1,
                        "kernel_name": "unrelated",
                    },
                    {
                        "kernel_id": 2,
                        "source_db": "same.db",
                        "config_key": "cfg",
                        "pid": 7,
                        "runtime_index": 2,
                        "begin_ns": 20,
                        "end_ns": 21,
                        "duration_ns": 1,
                        "kernel_name": "family_kernel",
                    },
                    {
                        "kernel_id": 3,
                        "source_db": "same.db",
                        "config_key": "cfg",
                        "pid": 7,
                        "runtime_index": 3,
                        "begin_ns": 30,
                        "end_ns": 31,
                        "duration_ns": 1,
                        "kernel_name": "family_kernel",
                    },
                ],
            )
            write_csv(
                analysis / "strict_ownership.csv",
                [
                    "kind",
                    "kernel_id",
                    "marker",
                    "event_id",
                    "stage",
                    "kernel_name",
                    "kernel_family",
                ],
                [
                    {
                        "kind": "process",
                        "kernel_id": 2,
                        "marker": "pra.fx_process.input2_layer0.mlp",
                        "event_id": "input2_layer0",
                        "stage": "mlp",
                        "kernel_name": "family_kernel",
                        "kernel_family": "family_a",
                    },
                    {
                        "kind": "process",
                        "kernel_id": 3,
                        "marker": "pra.fx_process.input1_layer0.mlp",
                        "event_id": "input1_layer0",
                        "stage": "mlp",
                        "kernel_name": "family_kernel",
                        "kernel_family": "family_a",
                    },
                ],
            )
            write_csv(
                analysis / "process_kernel_launch_order.csv",
                [
                    "event_id",
                    "stage",
                    "kernel_id",
                    "kernel_launch_order_in_process",
                ],
                [
                    {
                        "event_id": "input2_layer0",
                        "stage": "mlp",
                        "kernel_id": 2,
                        "kernel_launch_order_in_process": 1,
                    },
                    {
                        "event_id": "input1_layer0",
                        "stage": "mlp",
                        "kernel_id": 3,
                        "kernel_launch_order_in_process": 1,
                    },
                ],
            )
            selection = root / "selection.csv"
            write_csv(
                selection,
                [
                    "event_id",
                    "stage",
                    "matched_kernel_family",
                    "targeted_eligible",
                ],
                [
                    {
                        "event_id": "input1_layer0",
                        "stage": "mlp",
                        "matched_kernel_family": "family_a",
                        "targeted_eligible": "true",
                    }
                ],
            )
            metrics = root / "metrics.txt"
            metrics.write_text(
                '\n'.join(
                    [
                        'kernel-name:"family_kernel"',
                        "Process Id  7",
                        "Kernel Dispatch Index  10",
                        "Kernel Time  1 us",
                        "",
                        'kernel-name:"family_kernel"',
                        "Process Id  7",
                        "Kernel Dispatch Index  11",
                        "Kernel Time  2 us",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    "python3",
                    str(ANALYZER),
                    "--analysis-dir",
                    str(analysis),
                    "--metrics-file",
                    str(metrics),
                    "--selection-plan",
                    str(selection),
                    "--kind",
                    "pmc",
                    "--collection-policy",
                    "bounded-family-superset",
                    "--kernel-name-filter",
                    "family_",
                    "--capture-batch-id",
                    "batch-1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            summary = json.loads(
                (analysis / "hardware_metric_summary.json").read_text()
            )
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(summary["trace_total_kernel_count"], 3)
            self.assertEqual(summary["trace_candidate_kernel_count"], 2)
            self.assertEqual(summary["trace_excluded_kernel_count"], 1)
            self.assertEqual(summary["exact_name_order_matches"], 2)
            self.assertEqual(summary["strict_owned_metric_rows"], 1)
            self.assertEqual(summary["discarded_superset_match_count"], 1)
            self.assertEqual(summary["selected_exact_attribution_rate"], 1.0)
            self.assertTrue(summary["final_process_family_attribution_exact"])

            with (analysis / "hardware_kernel_metrics.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                output = list(csv.DictReader(handle))
            self.assertEqual(len(output), 1)
            self.assertEqual(output[0]["event_id"], "input1_layer0")


if __name__ == "__main__":
    unittest.main()
