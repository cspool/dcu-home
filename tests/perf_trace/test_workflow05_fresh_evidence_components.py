from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTER = REPO_ROOT / "scripts/perf_trace/build_fresh_run_dependency_adapter.py"
MODEL = REPO_ROOT / "scripts/perf_trace/build_traffic_resource_model.py"
ANALYZER = REPO_ROOT / "scripts/perf_trace/analyze_fresh_e2e_timeline.py"
VISUALIZER = REPO_ROOT / "scripts/perf_trace/generate_fresh_e2e_visualization.py"
LINEAGE_BUILDER = REPO_ROOT / "scripts/perf_trace/build_fresh_run_lineage_manifest.py"
LINEAGE_ID = "fresh-run:test-run:fresh-contract"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class FreshEvidenceComponentsTest(unittest.TestCase):
    def test_stage_tools_do_not_restore_cross_stage_source_equality_gates(self) -> None:
        tool_names = (
            "prepare_qwen_r02_fresh_fx_plan.py",
            "capture_qwen_fresh_run_fx.py",
            "audit_qwen_r02_fresh_fx.py",
            "generate_qwen_fx_process_handoff.py",
            "prepare_qwen_dcu_hardware_plan.py",
            "generate_segmented_process_attribution.py",
            "audit_qwen_segmented_process_attribution.py",
        )
        forbidden = (
            "current source revision differs from R01",
            "R01 source revision differs from the current checkout",
            "current revision differs from R01",
            "R01 contract revision differs from current source",
            "R02 source revision differs from R01",
            "R03 FX structural source revision differs from R01",
            'require_equal("source revision", source_revision, git_revision)',
        )
        for name in tool_names:
            text = (REPO_ROOT / "scripts" / "perf_trace" / name).read_text(
                encoding="utf-8"
            )
            with self.subTest(tool=name):
                for fragment in forbidden:
                    self.assertNotIn(fragment, text)
                self.assertIn("source_hash_equality_required", text)

        r05_generator = (
            REPO_ROOT / "scripts/perf_trace/generate_segmented_process_attribution.py"
        ).read_text(encoding="utf-8")
        r05_auditor = (
            REPO_ROOT / "scripts/perf_trace/audit_qwen_segmented_process_attribution.py"
        ).read_text(encoding="utf-8")
        for text in (r05_generator, r05_auditor):
            self.assertIn("stage_source_revisions", text)
            self.assertIn("r05_source_state", text)
            self.assertIn("r01_source_revision", text)
        self.assertIn('parser.add_argument("--r04-handoff"', r05_generator)
        self.assertIn('parser.add_argument("--source-root"', r05_generator)

    def test_deprecated_current_child_entry_points_only_forward_to_fresh_tools(
        self,
    ) -> None:
        wrappers = {
            "capture_qwen_current_child_fx.py": "capture_qwen_fresh_run_fx",
            "build_current_child_dependency_adapter.py": (
                "build_fresh_run_dependency_adapter"
            ),
        }
        for filename, module in wrappers.items():
            text = (REPO_ROOT / "scripts/perf_trace" / filename).read_text(
                encoding="utf-8"
            )
            with self.subTest(wrapper=filename):
                self.assertIn("Deprecated compatibility entry point", text)
                self.assertIn(f"from {module} import main", text)
                self.assertLess(len(text.splitlines()), 12)

    def test_lineage_builder_accepts_source_changes_in_one_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "project"
            run_root = project / "perf_trace" / "runtime" / "fresh" / "test-run"
            handoff_root = run_root / "handoffs"
            handoff_root.mkdir(parents=True)
            handoffs = []
            for index in range(1, 6):
                goal = f"R{index:02d}"
                handoff = handoff_root / f"{goal}.json"
                _write_json(
                    handoff,
                    {
                        "runtime_goal": goal,
                        "status": "complete",
                        "source_revision": f"different-stage-revision-{index}",
                    },
                )
                handoffs.append(
                    {
                        "source_goal": goal,
                        "path": str(handoff),
                        "sha256": _sha256(handoff),
                        "payload": json.loads(handoff.read_text(encoding="utf-8")),
                    }
                )
            ledger = run_root / "runtime_handoff_ledger.json"
            _write_json(
                ledger,
                {
                    "schema_version": 1,
                    "branch": "fresh",
                    "run_id": "test-run",
                    "handoffs": handoffs,
                },
            )
            contract = run_root / "artifacts" / "R01" / "contract.json"
            _write_json(contract, {"contract_id": "fresh-contract"})
            staging = run_root / "artifacts" / "R06" / "staging"
            staging.mkdir(parents=True)
            events = staging / "events.txt"
            ranges = staging / "ranges.txt"
            hardware = staging / "hardware.csv"
            events.write_text("input0_layer0\n", encoding="utf-8")
            ranges.write_text(
                "pra.fx_process.input0_layer0.qkv_projection\n",
                encoding="utf-8",
            )
            hardware.write_text("kernel_family\ngemm\n", encoding="utf-8")
            output = run_root / "artifacts" / "R06" / "result"

            subprocess.run(
                [
                    sys.executable,
                    str(LINEAGE_BUILDER),
                    "--project-root",
                    str(project),
                    "--runtime-root",
                    str(run_root),
                    "--branch",
                    "fresh",
                    "--run-id",
                    "test-run",
                    "--ledger",
                    str(ledger),
                    "--semantic-contract",
                    str(contract),
                    "--event-targets",
                    str(events),
                    "--range-targets",
                    str(ranges),
                    "--hardware-plan",
                    str(hardware),
                    "--output-dir",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            lineage = json.loads(
                (output / "fresh_run_lineage_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(lineage["status"], "PASS")
            self.assertFalse(lineage["source_hash_equality_required"])
            self.assertEqual(lineage["upstream_goals"], [
                "R01", "R02", "R03", "R04", "R05"
            ])

    def _build_adapter_fixture(self, root: Path) -> Path:
        revision = "fresh-run-revision"
        lineage = root / "lineage.json"
        _write_json(
            lineage,
            {
                "schema_version": 1,
                "status": "PASS",
                "lineage_id": LINEAGE_ID,
                "semantic_contract_id": "fresh-contract",
                "evidence_source_policy": "current_run_only",
                "source_change_policy": "stage_trace_instrumentation_allowed",
                "source_hash_equality_required": False,
            },
        )
        contract = root / "measurement_contract.json"
        _write_json(
            contract,
            {
                "contract_id": "fresh-contract",
                "contract_sha256": "a" * 64,
                "source": {"revision": revision},
            },
        )
        annotations = root / "annotations.csv"
        markers = [
            "pra.fx_process.input0_layer0.input_rmsnorm",
            "pra.fx_process.input0_layer0.qkv_projection",
            (
                "pra.fx_process.input0_layer0.output_projection."
                "part01_mixer_out_proj"
            ),
            "pra.fx_process.input0_layer0.mlp",
        ]
        _write_csv(
            annotations,
            ["kind", "message", "event_id", "stage", "phase", "workload_type"],
            [
                {
                    "kind": "process",
                    "message": marker,
                    "event_id": "input0_layer0",
                    "stage": marker.removeprefix(
                        "pra.fx_process.input0_layer0."
                    ),
                    "phase": "prefill",
                    "workload_type": "linear_attention",
                }
                for marker in markers
            ],
        )
        runtime = root / "runtime_calls.csv"
        _write_csv(runtime, ["process_owner"], [{"process_owner": markers[1]}])
        ownership = root / "strict_ownership.csv"
        _write_csv(
            ownership,
            ["kind", "marker"],
            [{"kind": "process", "marker": markers[1]}],
        )
        assignments = root / "template_assignments.csv"
        _write_csv(
            assignments,
            ["event_id", "template_event_id"],
            [{"event_id": "input0_layer0", "template_event_id": "input0_layer0"}],
        )

        fx_dir = root / "fx" / "input0_layer0"
        reconstruction = fx_dir / "fx_trace_reconstruction.json"
        _write_json(
            reconstruction,
            {
                "event_identity": {
                    "phase": "prefill",
                    "layer_type": "linear_attention",
                },
                "nodes": [
                    {
                        "name": "normalized",
                        "op": "call_function",
                        "target": "aten.rms_norm",
                        "process_stage": "input_rmsnorm",
                        "shape": [1, 4, 8],
                        "dtype": "torch.bfloat16",
                        "args": [],
                    },
                    {
                        "name": "qkv",
                        "op": "call_module",
                        "target": "qkv_proj",
                        "process_stage": "qkv_projection",
                        "shape": [1, 4, 24],
                        "dtype": "torch.bfloat16",
                        "args": ["normalized"],
                    },
                    {
                        "name": "projected",
                        "op": "call_module",
                        "target": "out_proj",
                        "process_stage": "output_projection",
                        "shape": [1, 4, 8],
                        "dtype": "torch.bfloat16",
                        "args": ["qkv"],
                    },
                    {
                        "name": "mlp_out",
                        "op": "call_function",
                        "target": "opaque.fused_mlp",
                        "process_stage": "mlp",
                        "shape": [1, 4, 8],
                        "dtype": "torch.bfloat16",
                        "args": ["projected"],
                    },
                ],
                "evidence_guards": {"opaque_custom_ops": ["opaque.fused_mlp"]},
            },
        )
        _write_json(fx_dir / "fx_trace_metadata.json", {"source_revision": revision})
        manifest = root / "fx_manifest.json"
        _write_json(
            manifest,
            {
                "results": [
                    {
                        "status": "ok",
                        "event_id": "input0_layer0",
                        "json": {
                            "path": str(reconstruction),
                            "sha256": _sha256(reconstruction),
                        },
                    }
                ]
            },
        )

        adapter_dir = root / "adapter"
        subprocess.run(
            [
                sys.executable,
                str(ADAPTER),
                "--lineage-manifest",
                str(lineage),
                "--measurement-contract",
                str(contract),
                "--annotations",
                str(annotations),
                "--runtime-calls",
                str(runtime),
                "--strict-ownership",
                str(ownership),
                "--template-assignments",
                str(assignments),
                "--fx-manifest",
                str(manifest),
                "--stage-source-revision",
                revision,
                "--output-dir",
                str(adapter_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return adapter_dir / "fresh_run_dependency_adapter.json"

    def _build_model_fixture(self, root: Path, adapter_path: Path) -> Path:
        hardware = root / "hardware.csv"
        _write_csv(
            hardware,
            [
                "event_id", "stage", "matched_kernel_family", "work_group_size",
                "VGPR_count", "SGPR_count", "shared_memory_size_bytes",
                "theoretical_occupancy_upper_bound_pct", "hardware_evidence_class",
            ],
            [{
                "event_id": "input0_layer0", "stage": "qkv_projection",
                "matched_kernel_family": "gemm", "work_group_size": 256,
                "VGPR_count": 64, "SGPR_count": 32,
                "shared_memory_size_bytes": 0,
                "theoretical_occupancy_upper_bound_pct": 100,
                "hardware_evidence_class": "replay_projected_current_family",
            }],
        )
        capabilities = root / "device_capabilities.json"
        _write_json(
            capabilities,
            {
                "architecture": "gfx936", "physical_device_id": 1,
                "wave_size": 64, "wave_limit": 40, "thread_limit": 2560,
                "vgpr_resource": 196608, "shared_memory_bytes": 65536,
                "sources": [{"kind": "current_probe", "sha256": "b" * 64}],
            },
        )
        output_dir = root / "model"
        subprocess.run(
            [sys.executable, str(MODEL), "--dependency-adapter", str(adapter_path),
             "--lineage-manifest", str(root / "lineage.json"),
             "--hardware-metrics", str(hardware), "--device-capabilities",
             str(capabilities), "--output-dir", str(output_dir)],
            check=True, capture_output=True, text=True,
        )
        return output_dir / "traffic_resource_model.json"

    def test_adapter_preserves_verified_and_opaque_unknown_edges(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_path = self._build_adapter_fixture(Path(temp_dir))
            adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
            self.assertEqual(adapter["status"], "complete")
            self.assertEqual(adapter["coverage"]["same_event_verified_edge_count"], 3)
            self.assertEqual(adapter["coverage"]["unknown_dependency_count"], 1)
            self.assertFalse(
                adapter["edge_semantics"]["temporal_adjacency_used_as_dependency"]
            )

    def test_traffic_and_resource_model_labels_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter_path = self._build_adapter_fixture(root)
            hardware = root / "hardware.csv"
            _write_csv(
                hardware,
                [
                    "event_id",
                    "stage",
                    "matched_kernel_family",
                    "work_group_size",
                    "VGPR_count",
                    "SGPR_count",
                    "shared_memory_size_bytes",
                    "theoretical_occupancy_upper_bound_pct",
                    "hardware_evidence_class",
                ],
                [
                    {
                        "event_id": "input0_layer0",
                        "stage": "qkv_projection",
                        "matched_kernel_family": "gemm",
                        "work_group_size": 256,
                        "VGPR_count": 64,
                        "SGPR_count": 32,
                        "shared_memory_size_bytes": 0,
                        "theoretical_occupancy_upper_bound_pct": 100,
                        "hardware_evidence_class": "replay_projected_current_family",
                    }
                ],
            )
            capabilities = root / "device_capabilities.json"
            _write_json(
                capabilities,
                {
                    "architecture": "gfx936",
                    "physical_device_id": 1,
                    "wave_size": 64,
                    "wave_limit": 40,
                    "thread_limit": 2560,
                    "vgpr_resource": 196608,
                    "shared_memory_bytes": 65536,
                    "sources": [{"kind": "current_probe", "sha256": "b" * 64}],
                },
            )
            output_dir = root / "model"
            subprocess.run(
                [
                    sys.executable,
                    str(MODEL),
                    "--lineage-manifest",
                    str(root / "lineage.json"),
                    "--dependency-adapter",
                    str(adapter_path),
                    "--hardware-metrics",
                    str(hardware),
                    "--device-capabilities",
                    str(capabilities),
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            model = json.loads(
                (output_dir / "traffic_resource_model.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(model["traffic_boundary"]["hbm_or_dram_traffic_claimed"])
            self.assertFalse(model["resource_boundary"]["achieved_occupancy_claimed"])
            self.assertEqual(model["lineage_id"], LINEAGE_ID)
            self.assertEqual(model["coverage"]["kernel_family_count"], 1)
            with (output_dir / "kernel_family_resource_model.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["theoretical_occupancy_upper_bound_pct"], "100.0")

    def test_full_request_analyzer_requires_real_samples_in_exact_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter_path = self._build_adapter_fixture(root)
            model_path = self._build_model_fixture(root, adapter_path)
            expected = [
                "pra.fx_process.input0_layer0.input_rmsnorm",
                "pra.fx_process.input0_layer0.qkv_projection",
                "pra.fx_process.input0_layer0.mlp",
            ]
            metadata = root / "profile.json"
            _write_json(metadata, {
                "status": "profile_complete_analysis_pending", "process_profile": "on",
                "lineage_id": LINEAGE_ID,
                "contract_id": "fresh-contract", "contract_sha256": "a" * 64,
                "expected_process_ranges": expected, "emitted_process_ranges": expected,
                "expected_process_range_count": 3,
                "request_start_realtime_ns": 900, "request_end_realtime_ns": 2200,
            })
            trace = root / "process_trace_summary.json"
            _write_json(trace, {"status": "PASS", "contract_id": "fresh-contract"})
            performance = root / "process_performance.csv"
            _write_csv(
                performance,
                ["process_range", "event_id", "stage", "hiptx_begin_ns",
                 "hiptx_end_ns", "hiptx_cpu_ms"],
                [
                    {"process_range": expected[0], "event_id": "input0_layer0",
                     "stage": "input_rmsnorm", "hiptx_begin_ns": 1000,
                     "hiptx_end_ns": 1100, "hiptx_cpu_ms": 0.0001},
                    {"process_range": expected[1], "event_id": "input0_layer0",
                     "stage": "qkv_projection", "hiptx_begin_ns": 1100,
                     "hiptx_end_ns": 1900, "hiptx_cpu_ms": 0.0008},
                    {"process_range": expected[2], "event_id": "input0_layer0",
                     "stage": "mlp", "hiptx_begin_ns": 1900,
                     "hiptx_end_ns": 2100, "hiptx_cpu_ms": 0.0002},
                ],
            )
            process_gpu = root / "process_gpu_timeline.csv"
            _write_csv(process_gpu, ["kernel_id", "process_range"], [
                {"kernel_id": "1", "process_range": expected[1]}
            ])
            kernels = root / "kernels.csv"
            _write_csv(
                kernels,
                ["kernel_id", "begin_ns", "end_ns", "device_id", "queue_id",
                 "kernel_name", "kernel_family"],
                [{"kernel_id": 1, "begin_ns": 1200, "end_ns": 1800,
                  "device_id": 1, "queue_id": 0, "kernel_name": "gemm",
                  "kernel_family": "gemm"}],
            )
            live_samples = root / "live.jsonl"
            live_samples.write_text(
                "".join(json.dumps({
                    "sequence": index, "read_status": 0,
                    "realtime_midpoint_ns": timestamp,
                    "alignment_uncertainty_ns": 50,
                    "mean_se_active_cu_pct": 25.0 + index,
                    "max_se_active_cu_pct": 50.0 + index,
                    "se_active_cu_pct": [25.0 + index] * 8,
                }) + "\n" for index, timestamp in enumerate((1200, 1500, 1800))),
                encoding="utf-8",
            )
            live_summary = root / "live_summary.json"
            _write_json(live_summary, {
                "status": "complete", "physical_device_index": 1,
                "metric": "se_active_cu_pct",
                "empirical_sub_millisecond_cadence": {"p50": True, "p95": True},
            })
            annotations = root / "r07_annotations.csv"
            _write_csv(
                annotations,
                ["annotation_id", "kind", "message", "begin_ns", "end_ns",
                 "event_id", "forward_id", "layer", "occurrence", "phase"],
                [
                    {"annotation_id": "request", "kind": "request",
                     "message": "request", "begin_ns": 900, "end_ns": 2200,
                     "event_id": "", "forward_id": "", "layer": "",
                     "occurrence": "", "phase": ""},
                    {"annotation_id": "forward0", "kind": "forward",
                     "message": "forward0", "begin_ns": 950, "end_ns": 2150,
                     "event_id": "", "forward_id": "forward0", "layer": "",
                     "occurrence": 0, "phase": "prefill"},
                    {"annotation_id": "layer0", "kind": "layer",
                     "message": "layer0", "begin_ns": 980, "end_ns": 2120,
                     "event_id": "input0_layer0", "forward_id": "forward0",
                     "layer": 0, "occurrence": 0, "phase": "prefill"},
                ],
            )
            runtime_calls = root / "r07_runtime_calls.csv"
            _write_csv(
                runtime_calls,
                ["runtime_index", "begin_ns", "end_ns", "duration_ns",
                 "api_name", "process_owner"],
                [
                    {"runtime_index": 10, "begin_ns": 1020, "end_ns": 1040,
                     "duration_ns": 20, "api_name": "hipLaunchKernel",
                     "process_owner": expected[0]},
                    {"runtime_index": 11, "begin_ns": 1200, "end_ns": 1230,
                     "duration_ns": 30, "api_name": "hipLaunchKernel",
                     "process_owner": expected[1]},
                    {"runtime_index": 12, "begin_ns": 1950, "end_ns": 1980,
                     "duration_ns": 30, "api_name": "hipLaunchKernel",
                     "process_owner": expected[2]},
                ],
            )
            strict = root / "r07_strict_ownership.csv"
            _write_csv(
                strict,
                ["kind", "marker", "kernel_id", "runtime_index"],
                [{"kind": "process", "marker": expected[1],
                  "kernel_id": 1, "runtime_index": 11}],
            )
            output = root / "analysis"
            subprocess.run(
                [sys.executable, str(ANALYZER), "--profile-metadata", str(metadata),
                 "--process-trace-summary", str(trace), "--annotations",
                 str(annotations), "--runtime-calls", str(runtime_calls),
                 "--strict-ownership", str(strict), "--process-performance",
                 str(performance), "--process-gpu-timeline", str(process_gpu),
                 "--kernels", str(kernels), "--live-samples", str(live_samples),
                 "--live-summary", str(live_summary), "--dependency-adapter",
                 str(adapter_path), "--traffic-resource-model", str(model_path),
                 "--output-dir", str(output), "--high-latency-count", "1"],
                check=True, capture_output=True, text=True,
            )
            manifest = json.loads((output / "fresh_e2e_analysis.json").read_text())
            self.assertTrue(manifest["full_request_observed_timeline"])
            self.assertEqual(manifest["analysis_type"], "fresh_run_full_request_e2e")
            self.assertEqual(manifest["lineage_id"], LINEAGE_ID)
            self.assertEqual(manifest["high_latency_processes_with_live_samples"], 1)
            self.assertEqual(set(manifest["normalized_tables"]), {
                "request_timeline", "process_timeline", "kernel_timeline",
                "live_utilization_aligned", "process_live_utilization",
                "kernel_concurrency", "queue_concurrency", "launch_gaps",
                "high_latency_processes", "dependency_state",
                "traffic_resource_attachment", "opportunity_candidates",
            })
            self.assertGreater(
                manifest["normalized_tables"]["kernel_concurrency"]["row_count"],
                0,
            )
            acceptance_dir = root / "acceptance"
            subprocess.run(
                [sys.executable, str(VISUALIZER), "--analysis-manifest",
                 str(output / "fresh_e2e_analysis.json"), "--output-dir",
                 str(acceptance_dir)],
                check=True, capture_output=True, text=True,
            )
            acceptance = json.loads(
                (acceptance_dir / "offline_acceptance_manifest.json").read_text()
            )
            self.assertTrue(acceptance["self_contained_offline"])
            self.assertTrue(acceptance["view_coverage"]["filters_search_zoom"])
            self.assertEqual(
                set(acceptance["outputs"]),
                {"index.html", "E2E_PROCESS_TIMELINE.html",
                 "E2E_PROCESS_TIMELINE_LOSSLESS.html",
                 "HIGH_LATENCY_PROCESS_HARDWARE_TIMELINE.html",
                 "CONCURRENCY_UTILIZATION.html"},
            )
            companion = acceptance["companions"][
                "E2E_PROCESS_TIMELINE.full.perfetto.json"
            ]
            self.assertEqual(
                set(acceptance["companions"]),
                {
                    "E2E_PROCESS_TIMELINE.full.perfetto.json",
                    "full_timeline_manifest.json",
                },
            )
            expected_event_count = (
                manifest["normalized_tables"]["request_timeline"]["row_count"]
                + manifest["normalized_tables"]["process_timeline"]["row_count"]
                + 2
                * manifest["normalized_tables"]["kernel_timeline"]["row_count"]
            )
            self.assertEqual(companion["event_count"], expected_event_count)
            self.assertTrue(companion["complete_timeline"])
            self.assertFalse(companion["sampling_performed"])
            full_manifest = json.loads(
                (acceptance_dir / "full_timeline_manifest.json").read_text()
            )
            self.assertEqual(full_manifest["status"], "PASS")
            self.assertTrue(full_manifest["formal_r09_r10_regeneration"])
            self.assertFalse(full_manifest["sampling_performed"])
            self.assertEqual(full_manifest["event_count"], expected_event_count)
            self.assertEqual(
                full_manifest["source_table_hashes"],
                acceptance["source_table_hashes"],
            )
            self.assertIn(
                "data-sampling-performed='false'",
                (acceptance_dir / "E2E_PROCESS_TIMELINE_LOSSLESS.html").read_text(),
            )
            self.assertNotIn(
                "<script src=",
                (acceptance_dir / "E2E_PROCESS_TIMELINE.html").read_text(),
            )


if __name__ == "__main__":
    unittest.main()
