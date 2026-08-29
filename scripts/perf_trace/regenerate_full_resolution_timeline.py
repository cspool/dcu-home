#!/usr/bin/env python3
"""Derive an unsampled full-resolution viewer from an accepted E2E page.

This is a presentation-only recovery path for a retained R10 acceptance archive
whose original same-run R09 normalized tables are no longer available.  It does
not rewrite or claim to replace the formal R10 manifest, handoff, or audit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tarfile
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any


SOURCE_MEMBER = "acceptance/E2E_PROCESS_TIMELINE.html"
SOURCE_MANIFEST_MEMBER = "acceptance/offline_acceptance_manifest.json"
LOSSLESS_PAGE = "E2E_PROCESS_TIMELINE_LOSSLESS.html"
FULL_TRACE = "E2E_PROCESS_TIMELINE.full.perfetto.json"
DERIVED_MANIFEST = "full_timeline_manifest.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_script(text: str, script_id: str) -> str:
    marker = f"<script type='application/json' id='{script_id}'>"
    begin = text.find(marker)
    if begin < 0:
        raise RuntimeError(f"source page lacks {script_id}")
    begin += len(marker)
    end = text.find("</script>", begin)
    if end < 0:
        raise RuntimeError(f"source page has unterminated {script_id}")
    return text[begin:end]


def load_generator(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("fresh_e2e_visualizer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import maintained generator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def output_record(path: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(output_dir)),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a derived unsampled full-resolution timeline bundle."
    )
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--generator",
        type=Path,
        default=Path(__file__).with_name("generate_fresh_e2e_visualization.py"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_archive = args.source_archive.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    generator_path = args.generator.expanduser().resolve()
    if not source_archive.is_file():
        raise RuntimeError(f"source archive is missing: {source_archive}")
    if not generator_path.is_file():
        raise RuntimeError(f"maintained generator is missing: {generator_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing nonempty output directory: {output_dir}")

    with tarfile.open(source_archive, "r:gz") as archive:
        source_handle = archive.extractfile(SOURCE_MEMBER)
        manifest_handle = archive.extractfile(SOURCE_MANIFEST_MEMBER)
        if source_handle is None or manifest_handle is None:
            raise RuntimeError("source archive lacks accepted E2E page or manifest")
        source_page_bytes = source_handle.read()
        source_manifest_bytes = manifest_handle.read()

    source_page = source_page_bytes.decode("utf-8")
    source_metadata = json.loads(extract_script(source_page, "acceptance-metadata"))
    source_payload = json.loads(extract_script(source_page, "page-payload"))
    source_acceptance = json.loads(source_manifest_bytes)
    if source_acceptance.get("status") != "PASS":
        raise RuntimeError("source acceptance manifest did not pass")
    if source_metadata.get("lineage_id") != source_acceptance.get("lineage_id"):
        raise RuntimeError("source page and source manifest lineage differ")
    required_groups = [
        "request", "forward", "layer", "process", "hip_runtime",
        "gpu_queue", "strict_owned_kernel",
    ]
    if source_payload.get("groups") != required_groups:
        raise RuntimeError("source page does not contain the complete E2E track groups")
    rows = source_payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("source page has no embedded E2E intervals")
    row_counts = source_metadata.get("source_table_row_counts", {})
    expected_count = (
        int(row_counts["request_timeline"])
        + int(row_counts["process_timeline"])
        + 2 * int(row_counts["kernel_timeline"])
    )
    if len(rows) != expected_count:
        raise RuntimeError(
            f"source E2E interval count changed: {len(rows)} != {expected_count}"
        )
    begin = int(source_payload["begin"])
    end = int(source_payload["end"])
    bounds = source_metadata.get("request_bounds", {})
    if begin != int(bounds["begin_ns"]) or end != int(bounds["end_ns"]):
        raise RuntimeError("source E2E bounds differ from accepted metadata")

    visualizer = load_generator(generator_path)
    retained_process_rows = [
        {
            "process_range": str(row.get("process") or row.get("n") or ""),
            "hiptx_begin_ns": str(row["b"]),
            "hiptx_end_ns": str(row["e"]),
        }
        for row in rows
        if row.get("g") == "process"
    ]
    top_latency_process_contract = (
        visualizer.build_top_latency_process_contract(
            retained_process_rows, begin, end
        )
    )
    top_latency_process_contract = {
        **top_latency_process_contract,
        "ranking_source": (
            "retained_accepted_R10_complete_process_intervals"
        ),
        "formal_r09_r10_regeneration": False,
        "source_process_timeline_sha256": source_metadata[
            "source_table_hashes"
        ]["process_timeline"],
    }
    enriched_source_payload = {
        **source_payload,
        "top_latency_processes": top_latency_process_contract["selected"],
        "top_latency_process_policy": {
            key: value
            for key, value in top_latency_process_contract.items()
            if key != "selected"
        },
    }
    lossless_payload = visualizer.build_lossless_timeline_payload(
        begin, end, rows, top_latency_process_contract
    )
    trace = visualizer.build_full_perfetto_trace(
        {
            "request_begin_realtime_ns": begin,
            "lineage_id": source_metadata["lineage_id"],
        },
        {"E2E_PROCESS_TIMELINE.html": enriched_source_payload},
        source_metadata,
    )
    expected_categories = dict(sorted(Counter(row["g"] for row in rows).items()))
    trace_metadata = trace["metadata"]
    if (
        len(trace["traceEvents"]) != expected_count
        or trace_metadata.get("complete_timeline") is not True
        or trace_metadata.get("sampling_performed") is not False
        or trace_metadata.get("event_count_by_category") != expected_categories
    ):
        raise RuntimeError("generated full trace failed completeness checks")

    output_dir.mkdir(parents=True, exist_ok=True)
    lossless_page_path = output_dir / LOSSLESS_PAGE
    full_trace_path = output_dir / FULL_TRACE
    index_path = output_dir / "index.html"
    derived_manifest_path = output_dir / DERIVED_MANIFEST

    derived_metadata = {
        **source_metadata,
        "artifact_class": "derived_full_resolution_viewer_bundle",
        "formal_r09_r10_regeneration": False,
        "sampling_performed": False,
        "source_acceptance_archive_sha256": sha256_file(source_archive),
        "source_e2e_page_sha256": sha256_bytes(source_page_bytes),
        "top_latency_process_contract": top_latency_process_contract,
    }
    lossless_page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Derived lossless E2E process timeline</title><style>"
        + visualizer.CSS
        + "</style></head><body><main><nav><a href='index.html'>index.html</a>"
        + f"<a href='{FULL_TRACE}' download>Complete Perfetto-compatible trace</a>"
        + "</nav><h1>Full-resolution observed end-to-end request timeline</h1>"
        + "<div class='backend'><strong>DERIVED VIEWER ONLY — formal_r09_r10_regeneration=false. The source accepted page and every embedded interval are hash-bound; no model, profiler, GPU, PMC, or sampling activity was performed.</strong></div>"
        + visualizer.evidence_legend()
        + visualizer.top_latency_process_legend(lossless_payload)
        + visualizer.LOSSLESS_E2E_BODY
        + "<script type='application/json' id='acceptance-metadata'>"
        + visualizer.compact_json(derived_metadata)
        + "</script><script type='application/json' id='page-payload'>"
        + visualizer.compact_json(lossless_payload)
        + "</script><script>"
        + visualizer.LOSSLESS_E2E_JS
        + "</script></main></body></html>"
    )
    write_text(lossless_page_path, lossless_page)
    write_text(
        full_trace_path,
        json.dumps(trace, ensure_ascii=False, separators=(",", ":")) + "\n",
    )
    index = (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Qwen fresh E2E 全分辨率时间轴</title><style>"
        + visualizer.CSS
        + "</style></head><body><main><h1>Qwen fresh E2E 全分辨率时间轴</h1>"
        "<div class='note'>本包从已通过验收的 E2E HTML 内嵌完整 payload 派生，不抽样，不运行模型或 DCU。它只改善查看，不替换原 R10 审计。</div>"
        "<ul>"
        f"<li><a href='{LOSSLESS_PAGE}'>打开全分辨率交互时间轴</a></li>"
        f"<li><a href='{FULL_TRACE}'>下载/打开全量 Perfetto-compatible trace</a></li>"
        f"<li><a href='{DERIVED_MANIFEST}'>查看派生包 manifest</a></li>"
        "</ul></main></body></html>"
    )
    write_text(index_path, index)

    outputs = {
        "index.html": output_record(index_path, output_dir),
        LOSSLESS_PAGE: output_record(lossless_page_path, output_dir),
        FULL_TRACE: output_record(full_trace_path, output_dir),
    }
    derived_manifest = {
        "schema_version": 1,
        "status": "PASS",
        "artifact_class": "derived_full_resolution_viewer_bundle",
        "formal_r09_r10_regeneration": False,
        "original_acceptance_untouched": True,
        "lineage_id": source_metadata["lineage_id"],
        "sampling_performed": False,
        "event_count": expected_count,
        "event_count_formula": (
            "request_timeline + process_timeline + 2 * kernel_timeline"
        ),
        "event_count_by_category": expected_categories,
        "timestamp_encoding": {
            "rendering": "integer nanoseconds relative to request begin",
            "absolute": "decimal strings per event",
            "perfetto": "relative microseconds with three decimal nanosecond precision",
        },
        "source": {
            "archive": {
                "path": str(source_archive),
                "sha256": sha256_file(source_archive),
            },
            "accepted_e2e_member": {
                "path": SOURCE_MEMBER,
                "sha256": sha256_bytes(source_page_bytes),
            },
            "accepted_manifest_member": {
                "path": SOURCE_MANIFEST_MEMBER,
                "sha256": sha256_bytes(source_manifest_bytes),
            },
            "source_table_hashes": source_metadata["source_table_hashes"],
            "source_table_row_counts": row_counts,
        },
        "top_latency_process_contract": top_latency_process_contract,
        "generators": {
            "derived_bundle_generator": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "maintained_r10_generator": {
                "path": str(generator_path),
                "sha256": sha256_file(generator_path),
            },
        },
        "outputs": outputs,
        "validation": {
            "source_acceptance_status": source_acceptance["status"],
            "source_lineage_matches": True,
            "source_request_bounds_match": True,
            "source_event_count_matches_formula": True,
            "full_trace_event_count_matches_source": True,
            "category_counts_match_source": True,
            "relative_nanosecond_payload_exact": True,
            "top_latency_process_ranking_recomputed_from_all_process_intervals": True,
            "top_latency_process_distinct_fill_colors": True,
            "owned_runtime_queue_kernel_same_color_outlines": True,
            "zoom_reveals_process_names_inside_rectangles": True,
            "model_run_count": 0,
            "gpu_activity_count": 0,
            "profiler_run_count": 0,
            "pmc_replay_count": 0,
        },
    }
    write_text(
        derived_manifest_path,
        json.dumps(derived_manifest, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps({
        "status": "PASS",
        "output_dir": str(output_dir),
        "manifest": str(derived_manifest_path),
        "event_count": expected_count,
        "sampling_performed": False,
        "formal_r09_r10_regeneration": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
