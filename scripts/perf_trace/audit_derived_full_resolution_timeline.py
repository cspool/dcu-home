#!/usr/bin/env python3
"""Independently audit a retained-data R10 presentation replay bundle."""

from __future__ import annotations

import argparse
import gc
import hashlib
import html
import json
import subprocess
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SOURCE_MEMBER = "acceptance/E2E_PROCESS_TIMELINE.html"
SOURCE_MANIFEST_MEMBER = "acceptance/offline_acceptance_manifest.json"
LOSSLESS_PAGE = "E2E_PROCESS_TIMELINE_LOSSLESS.html"
FULL_TRACE = "E2E_PROCESS_TIMELINE.full.perfetto.json"
DERIVED_MANIFEST = "full_timeline_manifest.json"
AUDIT_NAME = "R10_PRESENTATION_REPLAY_AUDIT.json"
DETERMINISTIC_FILES = (
    "index.html",
    LOSSLESS_PAGE,
    FULL_TRACE,
    DERIVED_MANIFEST,
)
PALETTE = (
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
)
CAVEAT = "Overlapping process intervals are not additive end-to-end attribution."


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON must be an object: {path}")
    return value


def extract_script(text: str, script_id: str) -> str:
    marker = f"<script type='application/json' id='{script_id}'>"
    begin = text.find(marker)
    if begin < 0:
        raise RuntimeError(f"page lacks {script_id}")
    begin += len(marker)
    end = text.find("</script>", begin)
    if end < 0:
        raise RuntimeError(f"page has unterminated {script_id}")
    return text[begin:end]


def canonical_row_hash(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def relative_rows(
    rows: Iterable[dict[str, Any]], begin: int
) -> Iterable[dict[str, Any]]:
    for row in rows:
        yield {
            **{
                key: value
                for key, value in row.items()
                if key not in {"b", "e"}
            },
            "b": int(row["b"]) - begin,
            "e": int(row["e"]) - begin,
            "b_abs": str(row["b"]),
            "e_abs": str(row["e"]),
        }


def build_expected_contract(
    rows: list[dict[str, Any]], begin: int, end: int, process_hash: str
) -> dict[str, Any]:
    process_rows = [row for row in rows if row.get("g") == "process"]
    names = [str(row.get("process") or row.get("n") or "") for row in process_rows]
    if not names or any(not name for name in names) or len(names) != len(set(names)):
        raise RuntimeError("source process intervals do not have unique exact names")
    ranked = sorted(
        (
            int(row["e"]) - int(row["b"]),
            int(row["b"]),
            str(row.get("process") or row.get("n")),
        )
        for row in process_rows
    )
    if any(duration < 0 for duration, _, _ in ranked):
        raise RuntimeError("source process interval has negative duration")
    total = sum(duration for duration, _, _ in ranked)
    request_span = end - begin
    if total <= 0 or request_span <= 0:
        raise RuntimeError("source duration totals are invalid")
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = [
        {
            "rank": rank,
            "process_range": process_range,
            "hiptx_begin_ns": process_begin,
            "observed_duration_ns": duration,
            "observed_process_duration_share": duration / total,
            "observed_request_span_ratio": duration / request_span,
            "request_span_ratio_caveat": CAVEAT,
            "color": PALETTE[rank - 1],
        }
        for rank, (duration, process_begin, process_range) in enumerate(
            ranked[: len(PALETTE)], start=1
        )
    ]
    return {
        "schema_version": 1,
        "ranking_source": "retained_accepted_R10_complete_process_intervals",
        "ranking_duration": "hiptx_end_ns - hiptx_begin_ns",
        "ranking_order": [
            "observed_duration_ns_descending",
            "hiptx_begin_ns_ascending",
            "process_range_ascending",
        ],
        "configured_count": len(PALETTE),
        "selected_count": len(selected),
        "palette": list(PALETTE),
        "observed_process_duration_total_ns": total,
        "observed_request_span_ns": request_span,
        "request_span_ratio_caveat": CAVEAT,
        "selected": selected,
        "formal_r09_r10_regeneration": False,
        "source_process_timeline_sha256": process_hash,
    }


def stream_token_count(path: Path, token: bytes) -> int:
    count = 0
    overlap = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            value = overlap + chunk
            count += value.count(token)
            overlap = value[-max(0, len(token) - 1) :]
    return count


def read_trace_suffix(path: Path) -> dict[str, Any]:
    marker = b'],"displayTimeUnit":'
    with path.open("rb") as handle:
        size = handle.seek(0, 2)
        handle.seek(max(0, size - (4 << 20)))
        tail = handle.read()
    index = tail.rfind(marker)
    if index < 0:
        raise RuntimeError("complete trace suffix is unavailable")
    return json.loads(b'{"displayTimeUnit":' + tail[index + len(marker) :])


class Checker:
    def __init__(self) -> None:
        self.checks: list[str] = []

    def require(self, condition: bool, label: str) -> None:
        if not condition:
            raise RuntimeError(f"independent replay audit failed: {label}")
        self.checks.append(label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a retained-data R10 presentation replay."
    )
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--derived-generator", type=Path, required=True)
    parser.add_argument("--maintained-generator", type=Path, required=True)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_archive = args.source_archive.expanduser().resolve()
    bundle_dir = args.bundle_dir.expanduser().resolve()
    reference_dir = args.reference_dir.expanduser().resolve()
    derived_generator = args.derived_generator.expanduser().resolve()
    maintained_generator = args.maintained_generator.expanduser().resolve()
    node = args.node.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else bundle_dir / AUDIT_NAME
    )
    checker = Checker()
    for path, label in (
        (source_archive, "source archive"),
        (derived_generator, "derived generator"),
        (maintained_generator, "maintained generator"),
        (node, "Node.js syntax checker"),
    ):
        checker.require(path.is_file(), f"{label} exists")
    checker.require(bundle_dir.is_dir(), "bundle directory exists")
    checker.require(reference_dir.is_dir(), "determinism reference exists")
    checker.require(not output.exists(), "audit output is new")

    with tarfile.open(source_archive, "r:gz") as archive:
        source_page_handle = archive.extractfile(SOURCE_MEMBER)
        source_manifest_handle = archive.extractfile(SOURCE_MANIFEST_MEMBER)
        checker.require(source_page_handle is not None, "source page member exists")
        checker.require(
            source_manifest_handle is not None, "source manifest member exists"
        )
        source_page_bytes = source_page_handle.read()
        source_manifest_bytes = source_manifest_handle.read()
    source_manifest = json.loads(source_manifest_bytes)
    checker.require(source_manifest.get("status") == "PASS", "source R10 passed")
    source_text = source_page_bytes.decode("utf-8")
    source_metadata = json.loads(extract_script(source_text, "acceptance-metadata"))
    source_payload = json.loads(extract_script(source_text, "page-payload"))
    del source_text
    begin = int(source_payload["begin"])
    end = int(source_payload["end"])
    source_rows = source_payload.get("rows")
    checker.require(isinstance(source_rows, list), "source rows parse")
    expected_count = (
        int(source_metadata["source_table_row_counts"]["request_timeline"])
        + int(source_metadata["source_table_row_counts"]["process_timeline"])
        + 2 * int(source_metadata["source_table_row_counts"]["kernel_timeline"])
    )
    checker.require(len(source_rows) == expected_count, "source event count is complete")
    expected_categories = dict(sorted(Counter(row["g"] for row in source_rows).items()))
    expected_contract = build_expected_contract(
        source_rows,
        begin,
        end,
        source_metadata["source_table_hashes"]["process_timeline"],
    )
    expected_relative_hash = canonical_row_hash(relative_rows(source_rows, begin))
    source_lineage = source_metadata["lineage_id"]
    del source_payload, source_rows
    gc.collect()

    manifest_path = bundle_dir / DERIVED_MANIFEST
    page_path = bundle_dir / LOSSLESS_PAGE
    trace_path = bundle_dir / FULL_TRACE
    index_path = bundle_dir / "index.html"
    for path in (manifest_path, page_path, trace_path, index_path):
        checker.require(path.is_file(), f"{path.name} exists")
    manifest = load_object(manifest_path)
    checker.require(manifest.get("status") == "PASS", "derived manifest passed")
    checker.require(
        manifest.get("formal_r09_r10_regeneration") is False,
        "bundle denies formal R09/R10 regeneration",
    )
    checker.require(
        manifest.get("original_acceptance_untouched") is True,
        "bundle preserves original acceptance",
    )
    checker.require(manifest.get("sampling_performed") is False, "bundle denies sampling")
    checker.require(manifest.get("lineage_id") == source_lineage, "lineage matches")
    checker.require(manifest.get("event_count") == expected_count, "manifest event count matches")
    checker.require(
        manifest.get("event_count_by_category") == expected_categories,
        "manifest categories match source",
    )
    checker.require(
        manifest.get("top_latency_process_contract") == expected_contract,
        "manifest top-process ranking and palette reproduce the source",
    )
    checker.require(
        manifest["source"]["archive"]["sha256"] == sha256_file(source_archive),
        "manifest binds source archive",
    )
    checker.require(
        manifest["source"]["accepted_e2e_member"]["sha256"]
        == sha256_bytes(source_page_bytes),
        "manifest binds accepted E2E member",
    )
    checker.require(
        manifest["source"]["accepted_manifest_member"]["sha256"]
        == sha256_bytes(source_manifest_bytes),
        "manifest binds accepted manifest member",
    )
    for name, path in (
        ("index.html", index_path),
        (LOSSLESS_PAGE, page_path),
        (FULL_TRACE, trace_path),
    ):
        record = manifest["outputs"][name]
        checker.require(record["path"] == name, f"{name} uses a relative path")
        checker.require(record["sha256"] == sha256_file(path), f"{name} hash matches")
        checker.require(record["size_bytes"] == path.stat().st_size, f"{name} size matches")

    page_text = page_path.read_text(encoding="utf-8")
    for token in (
        "<script src=",
        "http://",
        "https://",
        "fetch(",
        "XMLHttpRequest",
        "WebSocket",
    ):
        checker.require(token not in page_text, f"lossless page excludes {token}")
    output_metadata = json.loads(extract_script(page_text, "acceptance-metadata"))
    output_payload = json.loads(extract_script(page_text, "page-payload"))
    checker.require(output_metadata["lineage_id"] == source_lineage, "page lineage matches")
    checker.require(
        output_metadata["formal_r09_r10_regeneration"] is False,
        "page identifies presentation-only replay",
    )
    checker.require(output_payload["origin_ns"] == str(begin), "page preserves absolute origin")
    checker.require(
        output_payload["begin"] == 0 and output_payload["end"] == end - begin,
        "page uses exact relative nanoseconds",
    )
    output_rows = output_payload.get("rows")
    checker.require(isinstance(output_rows, list), "output rows parse")
    checker.require(len(output_rows) == expected_count, "page retains every interval")
    checker.require(
        canonical_row_hash(output_rows) == expected_relative_hash,
        "every relative interval exactly reproduces the accepted payload",
    )
    output_contract = {
        **output_payload["top_latency_process_policy"],
        "selected": output_payload["top_latency_processes"],
    }
    checker.require(output_contract == expected_contract, "page embeds exact top-process contract")
    checker.require(
        len({item["color"] for item in output_contract["selected"]}) == len(PALETTE),
        "top ten colors are distinct",
    )
    checker.require(
        f"data-top-latency-process-count='{len(PALETTE)}'" in page_text,
        "page includes the top-process legend",
    )
    for item in expected_contract["selected"]:
        checker.require(
            f"data-process-range='{html.escape(item['process_range'])}'" in page_text,
            f"legend includes rank {item['rank']} exact process name",
        )
        checker.require(
            f"data-color='{item['color']}'" in page_text,
            f"legend includes rank {item['rank']} color",
        )
    for token, label in (
        ("g==='process'&&top?top.color", "selected process fill logic"),
        ("ownedGroups.has(g)", "owned interval outline logic"),
        ("X.fillText(name", "in-rectangle process label logic"),
        ("w>=tw+8", "zoom-dependent full-name fit gate"),
        ("const topByProcess", "exact process-to-color mapping"),
    ):
        checker.require(token in page_text, f"page contains {label}")
    del page_text, output_payload, output_rows
    gc.collect()
    node_check = subprocess.run(
        [
            str(node),
            "-e",
            (
                "const fs=require('fs'),t=fs.readFileSync(process.argv[1],'utf8'),"
                "re=/<script([^>]*)>([\\s\\S]*?)<\\/script>/g;"
                "let m,n=0;while((m=re.exec(t))!==null){"
                "if(!m[1].includes('application/json')){new Function(m[2]);n++;}}"
                "if(n!==1)throw new Error('expected one application script, got '+n);"
            ),
            str(page_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    checker.require(
        node_check.returncode == 0,
        "lossless page inline application JavaScript parses",
    )

    trace_suffix = read_trace_suffix(trace_path)
    trace_metadata = trace_suffix.get("metadata", {})
    checker.require(trace_suffix.get("displayTimeUnit") == "ns", "trace time unit is ns")
    checker.require(trace_metadata.get("event_count") == expected_count, "trace count matches")
    checker.require(
        trace_metadata.get("event_count_by_category") == expected_categories,
        "trace categories match",
    )
    checker.require(trace_metadata.get("sampling_performed") is False, "trace denies sampling")
    checker.require(
        trace_metadata.get("top_latency_processes") == expected_contract["selected"],
        "trace metadata embeds the exact top-process mapping",
    )
    checker.require(
        stream_token_count(trace_path, b'"top_latency_process_rank":') == len(PALETTE),
        "trace annotates exactly ten process events",
    )
    checker.require(
        stream_token_count(trace_path, b'"top_latency_owner_rank":') > 0,
        "trace annotates owned runtime/queue/kernel events",
    )
    validation = manifest.get("validation", {})
    for key in (
        "top_latency_process_ranking_recomputed_from_all_process_intervals",
        "top_latency_process_distinct_fill_colors",
        "owned_runtime_queue_kernel_same_color_outlines",
        "zoom_reveals_process_names_inside_rectangles",
    ):
        checker.require(validation.get(key) is True, f"manifest validates {key}")

    deterministic_hashes: dict[str, str] = {}
    for name in DETERMINISTIC_FILES:
        primary = bundle_dir / name
        reference = reference_dir / name
        checker.require(reference.is_file(), f"determinism reference {name} exists")
        primary_hash = sha256_file(primary)
        checker.require(
            primary_hash == sha256_file(reference),
            f"independent regeneration is byte-identical for {name}",
        )
        deterministic_hashes[name] = primary_hash

    audit = {
        "schema_version": 1,
        "status": "PASS",
        "artifact_class": "retained_r10_presentation_replay_independent_audit",
        "formal_r09_r10_regeneration": False,
        "original_acceptance_untouched": True,
        "lineage_id": source_lineage,
        "check_count": len(checker.checks),
        "failure_count": 0,
        "source_archive": {
            "path": str(source_archive),
            "sha256": sha256_file(source_archive),
        },
        "bundle": {
            "path": str(bundle_dir),
            "manifest_sha256": sha256_file(manifest_path),
            "lossless_page_sha256": sha256_file(page_path),
            "complete_trace_sha256": sha256_file(trace_path),
        },
        "generators": {
            "derived": {
                "path": str(derived_generator),
                "sha256": sha256_file(derived_generator),
            },
            "maintained": {
                "path": str(maintained_generator),
                "sha256": sha256_file(maintained_generator),
            },
            "auditor": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "javascript_syntax_checker": {
                "path": str(node),
                "version": subprocess.run(
                    [str(node), "--version"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
            },
        },
        "validation": {
            "complete_interval_count": expected_count,
            "category_counts": expected_categories,
            "relative_interval_projection_exact": True,
            "top_latency_process_contract_exact": True,
            "distinct_color_count": len(PALETTE),
            "owned_interval_outline_contract_present": True,
            "zoom_process_name_contract_present": True,
            "self_contained_offline": True,
            "inline_application_javascript_syntax": "PASS",
            "sampling_performed": False,
            "model_run_count": 0,
            "gpu_activity_count": 0,
            "profiler_run_count": 0,
            "pmc_replay_count": 0,
        },
        "determinism": {
            "status": "PASS",
            "method": (
                "two independent retained-data generator invocations in "
                "separate output directories with byte-for-byte comparison"
            ),
            "reference_location_at_validation": str(reference_dir),
            "compared_file_count": len(DETERMINISTIC_FILES),
            "identical_sha256_by_file": deterministic_hashes,
        },
        "top_latency_processes": expected_contract["selected"],
    }
    output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "audit": str(output),
                "check_count": len(checker.checks),
                "event_count": expected_count,
                "distinct_color_count": len(PALETTE),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
