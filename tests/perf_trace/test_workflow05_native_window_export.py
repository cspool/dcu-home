from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPORTER = ROOT / "scripts/perf_trace/export_workflow05_native_windows.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Workflow05NativeWindowExportTest(unittest.TestCase):
    def create_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        database = root / "capture.db"
        with sqlite3.connect(database) as connection:
            connection.executescript(
                """
                CREATE TABLE HIP_cfg (
                  BeginNs INTEGER, EndNs INTEGER, pid INTEGER, tid INTEGER,
                  Name INTEGER, args TEXT, _Index INTEGER, State INTEGER,
                  DurationNs INTEGER, ExtIndex INTEGER
                );
                CREATE TABLE HIPOPS_cfg (
                  BeginNs INTEGER, EndNs INTEGER, dev_id INTEGER, queue_id TEXT,
                  Name TEXT, pid INTEGER, tid INTEGER, _Index INTEGER,
                  DurationNs INTEGER, PARS TEXT, accum_id INTEGER
                );
                CREATE TABLE HIPTX_cfg (
                  type INTEGER, cid INTEGER, _Index INTEGER, BeginNs INTEGER,
                  EndNs INTEGER, pid INTEGER, tid INTEGER, rid INTEGER,
                  message TEXT, begin_Index INTEGER, end_Index INTEGER
                );
                """
            )
            connection.executemany(
                "INSERT INTO HIP_cfg VALUES (?,?,?,?,?,?,?,?,?,?)",
                [
                    (100, 110, 7, 7, 1, "a", 10, 0, 10, 0),
                    (120, 140, 7, 7, 1, "b", 11, 0, 20, 0),
                ],
            )
            connection.execute(
                "INSERT INTO HIPOPS_cfg VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (200, 250, 1, "0", "kernel_a", 7, 0, 11, 50, "p", 1),
            )
            connection.execute(
                "INSERT INTO HIPTX_cfg VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    1,
                    1,
                    9,
                    90,
                    150,
                    7,
                    7,
                    1,
                    "pra.fx_process.input1_layer0.mlp",
                    10,
                    11,
                ),
            )
        contract = root / "contract.json"
        contract.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "verified_deferred_export_contract",
                    "source_database": {
                        "path": str(database),
                        "sha256_after_validation": sha256(database),
                        "hip_table": "HIP_cfg",
                        "hipops_table": "HIPOPS_cfg",
                        "hiptx_table": "HIPTX_cfg",
                    },
                    "windows": [
                        {
                            "selection_rank": 1,
                            "stable_key": "contract/1/0/1/mlp",
                            "event_id": "input1_layer0",
                            "exact_process_range": "pra.fx_process.input1_layer0.mlp",
                            "runtime_index_begin": 10,
                            "runtime_index_end": 11,
                            "strict_contained_runtime_call_count": 2,
                            "strict_owned_kernel_count": 1,
                            "strict_owned_kernel_duration_ns": 50,
                        }
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        fake = root / "hipprof"
        fake.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import json
                import sqlite3
                import sys
                from pathlib import Path

                args = sys.argv[1:]
                def value(flag):
                    return args[args.index(flag) + 1]
                database = Path(value('--db'))
                output_type = int(value('--output-type'))
                prefix = Path(value('-o'))
                with sqlite3.connect(database) as connection:
                    connection.execute('CREATE TABLE RANGE_SUMMARY (name TEXT)')
                    connection.execute('INSERT INTO RANGE_SUMMARY VALUES (?)', ('ok',))
                if output_type == 2:
                    prefix.with_name(prefix.name + '_stream.pftrace').write_bytes(b'perfetto')
                else:
                    events = [
                      {'ph':'M','name':'process_name','pid':1,'args':{'name':'[1] Runtime API'}},
                      {'ph':'M','name':'process_name','pid':2,'args':{'name':'[2] Stream on Device 1'}},
                      {'ph':'X','cat':'HIP','name':'hipA','pid':1,'tid':7,'ts':0.01,'dur':0.01,'args':{'BeginNs':100,'EndNs':110,'index':10}},
                      {'ph':'X','cat':'HIP','name':'hipB','pid':1,'tid':7,'ts':0.03,'dur':0.02,'args':{'BeginNs':120,'EndNs':140,'index':11}},
                      {'ph':'X','cat':'HIPOPS','name':'kernel_a','pid':2,'tid':'Stream0','ts':0.11,'dur':0.05,'args':{'BeginNs':200,'EndNs':250,'index':11}},
                      {'ph':'s','cat':'DataFlow','name':'flow','pid':1,'tid':7,'ts':0.03,'id':11},
                      {'ph':'t','cat':'DataFlow','name':'flow','pid':2,'tid':'Stream0','ts':0.11,'id':11},
                    ]
                    prefix.with_name(prefix.name + '_stream.json').write_text(json.dumps({'traceEvents':events}))
                """
            ),
            encoding="utf-8",
        )
        fake.chmod(0o755)
        return database, contract, fake

    def test_native_formats_use_copies_and_preserve_source(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            database, contract, fake = self.create_fixture(root)
            source_sha = sha256(database)
            output = root / "output"
            subprocess.run(
                [
                    sys.executable,
                    str(EXPORTER),
                    "--contract",
                    str(contract),
                    "--output-dir",
                    str(output),
                    "--hipprof-bin",
                    str(fake),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(sha256(database), source_sha)
            manifest = json.loads(
                (output / "workflow05_native_hipprof_window_exports.json").read_text()
            )
            self.assertEqual(manifest["status"], "pass")
            self.assertTrue(manifest["source_database"]["unchanged"])
            self.assertEqual(
                manifest["exporter"]["path"], str(EXPORTER.resolve())
            )
            self.assertEqual(manifest["library_directories"], [])
            self.assertEqual(
                [attempt["format"] for attempt in manifest["windows"][0]["attempts"]],
                ["pftrace", "chrome-json"],
            )
            for attempt in manifest["windows"][0]["attempts"]:
                self.assertEqual(attempt["status"], "pass")
                self.assertTrue(
                    attempt["copy_mutation"]["known_range_summary_mutation_only"]
                )
            chrome = manifest["windows"][0]["attempts"][1]
            self.assertEqual(chrome["semantic_validation"]["status"], "pass")
            self.assertEqual(
                chrome["exact_process_marker_overlay"]["mode"],
                "native_events_plus_exact_db_marker_overlay",
            )
            candidate = Path(chrome["trace_files"][0]["path"])
            self.assertIn("_with_process_marker", candidate.name)
            payload = json.loads(candidate.read_text())
            markers = [
                event
                for event in payload["traceEvents"]
                if event.get("name") == "pra.fx_process.input1_layer0.mlp"
            ]
            self.assertEqual(len(markers), 1)
            self.assertFalse(chrome["gpu_or_model_activity"])


if __name__ == "__main__":
    unittest.main()
