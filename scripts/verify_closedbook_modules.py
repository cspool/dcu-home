#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import ast
import csv
import json
import py_compile
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/cscc/closedbook/integration_manifest.json"
FORBIDDEN_CORE_IMPORTS = {"vllm", "sglang"}


def fail(message: str) -> None:
    raise SystemExit(f"verify_closedbook_modules: ERROR: {message}")


def check_core_imports(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            if name.split(".", 1)[0] in FORBIDDEN_CORE_IMPORTS:
                fail(f"portable core imports host framework in {path}: {name}")


def main() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if data["schema_version"] != 1:
        fail("unsupported integration manifest schema")
    core_files: set[Path] = set()
    for module in data["modules"]:
        for relative in module["portable_files"]:
            path = ROOT / relative
            if not path.is_file():
                fail(f"module {module['id']} portable file is missing: {relative}")
            core_files.add(path)
        for relative, anchor in module["host_anchors"]:
            path = ROOT / relative
            if not path.is_file():
                fail(f"module {module['id']} host file is missing: {relative}")
            if anchor not in path.read_text(encoding="utf-8"):
                fail(f"module {module['id']} host anchor is missing: {relative}: {anchor}")

    for path in sorted(core_files):
        if path.suffix == ".py":
            check_core_imports(path)
            py_compile.compile(str(path), doraise=True)

    profile = ROOT / (
        "qwen35_rocm_opt/profiles/"
        "gfx936_qwen3_5_27b_bf16_tn_m4096.csv"
    )
    rows = list(csv.reader(profile.open(encoding="utf-8")))
    validators = [row for row in rows if row and row[0] == "Validator"]
    results = [row for row in rows if row and row[0] != "Validator"]
    if (len(validators), len(results)) != (5, 5):
        fail(
            "portable TunableOp profile must contain exactly 5 validators "
            f"and 5 results, got {len(validators)} and {len(results)}"
        )

    required_docs = (
        "docs/cscc/MODULAR_CLOSED_BOOK.md",
        "docs/cscc/closedbook/PERSON_A_ATTENTION.md",
        "docs/cscc/closedbook/PERSON_B_GDN.md",
        "docs/cscc/closedbook/PERSON_C_GEMV_RUNTIME.md",
    )
    for relative in required_docs:
        if not (ROOT / relative).is_file():
            fail(f"closed-book document is missing: {relative}")

    probe = (
        "import sys; import qwen35_rocm_opt; "
        "import qwen35_rocm_opt.attention, qwen35_rocm_opt.gdn; "
        "import qwen35_rocm_opt.gemv, qwen35_rocm_opt.runtime; "
        "assert not any(x == 'vllm' or x.startswith('vllm.') or "
        "x == 'sglang' or x.startswith('sglang.') for x in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", probe], cwd=ROOT, check=True)
    print(
        "verify_closedbook_modules: OK: 3 modules, host anchors, "
        "host-free core imports, profile 5+5, and closed-book documents"
    )


if __name__ == "__main__":
    main()
