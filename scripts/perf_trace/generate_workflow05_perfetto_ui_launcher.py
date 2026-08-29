#!/usr/bin/env python3
"""Generate a thin launcher for Perfetto UI's official embedding interface.

The generated page does not render or analyze a trace.  When served from
localhost over HTTP, it performs Perfetto's PING/PONG handshake and posts the
exact trace bytes to the upstream UI as an ArrayBuffer.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse


BACKEND_ORDER = [
    "perfetto_trace_processor_python_api",
    "perfetto_trace_processor_cli",
    "perfetto_ui_local_file",
    "custom_plotly_timeline_fallback",
]


class LauncherError(RuntimeError):
    """Fail-closed Perfetto UI launcher generation error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LauncherError(f"{path} must contain one JSON object")
    return value


def js(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an HTTP-served Workflow05 launcher using Perfetto UI's "
            "official PING/PONG and postMessage(ArrayBuffer) interface."
        )
    )
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--attempt-manifest", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--title", default="Workflow05 selected process trace")
    parser.add_argument(
        "--perfetto-ui-url",
        default="https://ui.perfetto.dev/#!/?mode=embedded",
    )
    args = parser.parse_args()

    trace = args.trace.resolve()
    attempts_path = args.attempt_manifest.resolve()
    output_html = args.output_html.resolve()
    output_manifest = args.output_manifest.resolve()
    if not trace.is_file() or trace.stat().st_size == 0:
        raise LauncherError(f"trace is missing or empty: {trace}")
    if not attempts_path.is_file():
        raise LauncherError(f"attempt manifest is missing: {attempts_path}")
    if output_html.exists() or output_manifest.exists():
        raise LauncherError("refusing to overwrite launcher outputs")
    if len({trace.parent, output_html.parent, output_manifest.parent}) != 1:
        raise LauncherError(
            "trace, launcher HTML, and launcher manifest must share a directory"
        )
    parsed_ui = urlparse(args.perfetto_ui_url)
    loopback = {"localhost", "127.0.0.1", "::1"}
    if not parsed_ui.netloc or not parsed_ui.hostname:
        raise LauncherError("Perfetto UI URL must contain an origin")
    if parsed_ui.scheme != "https" and not (
        parsed_ui.scheme == "http" and parsed_ui.hostname in loopback
    ):
        raise LauncherError("Perfetto UI URL must use HTTPS or loopback HTTP")
    ui_origin = f"{parsed_ui.scheme}://{parsed_ui.netloc}"

    attempts = load_object(attempts_path)
    if attempts.get("policy") != "open_source_first_with_labeled_custom_fallback":
        raise LauncherError("attempt manifest has an incompatible policy")
    if attempts.get("preferred_backend_order") != BACKEND_ORDER:
        raise LauncherError("attempt manifest backend order is incompatible")
    manifest_trace = attempts.get("trace")
    if not isinstance(manifest_trace, dict):
        raise LauncherError("attempt manifest has no trace identity")
    trace_sha = sha256_file(trace)
    if (
        Path(str(manifest_trace.get("path", ""))).resolve() != trace
        or manifest_trace.get("sha256") != trace_sha
    ):
        raise LauncherError("attempt manifest trace path/hash mismatch")
    attempt_rows = attempts.get("attempts")
    if not isinstance(attempt_rows, list) or len(attempt_rows) < 3:
        raise LauncherError("attempt manifest lacks ordered interface attempts")
    actual_order = [
        row.get("backend") for row in attempt_rows if isinstance(row, dict)
    ]
    if actual_order[:3] != BACKEND_ORDER[:3]:
        raise LauncherError("attempt records do not follow the required order")
    ui_attempt = attempt_rows[2]
    if not isinstance(ui_attempt, dict) or ui_attempt.get("ui_url") != args.perfetto_ui_url:
        raise LauncherError("Perfetto UI URL differs from the recorded attempt")

    rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            html.escape(str(row.get("backend", ""))),
            html.escape(str(row.get("status", ""))),
            html.escape(str(row.get("reason", ""))),
        )
        for row in attempt_rows
        if isinstance(row, dict)
    )
    trace_url = "./" + quote(trace.name)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(args.title)}</title>
<style>body{{font:14px system-ui;margin:16px;color:#202124}}button{{padding:8px 14px}}
#status{{margin:10px 0;padding:8px;background:#eef3f8}}table{{border-collapse:collapse;width:100%;margin:12px 0}}
th,td{{border:1px solid #ccd;padding:5px;text-align:left}}iframe{{border:1px solid #bbb;width:100%;height:720px}}</style></head>
<body><h1>{html.escape(args.title)}</h1>
<p>This page is a thin adapter to the official Perfetto UI. It contains no custom timeline renderer.
Serve this directory from localhost over HTTP; <code>file://</code> is unsupported by the embedding protocol.</p>
<p><b>Trace SHA-256:</b> <code>{trace_sha}</code></p>
<button id="open">Open exact trace bytes in Perfetto UI</button>
<div id="status">Not attempted in this browser.</div>
<details><summary>Recorded parser/UI attempts</summary><table><thead><tr><th>Backend</th><th>Status</th><th>Reason</th></tr></thead><tbody>{rows}</tbody></table></details>
<iframe id="perfetto" src="{html.escape(args.perfetto_ui_url, quote=True)}"></iframe>
<script>
const iframe = document.getElementById('perfetto');
const statusBox = document.getElementById('status');
const UI_ORIGIN = {js(ui_origin)};
const TRACE_URL = {js(trace_url)};
function waitForReady() {{
  return new Promise((resolve, reject) => {{
    const timeout = setTimeout(() => {{ cleanup(); reject(new Error('Perfetto PONG timeout')); }}, 30000);
    const interval = setInterval(() => iframe.contentWindow.postMessage('PING', UI_ORIGIN), 100);
    function onMessage(event) {{
      if (event.source === iframe.contentWindow && event.origin === UI_ORIGIN && event.data === 'PONG') {{ cleanup(); resolve(); }}
    }}
    function cleanup() {{ clearTimeout(timeout); clearInterval(interval); window.removeEventListener('message', onMessage); }}
    window.addEventListener('message', onMessage);
  }});
}}
async function openTrace() {{
  statusBox.textContent = 'ATTEMPTING official Perfetto UI PING/PONG interface…';
  try {{
    await waitForReady();
    const response = await fetch(TRACE_URL, {{cache: 'no-store'}});
    if (!response.ok) throw new Error(`trace fetch failed: ${{response.status}}`);
    const buffer = await response.arrayBuffer();
    iframe.contentWindow.postMessage({{perfetto: {{buffer, title: {js(args.title)}, fileName: {js(trace.name)}, localOnly: true, keepApiOpen: true}}}}, UI_ORIGIN);
    statusBox.textContent = 'PASSED UI handshake; exact trace bytes posted. Confirm parse status in Perfetto UI.';
  }} catch (error) {{
    statusBox.textContent = `FAILED official Perfetto UI attempt: ${{error}}. Use only a visibly labeled CUSTOM FALLBACK.`;
  }}
}}
document.getElementById('open').addEventListener('click', openTrace);
</script></body></html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(page, encoding="utf-8")
    launcher_manifest = {
        "schema_version": 1,
        "status": "ready_for_browser_handshake_attempt",
        "role": "thin_official_perfetto_ui_launcher_not_custom_renderer",
        "official_interface": (
            "iframe mode=embedded; PING/PONG; "
            "postMessage({perfetto:{buffer:ArrayBuffer,title,...}})"
        ),
        "official_interface_reference": (
            "https://perfetto.dev/docs/visualization/embedding-the-ui"
        ),
        "perfetto_ui_version": "unresolved_until_runtime_ui_inspection",
        "serving_requirement": "localhost_http_not_file_url",
        "perfetto_ui_url": args.perfetto_ui_url,
        "perfetto_ui_origin": ui_origin,
        "browser_validation_status": "pending_runtime_handshake_and_ui_parse",
        "trace": {
            "path": str(trace),
            "sha256": trace_sha,
            "size_bytes": trace.stat().st_size,
        },
        "attempt_manifest": {
            "path": str(attempts_path),
            "sha256": sha256_file(attempts_path),
        },
        "launcher": {
            "path": str(output_html),
            "sha256": sha256_file(output_html),
        },
        "custom_timeline_renderer_present": False,
        "gpu_or_model_activity": False,
    }
    output_manifest.write_text(
        json.dumps(launcher_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(launcher_manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
