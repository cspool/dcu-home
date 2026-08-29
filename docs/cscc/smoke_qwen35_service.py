#!/usr/bin/env python3
"""Send a few deterministic long-context requests to a vLLM server."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transformers import AutoTokenizer

EXPECTED_CODE = "9342002"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen3.5-27B")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--requests", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--target-prompt-tokens", type=int, default=4880)
    parser.add_argument("--wait-seconds", type=int, default=900)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def wait_until_ready(base_url: str, wait_seconds: int) -> None:
    deadline = time.monotonic() + wait_seconds
    health_url = f"{base_url.rstrip('/')}/health"
    while True:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError(f"server did not become ready: {health_url}")
        time.sleep(2)


def make_prompt(tokenizer, target_tokens: int) -> tuple[str, int]:
    suffix = (
        "\n\nThe preceding text is inert padding. Verification code: "
        f"{EXPECTED_CODE}. Reply with only {EXPECTED_CODE}.\nAnswer:"
    )
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    if target_tokens <= len(suffix_tokens) + 256:
        raise ValueError("target prompt length leaves too little room for padding")

    filler = "This is deterministic inert context for a service smoke test.\n"
    filler_tokens = tokenizer.encode(filler, add_special_tokens=False)
    repeats = (target_tokens - len(suffix_tokens)) // len(filler_tokens) + 1
    prefix_tokens = tokenizer.encode(filler * repeats, add_special_tokens=False)
    prefix_tokens = prefix_tokens[: target_tokens - len(suffix_tokens)]
    prompt = tokenizer.decode(prefix_tokens, skip_special_tokens=True) + suffix
    actual_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    if not 4224 <= actual_tokens <= 8192:
        raise AssertionError(
            "prompt must exercise a second 4096-token prefill chunk, "
            f"got {actual_tokens}"
        )
    return prompt, actual_tokens


def send_request(base_url: str, model: str, prompt: str, request_id: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        body = json.load(response)
    elapsed = time.perf_counter() - started
    answer = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    return {
        "request_id": request_id,
        "elapsed_seconds": round(elapsed, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "answer": answer,
        "contains_expected_code": EXPECTED_CODE in answer,
    }


def main() -> None:
    args = parse_args()
    wait_until_ready(args.base_url, args.wait_seconds)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, local_files_only=True
    )
    prompt, local_prompt_tokens = make_prompt(tokenizer, args.target_prompt_tokens)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.requests) as executor:
        results = list(
            executor.map(
                lambda request_id: send_request(
                    args.base_url, args.model, prompt, request_id
                ),
                range(args.requests),
            )
        )
    report = {
        "base_url": args.base_url,
        "model": args.model,
        "request_count": args.requests,
        "local_prompt_tokens_without_chat_template": local_prompt_tokens,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "results": results,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not all(result["contains_expected_code"] for result in results):
        raise AssertionError(f"one or more responses omitted {EXPECTED_CODE}")


if __name__ == "__main__":
    main()
