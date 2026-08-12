# TP2 / concurrency 10 evaluation

> 最新调度优化、最终结果和一键构建/启动方式见
> `docs/cscc/TP2_BATCH10_BUILD_AND_RUN.md`。本文保留上一阶段的原始评测记录。

## Scope

This branch adapts the dual-card service to one TP=2, DP=1 instance and tests it
with at most 10 concurrent requests. It is an independent worktree derived from
source commit `2b4b211`; the original `pra2026-bh408-gqa-page784-k5120-batch8`
worktree was not modified.

The downloaded evaluation package used for this run is:

```text
/public/home/tangyu408/cscc-testdata-20260812/extracted
```

The service was deliberately started with that package's unmodified
`start_vllm.sh`. The small and full tests used its unmodified
`run_throughput.sh` and `run_accuracy.sh` entry points. The only throughput
override was `CONCURRENCY=10`.

## Service adaptation and safety fallbacks

- `scripts/serve_cscc_tp2_batch10.sh` mirrors the downloaded service contract:
  port 8001, TP=2, `max_model_len=32768`, `max_num_seqs=128`,
  `max_num_batched_tokens=4096`, and `gpu_memory_utilization=0.95`.
- The gfx936 packed recurrent-GEMV specialization is now restricted to its
  exact TP1 shape, width 10240. TP2 width 5120 falls back to the official packed
  decode implementation instead of entering a kernel with hard-coded TP1 head
  geometry.
- The ROCm graph safety gate accepts `(TP, DP)=(2, 1)` as well as `(1, 2)` and
  caps the exact Qwen3.5-27B dual-card graph capture sizes at 16. Larger and
  unrecognized shapes continue through normal eager/compiled fallbacks.
- The client benchmark helper defaults to concurrency 10 and keeps EOS handling
  identical to the downloaded evaluator (`ignore_eos=False`).

The clean wheel used for the run was:

```text
/public/home/tangyu408/Qwen_DCU_Worker_0/tp2_batch10_validation_20260812/wheel/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl
SHA-256 3d6a72868fb337d05639d978cf1915ad37e94bdaed04e2fc246fe4fdbd9749be
```

Source verification, wheel verification, and installed-file hash comparison
all passed before the service was started.

## Runtime evidence

The server log proves that the effective topology and limits were:

```text
tensor_parallel_size=2
data_parallel_size=1
world_size=2
cudagraph_capture_sizes=[1, 2, 4, 8, 16]
GPU KV cache size=265,776 tokens
maximum 32,768-token concurrency=30.13x
```

Both TP workers initialized on the two HCU devices. The service used the same
process continuously for the small tests, all full throughput buckets, all four
accuracy datasets, and the final recovery request.

## Small-sample gate

Before running the complete datasets, two rows from every throughput bucket and
one row from every accuracy dataset were tested:

| test | completed | failed / rc | result |
|---|---:|---:|---:|
| 4-8K throughput | 2 | 0 | pass |
| 8-16K throughput | 2 | 0 | pass |
| 16-32K throughput | 2 | 0 | pass |
| HotpotQA accuracy | 1 | 0 | 20.00 |
| GovReport accuracy | 1 | 0 | 32.81 |
| Retrieval accuracy | 1 | 0 | 100.00 |
| Aggregation accuracy | 1 | 0 | 100.00 |

## Full throughput

Command contract:

```bash
MODEL_DIR=/root/Qwen3.5-27B CONCURRENCY=10 \
  bash /public/home/tangyu408/cscc-testdata-20260812/extracted/run_throughput.sh all
```

| input bucket | completed | failed | max concurrency | duration (s) | input tokens | output tokens | output tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4-8K | 80 | 0 | 10 | 283.17 | 521,754 | 30,527 | 107.81 |
| 8-16K | 70 | 0 | 10 | 382.52 | 907,806 | 20,632 | 53.94 |
| 16-32K | 60 | 0 | 10 | 872.70 | 1,544,784 | 20,409 | 23.39 |
| **total** | **210** | **0** | **10** | **1,538.39** | **2,974,344** | **71,568** | - |

The heaviest bucket continuously replaced mixed-length requests across six
client concurrency windows without OOM, HTTP failure, or engine restart.

## Full accuracy

Command contract:

```bash
bash /public/home/tangyu408/cscc-testdata-20260812/extracted/run_accuracy.sh all
```

| dataset | rows | rc | seconds | final score |
|---|---:|---:|---:|---:|
| HotpotQA | 25 | 0 | 202 | 67.84 |
| GovReport | 30 | 0 | 626 | 32.79 |
| Retrieval multi-point | 20 | 0 | 176 | 100.00 (20/20) |
| Aggregation keyword | 20 | 0 | 545 | 75.00 (15/20) |
| **total** | **95** | **0** | **1,549** | - |

OpenCompass's intermediate exact-string Retrieval metric was 15%, because it
compared explanatory text with a bare target. The downloaded evaluator's final
RULER scorer correctly found every requested identifier and reported 20/20.
This is an evaluator-stage distinction, not a service accuracy defect.

The five Aggregation misses were different: all five responses reached exactly
1024 output tokens with `finish_reason=length`. The model emitted verbose
per-word index/count explanations and was truncated before listing all ten
answers. The other 15 responses stopped naturally and passed. This is not an
OOM, TP2 corruption, or scorer parsing failure. The service does not recognize
evaluation prompts or inject task-specific instructions.

## Failure and recovery audit

- Full server-log scans found no OOM, traceback, engine-core failure, HTTP 5xx,
  or fatal error.
- Expected warnings were limited to an unknown profile metadata variable, CPU
  thread reduction, safe ROCm GELU fallback, a no-effect optional compile op,
  generation-config defaults, and the existing short-sequence FLA format
  heuristic. Requests associated with the latter returned HTTP 200 correctly.
- After all throughput and accuracy tests, a fresh request returned HTTP 200,
  exact text `SERVICE_RECOVERY_OK`, and `finish_reason=stop`.
- The service then shut down normally. No vLLM process remained, and both cards
  returned to 2 MiB used VRAM.

## Evidence locations

Persistent runtime artifacts are outside the source repository:

```text
/public/home/tangyu408/Qwen_DCU_Worker_0/tp2_batch10_validation_20260812/
  server_downloaded_start.log
  small/
  full/throughput/{4-8K,8-16K,16-32K}-result.json
  full/throughput/client.log
  full/accuracy-client.log
  full/accuracy_runtime/work_accuracy_only_20260813_000527_192410/
  full/service-recovery.json
  full/service-recovery.status
  wheel/
```

The repository intentionally does not include the model, downloaded evaluator,
wheel, raw predictions, or large benchmark logs.
