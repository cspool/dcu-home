# Batch-eight experiment and validation evidence

This directory archives the evidence used to evaluate the DP2, global-batch-eight
implementation on two DCUs (local batch four per DP rank).

- `official_ablation_20260811/` contains the microbenchmark scripts, raw JSON,
  logs, and the ablation summary against the official implementation.
- `implementation_validation_20260811/` contains kernel probes, service logs,
  throughput runs, scheduler experiments, and accuracy-validation outputs.
- `gqa_bm32_validation_79a20b0/` contains the final GQA BM32 service results,
  runtime logs, summary, and SHA-256 manifest.

The archive intentionally excludes generated wheels, installed Python trees,
shared objects, Triton/build caches, `__pycache__`, and the copied official source
tree. Those artifacts are large and reproducible from the tracked source; none is
required to inspect the recorded measurements. The official launch and benchmark
scripts were not modified to create this archive.

The GQA directory keeps its original `MANIFEST.sha256`, including the omitted
wheel entry, for provenance. Use `MANIFEST_ARCHIVE.sha256` to verify every file
from that manifest which is actually stored in Git.
