# GQA6 page784 / K5120 validation evidence

This directory archives the tests and experiment outputs for the focused
Qwen3.5 ROCm branch `repro-gqa-page784-k5120`.

## Contents

- `final-dp2-validation/`: full DP=2 accuracy, throughput, service, operator,
  build, and runtime-kernel evidence produced from source commit `5355cea`.
- `operator-ablations/gqa6-readable-direct/`: focused GQA6/page784 kernel and
  operator A/B results collected before the final validation.
- `probes/`: standalone page784 layout and parity probes used during the
  implementation experiments.

The final validation directory preserves the raw benchmark inputs and outputs
alongside `FINAL_REPORT.md` and `summary.json`. Its `MANIFEST.sha256` uses paths
relative to that directory so the archive can be verified after checkout:

```bash
cd evidence/gqa-page784-k5120/final-dp2-validation
sha256sum --check MANIFEST.sha256
```

Large build products, installed Python trees, runtime caches, and wheel files
are intentionally excluded. The artifact hashes and build log needed for
provenance remain in `final-dp2-validation/provenance/`.
