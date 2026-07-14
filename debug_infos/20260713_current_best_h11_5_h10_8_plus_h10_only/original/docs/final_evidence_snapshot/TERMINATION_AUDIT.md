# R25 five-hour termination audit

## Decision

The R25 goal started at epoch `1783824849`. Its five-hour threshold was
epoch `1783842849`, exactly `18000` seconds later. A post-threshold remote
observation was made at epoch `1783842949` (`2026-07-12T07:55:49Z`), for an
actual elapsed duration of `18100` seconds.

Therefore the five-hour duration condition is satisfied. The R25 goal ends
by this duration condition alone. It did **not** reach either performance
condition:

- final three-run mean score: `88.5484555040153`, below `90`;
- 20/50/30 weighted throughput improvement relative to R24:
  `+10.2361569769%`, below `+20%`.

This audit does not relabel either performance condition as passed.

## Completed validation closure

The retained candidate is H11.5 wide-causal GQA6 prefill plus H10.8 gfx936
strided LLMM1. Before the duration threshold it completed:

- three unchanged full `run_throughput.sh all` runs, totaling `450/450`
  successful requests with `failed=0`;
- TTFT and reconstructed global request-TPOT SLA checks;
- fixed `run_accuracy.sh all`, status `0`, with final results
  `77.96 / 33.05 / 100.00 / 100.00` and `K=1.00`;
- deterministic equality of `output_lens` and `generated_texts` across all
  three candidate full runs;
- healthy service checks before shutdown and a clean shutdown.

The full, accuracy, SLA, output, wheel, source, runtime, official baseline,
and R24 anchor artifacts are frozen in this evidence package.

## Post-threshold terminal-state verification

At epoch `1783842949`, after the five-hour threshold:

- `http://127.0.0.1:8001/health` returned curl HTTP code `000`;
- matching vLLM serve, vLLM benchmark, throughput, accuracy, and OpenCompass
  process count was `0`;
- port 8001 LISTEN count was `0` independently through `/proc/net/tcp*`,
  `netstat`, and `lsof`;
- the three fixed script SHA256 values still matched the frozen manifest;
- repository `git status --short` still matched the frozen source status;
- installed `_rocm_C.abi3.so` remained byte-identical to the final wheel
  member at SHA256
  `51e4839b564355279fcca4bc426ccd1da0a5f03d0e39006210960e99fd124ab1`.

The valid pre-threshold port attestation is retained at
`serve_final/port_8001_after_stop_valid.txt`; the post-threshold observations
are frozen under `termination/`. The native ROCm extension attestation is
`runtime/rocm_native_extension_attestation.txt`.

## Integrity transition

Before adding this termination audit, the final evidence manifest SHA256 was
`6d48c0fabfeef1d38e162016f312bb39f0df486bab3746c65d7bcff1a2a1ca0d`
and its identity-manifest SHA256 was
`15ca1eea38676663499c5ebdade018b2eaa2d399e80b07db9fcda5f43e5541d6`.
Those hashes describe the completed performance/accuracy package before the
five-hour audit. The top-level final manifest and identity are regenerated
after this audit and the terminal-state documentation copies are added.

Final conclusion: H11.5 + H10.8 remains the best fully closed R25 candidate,
with score `88.5484555040153`, relative R24 gain `+10.2361569769%`, SLA pass,
and `K=1.00`; the overall R25 goal terminates because elapsed time exceeded
five hours, not because the score or relative-performance thresholds passed.
