# P9 C100 fixed all3 screen

Decision: **REJECT / do not advance to full**.

- Candidate state: installed candidate wheel `f877d08f…`, H10 confirmed profile on,
  S32 off, Hg3 off. TunableOp verbose was unset; tuning and record were `0`.
- Protocol: three complete fixed `./run_throughput.sh all 3` rounds; natural active
  window `664.215310295 s`; idle padding `0`; fixed script SHA remained
  `adf0cf91266745b37df916926c7d495ec79f00a11be653c219d1d5df4d93c681`.
- Correctness/completion: 27/27 completed, failed `0`; every paired input length,
  output length and complete generated-text SHA matched the frozen baseline.
- Mean output throughput candidate vs baseline:
  - 4–8K: `13.056761833` vs `12.948554862`, `+0.835668%`.
  - 8–16K: `15.886377602` vs `15.771696925`, `+0.727130%`.
  - 16–32K: `10.030269961` vs `9.889236403`, `+1.426132%`.
  - Weighted 20/50/30: `+0.958538041%`, below the frozen `+1%` screen gate.
- SLA: pass. Maximum TTFT P99 was `1811.759/3761.892/6348.377 ms`, below
  paired limits `2775.973/5796.599/9774.263 ms`; maximum global pooled TPOT P99
  was `47.751407 ms`, below `71.536465 ms`.
- Route/profile: pass. H10 INIT/PRE_CAPTURE ready and profile SHA matched;
  S32/Hg3 used fallback with no HIT. `ResultEntry found=0`, `Finding fastest=0`,
  and no traceback, confirming verbose/tuning were absent from measurement.
- Cleanup: service stopped; runtime/8001 and source guards pass. The pinned
  baseline wheel `03568ba8…` was force-restored and verified; HCU use was `0.0%`.

Primary evidence is in `all3_natural_600/audit.json`, `service/route_audit.json`,
and `99_baseline_restore/`.
