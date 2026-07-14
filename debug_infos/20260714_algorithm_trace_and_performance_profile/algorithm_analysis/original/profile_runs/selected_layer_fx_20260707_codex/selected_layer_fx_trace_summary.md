# Selected-Layer FX Trace Summary

Run directory: `profile_runs/selected_layer_fx_20260707_codex`

| Context | Targets | Captured | FX OK | FX Errors | Missing artifacts | Benchmark |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 16-32K | `input1_layer3,input2_layer31,input4_layer59` | `input1_layer3,input2_layer31,input4_layer59` | 3 | 0 | 0 | completed=1, failed=0, input=20574, output=23 |
| 4-8K | `input1_layer3,input1_layer31,input1_layer59` | `input1_layer3,input1_layer31,input1_layer59` | 3 | 0 | 0 | completed=1, failed=0, input=7574, output=88 |
| 8-16K | `input1_layer3,input3_layer59,input4_layer31` | `input1_layer3,input3_layer59,input4_layer31` | 3 | 0 | 0 | completed=1, failed=0, input=13962, output=92 |

## Per-Context Details

### 16-32K
- metadata: `profile_runs/selected_layer_fx_20260707_codex/contexts/16-32K/fx_trace/run_metadata.json`
- manifest: `profile_runs/selected_layer_fx_20260707_codex/contexts/16-32K/fx_trace/fx_layer_trace_manifest.csv`
- observed layer events: `1856`
- fx samples/traces/errors: `3` / `3` / `0`
- node counts: `input1_layer3=155, input2_layer31=155, input4_layer59=155`

### 4-8K
- metadata: `profile_runs/selected_layer_fx_20260707_codex/contexts/4-8K/fx_trace/run_metadata.json`
- manifest: `profile_runs/selected_layer_fx_20260707_codex/contexts/4-8K/fx_trace/fx_layer_trace_manifest.csv`
- observed layer events: `5760`
- fx samples/traces/errors: `3` / `3` / `0`
- node counts: `input1_layer3=155, input1_layer31=155, input1_layer59=155`

### 8-16K
- metadata: `profile_runs/selected_layer_fx_20260707_codex/contexts/8-16K/fx_trace/run_metadata.json`
- manifest: `profile_runs/selected_layer_fx_20260707_codex/contexts/8-16K/fx_trace/fx_layer_trace_manifest.csv`
- observed layer events: `6144`
- fx samples/traces/errors: `3` / `3` / `0`
- node counts: `input1_layer3=155, input3_layer59=155, input4_layer31=155`
