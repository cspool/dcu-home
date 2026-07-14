# Selected-Layer FX Process Reconstruction Summary

Run directory: `profile_runs/selected_layer_fx_20260707_codex`
Events reconstructed: `9`
Node counts: `[155]`
Process counts: `[12]`
All nodes assigned: `True`
No duplicate assignments: `True`

| Context | Event | Layer | Forward | q_len | Nodes | Processes | Unassigned |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 16-32K | `input1_layer3` | 3 | 1 | 4096 | 155 | 12 | `[]` |
| 16-32K | `input2_layer31` | 31 | 2 | 4096 | 155 | 12 | `[]` |
| 16-32K | `input4_layer59` | 59 | 4 | 4096 | 155 | 12 | `[]` |
| 4-8K | `input1_layer3` | 3 | 1 | 4096 | 155 | 12 | `[]` |
| 4-8K | `input1_layer31` | 31 | 1 | 4096 | 155 | 12 | `[]` |
| 4-8K | `input1_layer59` | 59 | 1 | 4096 | 155 | 12 | `[]` |
| 8-16K | `input1_layer3` | 3 | 1 | 4096 | 155 | 12 | `[]` |
| 8-16K | `input3_layer59` | 59 | 3 | 4096 | 155 | 12 | `[]` |
| 8-16K | `input4_layer31` | 31 | 4 | 1685 | 155 | 12 | `[]` |

Per-event outputs:

- `16-32K/input1_layer3`: `profile_runs/selected_layer_fx_20260707_codex/contexts/16-32K/fx_trace/traces/input1_layer3`
- `16-32K/input2_layer31`: `profile_runs/selected_layer_fx_20260707_codex/contexts/16-32K/fx_trace/traces/input2_layer31`
- `16-32K/input4_layer59`: `profile_runs/selected_layer_fx_20260707_codex/contexts/16-32K/fx_trace/traces/input4_layer59`
- `4-8K/input1_layer3`: `profile_runs/selected_layer_fx_20260707_codex/contexts/4-8K/fx_trace/traces/input1_layer3`
- `4-8K/input1_layer31`: `profile_runs/selected_layer_fx_20260707_codex/contexts/4-8K/fx_trace/traces/input1_layer31`
- `4-8K/input1_layer59`: `profile_runs/selected_layer_fx_20260707_codex/contexts/4-8K/fx_trace/traces/input1_layer59`
- `8-16K/input1_layer3`: `profile_runs/selected_layer_fx_20260707_codex/contexts/8-16K/fx_trace/traces/input1_layer3`
- `8-16K/input3_layer59`: `profile_runs/selected_layer_fx_20260707_codex/contexts/8-16K/fx_trace/traces/input3_layer59`
- `8-16K/input4_layer31`: `profile_runs/selected_layer_fx_20260707_codex/contexts/8-16K/fx_trace/traces/input4_layer31`
