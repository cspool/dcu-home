# Patch Trace Summary

Run directory: `remote-home/testdata/profile_runs/patch_trace_20260707_codex`

| Context | Events | Prefill chunks | Decode steps | Finished outputs | Patch errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16-32K | 554 | 6 [4096, 4096, 4096, 4096, 4096, 105] | 23 | 1 | 0 |
| 4-8K | 1158 | 2 [4096, 3489] | 88 | 1 | 0 |
| 8-16K | 1294 | 4 [4096, 4096, 4096, 1685] | 92 | 1 | 0 |

## Validation

### 16-32K
- `benchmark_result`: `{'completed': 1, 'failed': 0, 'total_input_tokens': 20574, 'total_output_tokens': 23, 'mean_ttft_ms': 36891.82796981186, 'mean_tpot_ms': 74.27299105223608, 'mean_e2el_ms': 38525.83377296105}`
- `has_events`: `True`
- `missing_required_event_types`: `[]`
- `missing_join_keys`: `{}`
- `patch_errors`: `0`
- `benchmark_completed`: `True`
- `benchmark_failed_zero`: `True`
- `prefill_chunks_match_expected`: `True`
- `decode_steps_match_expected`: `True`
- `has_finished_output`: `True`
- `event_type_counts`: `{'attention_batch_metadata': 29, 'attention_forward_begin': 96, 'attention_forward_end': 96, 'batch_constructed': 29, 'engine_step_begin': 31, 'engine_step_end': 31, 'kv_allocate_slots': 29, 'kv_get_computed_blocks': 1, 'kv_take_new_block_ids': 30, 'model_execute_begin': 30, 'model_execute_end': 30, 'patch_loaded': 4, 'sample_tokens': 29, 'sampler_call': 29, 'scheduler_step': 30, 'scheduler_update_output': 30}`
- `phase_counts`: `{'decode': 23, 'empty': 1, 'prefill_chunk': 6}`
- `output_token_total_by_req`: `{'chatcmpl-bench-f596a9c5-0-a5c840c5': 23}`
- `finished_outputs`: `[{'client_index': 0, 'finish_reason': 'stop', 'finished': True, 'new_token_count': 1, 'new_token_head': [248046], 'num_cached_tokens': 0, 'request_id': 'chatcmpl-bench-f596a9c5-0-a5c840c5', 'stop_reason': None}]`

### 4-8K
- `benchmark_result`: `{'completed': 1, 'failed': 0, 'total_input_tokens': 7574, 'total_output_tokens': 88, 'mean_ttft_ms': 16609.11333281547, 'mean_tpot_ms': 72.4827829084691, 'mean_e2el_ms': 22915.11544585228}`
- `has_events`: `True`
- `missing_required_event_types`: `[]`
- `missing_join_keys`: `{}`
- `patch_errors`: `0`
- `benchmark_completed`: `True`
- `benchmark_failed_zero`: `True`
- `prefill_chunks_match_expected`: `True`
- `decode_steps_match_expected`: `True`
- `has_finished_output`: `True`
- `event_type_counts`: `{'attention_batch_metadata': 90, 'attention_forward_begin': 32, 'attention_forward_end': 32, 'batch_constructed': 90, 'engine_step_begin': 92, 'engine_step_end': 92, 'kv_allocate_slots': 90, 'kv_get_computed_blocks': 1, 'kv_take_new_block_ids': 91, 'model_execute_begin': 91, 'model_execute_end': 91, 'patch_loaded': 4, 'sample_tokens': 90, 'sampler_call': 90, 'scheduler_step': 91, 'scheduler_update_output': 91}`
- `phase_counts`: `{'decode': 88, 'empty': 1, 'prefill_chunk': 2}`
- `output_token_total_by_req`: `{'chatcmpl-bench-0995cee0-0-a5dd8ba0': 88}`
- `finished_outputs`: `[{'client_index': 0, 'finish_reason': 'stop', 'finished': True, 'new_token_count': 1, 'new_token_head': [248046], 'num_cached_tokens': 0, 'request_id': 'chatcmpl-bench-0995cee0-0-a5dd8ba0', 'stop_reason': None}]`

### 8-16K
- `benchmark_result`: `{'completed': 1, 'failed': 0, 'total_input_tokens': 13962, 'total_output_tokens': 92, 'mean_ttft_ms': 24588.929723016918, 'mean_tpot_ms': 73.6230779899755, 'mean_e2el_ms': 31288.62982010469}`
- `has_events`: `True`
- `missing_required_event_types`: `[]`
- `missing_join_keys`: `{}`
- `patch_errors`: `0`
- `benchmark_completed`: `True`
- `benchmark_failed_zero`: `True`
- `prefill_chunks_match_expected`: `True`
- `decode_steps_match_expected`: `True`
- `has_finished_output`: `True`
- `event_type_counts`: `{'attention_batch_metadata': 96, 'attention_forward_begin': 64, 'attention_forward_end': 64, 'batch_constructed': 96, 'engine_step_begin': 98, 'engine_step_end': 98, 'kv_allocate_slots': 96, 'kv_get_computed_blocks': 1, 'kv_take_new_block_ids': 97, 'model_execute_begin': 97, 'model_execute_end': 97, 'patch_loaded': 4, 'sample_tokens': 96, 'sampler_call': 96, 'scheduler_step': 97, 'scheduler_update_output': 97}`
- `phase_counts`: `{'decode': 92, 'empty': 1, 'prefill_chunk': 4}`
- `output_token_total_by_req`: `{'chatcmpl-bench-607a754c-0-81e80afa': 92}`
- `finished_outputs`: `[{'client_index': 0, 'finish_reason': 'stop', 'finished': True, 'new_token_count': 1, 'new_token_head': [248046], 'num_cached_tokens': 0, 'request_id': 'chatcmpl-bench-607a754c-0-81e80afa', 'stop_reason': None}]`
