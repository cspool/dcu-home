# FX Process Reconstruction

Event: `input4_layer59`
Context: `16-32K`
Layer: `59` (`full_attention`)
Forward/phase/q_len: `4` / `prefill_chunk` / `4096`
Node coverage: `155/155`

This file is reconstruction evidence only: process labels are rule-based labels over the fixed-input FX DAG.

## Process Table

| Order | Process | Node ranges | Targets |
| ---: | --- | --- | ---: |
| 1 | `runtime_inputs` Runtime FX inputs | `0-2` | 3 |
| 2 | `pre_attention_residual_rmsnorm` Pre-attention residual add and input RMSNorm | `3-15` | 8 |
| 3 | `qkv_projection_and_split` Fused Q/gate/K/V projection and head reshape | `17-31` | 9 |
| 4 | `q_head_rmsnorm` Q head RMSNorm | `32-45` | 9 |
| 5 | `k_head_rmsnorm` K head RMSNorm | `46-59` | 9 |
| 6 | `mrope_table_lookup` MROPE cos/sin table lookup and axis remap | `60-84` | 8 |
| 7 | `q_rope_apply` Q RoPE application | `85-101` | 9 |
| 8 | `k_rope_apply` K RoPE application | `102-118` | 9 |
| 9 | `vllm_attention_and_kv_cache` vLLM attention and KV cache update | `119-125` | 4 |
| 10 | `attention_gate_projection_residual` Attention gate, output projection, and residual | `16, 126-132, 135` | 9 |
| 11 | `post_attention_rmsnorm` Post-attention RMSNorm | `133-134, 136-145` | 8 |
| 12 | `mlp_and_layer_output` MLP and layer output tuple | `146-154` | 7 |

## 1. Runtime FX inputs

- process_id: `runtime_inputs`
- rule: Fixed sampled layer inputs and control tensors exposed as FX placeholders.
- node ranges: `0-2`
- target set: `arg0_1, arg1_1, arg2_1`

```text
#000 arg0_1                         placeholder    arg0_1                                        shape=[3, 4096] dtype=torch.int64 users=index
#001 arg1_1                         placeholder    arg1_1                                        shape=[4096, 5120] dtype=torch.bfloat16 users=add
#002 arg2_1                         placeholder    arg2_1                                        shape=[4096, 5120] dtype=torch.bfloat16 users=add
```

## 2. Pre-attention residual add and input RMSNorm

- process_id: `pre_attention_residual_rmsnorm`
- rule: Combine hidden_states with residual, then run RMSNorm over the Hidden axis.
- node ranges: `3-15`
- target set: `_param_constant0, aten._to_copy.default, aten.add.Tensor, aten.detach.default, aten.mean.dim, aten.mul.Tensor, aten.pow.Tensor_Scalar, aten.rsqrt.default`

```text
#003 _param_constant0               get_attr       _param_constant0                              shape=[5120] dtype=torch.bfloat16 users=detach
#004 detach                         call_function  aten.detach.default                           shape=[5120] dtype=torch.bfloat16 users=_to_copy_1
#005 add                            call_function  aten.add.Tensor                               shape=[4096, 5120] dtype=torch.bfloat16 users=_to_copy,add_9
#006 _to_copy                       call_function  aten._to_copy.default                         shape=[4096, 5120] dtype=torch.float32 users=mul,pow_1
#007 pow_1                          call_function  aten.pow.Tensor_Scalar                        shape=[4096, 5120] dtype=torch.float32 users=mean
#008 mean                           call_function  aten.mean.dim                                 shape=[4096, 1] dtype=torch.float32 users=add_1
#009 add_1                          call_function  aten.add.Tensor                               shape=[4096, 1] dtype=torch.float32 users=rsqrt
#010 rsqrt                          call_function  aten.rsqrt.default                            shape=[4096, 1] dtype=torch.float32 users=mul
#011 mul                            call_function  aten.mul.Tensor                               shape=[4096, 5120] dtype=torch.float32 users=mul_1
#012 _to_copy_1                     call_function  aten._to_copy.default                         shape=[5120] dtype=torch.float32 users=add_2
#013 add_2                          call_function  aten.add.Tensor                               shape=[5120] dtype=torch.float32 users=mul_1
#014 mul_1                          call_function  aten.mul.Tensor                               shape=[4096, 5120] dtype=torch.float32 users=_to_copy_2
#015 _to_copy_2                     call_function  aten._to_copy.default                         shape=[4096, 5120] dtype=torch.bfloat16 users=empty_like,mm
```

## 3. Fused Q/gate/K/V projection and head reshape

- process_id: `qkv_projection_and_split`
- rule: Project normalized hidden states, split fused projection into Q/gate/K/V branches, and reshape Q heads.
- node ranges: `17-31`
- target set: `<built-in function getitem>, _param_constant1, aten._unsafe_view.default, aten.clone.default, aten.mm.default, aten.split.Tensor, aten.split_with_sizes.default, aten.t.default, aten.view.default`

```text
#017 _param_constant1               get_attr       _param_constant1                              shape=[14336, 5120] dtype=torch.bfloat16 users=t
#018 t                              call_function  aten.t.default                                shape=[5120, 14336] dtype=torch.bfloat16 users=mm
#019 mm                             call_function  aten.mm.default                               shape=[4096, 14336] dtype=torch.bfloat16 users=split_with_sizes
#020 split_with_sizes               call_function  aten.split_with_sizes.default                 shape=- dtype=- users=getitem,getitem_1,getitem_2
#021 getitem                        call_function  <built-in function getitem>                   shape=[4096, 12288] dtype=torch.bfloat16 users=view
#022 getitem_1                      call_function  <built-in function getitem>                   shape=[4096, 1024] dtype=torch.bfloat16 users=view_3
#023 getitem_2                      call_function  <built-in function getitem>                   shape=[4096, 1024] dtype=torch.bfloat16 users=view_12
#024 view                           call_function  aten.view.default                             shape=[4096, 24, 512] dtype=torch.bfloat16 users=split
#025 split                          call_function  aten.split.Tensor                             shape=- dtype=- users=getitem_3,getitem_4
#026 getitem_3                      call_function  <built-in function getitem>                   shape=[4096, 24, 256] dtype=torch.bfloat16 users=clone
#027 getitem_4                      call_function  <built-in function getitem>                   shape=[4096, 24, 256] dtype=torch.bfloat16 users=clone_1
#028 clone                          call_function  aten.clone.default                            shape=[4096, 24, 256] dtype=torch.bfloat16 users=_unsafe_view
#029 _unsafe_view                   call_function  aten._unsafe_view.default                     shape=[4096, 6144] dtype=torch.bfloat16 users=view_1
#030 clone_1                        call_function  aten.clone.default                            shape=[4096, 24, 256] dtype=torch.bfloat16 users=_unsafe_view_1
#031 _unsafe_view_1                 call_function  aten._unsafe_view.default                     shape=[4096, 6144] dtype=torch.bfloat16 users=sigmoid
```

## 4. Q head RMSNorm

- process_id: `q_head_rmsnorm`
- rule: Normalize Q head vectors over head_dim before rotary embedding.
- node ranges: `32-45`
- target set: `_param_constant2, aten._to_copy.default, aten.add.Tensor, aten.detach.default, aten.mean.dim, aten.mul.Tensor, aten.pow.Tensor_Scalar, aten.rsqrt.default, aten.view.default`

```text
#032 view_1                         call_function  aten.view.default                             shape=[4096, 24, 256] dtype=torch.bfloat16 users=_to_copy_3
#033 _param_constant2               get_attr       _param_constant2                              shape=[256] dtype=torch.bfloat16 users=detach_1
#034 detach_1                       call_function  aten.detach.default                           shape=[256] dtype=torch.bfloat16 users=_to_copy_4
#035 _to_copy_3                     call_function  aten._to_copy.default                         shape=[4096, 24, 256] dtype=torch.float32 users=mul_2,pow_2
#036 pow_2                          call_function  aten.pow.Tensor_Scalar                        shape=[4096, 24, 256] dtype=torch.float32 users=mean_1
#037 mean_1                         call_function  aten.mean.dim                                 shape=[4096, 24, 1] dtype=torch.float32 users=add_3
#038 add_3                          call_function  aten.add.Tensor                               shape=[4096, 24, 1] dtype=torch.float32 users=rsqrt_1
#039 rsqrt_1                        call_function  aten.rsqrt.default                            shape=[4096, 24, 1] dtype=torch.float32 users=mul_2
#040 mul_2                          call_function  aten.mul.Tensor                               shape=[4096, 24, 256] dtype=torch.float32 users=mul_3
#041 _to_copy_4                     call_function  aten._to_copy.default                         shape=[256] dtype=torch.float32 users=add_4
#042 add_4                          call_function  aten.add.Tensor                               shape=[256] dtype=torch.float32 users=mul_3
#043 mul_3                          call_function  aten.mul.Tensor                               shape=[4096, 24, 256] dtype=torch.float32 users=_to_copy_5
#044 _to_copy_5                     call_function  aten._to_copy.default                         shape=[4096, 24, 256] dtype=torch.bfloat16 users=view_2
#045 view_2                         call_function  aten.view.default                             shape=[4096, 6144] dtype=torch.bfloat16 users=view_5
```

## 5. K head RMSNorm

- process_id: `k_head_rmsnorm`
- rule: Normalize K head vectors over head_dim before rotary embedding.
- node ranges: `46-59`
- target set: `_param_constant3, aten._to_copy.default, aten.add.Tensor, aten.detach.default, aten.mean.dim, aten.mul.Tensor, aten.pow.Tensor_Scalar, aten.rsqrt.default, aten.view.default`

```text
#046 view_3                         call_function  aten.view.default                             shape=[4096, 4, 256] dtype=torch.bfloat16 users=_to_copy_6
#047 _param_constant3               get_attr       _param_constant3                              shape=[256] dtype=torch.bfloat16 users=detach_2
#048 detach_2                       call_function  aten.detach.default                           shape=[256] dtype=torch.bfloat16 users=_to_copy_7
#049 _to_copy_6                     call_function  aten._to_copy.default                         shape=[4096, 4, 256] dtype=torch.float32 users=mul_4,pow_3
#050 pow_3                          call_function  aten.pow.Tensor_Scalar                        shape=[4096, 4, 256] dtype=torch.float32 users=mean_2
#051 mean_2                         call_function  aten.mean.dim                                 shape=[4096, 4, 1] dtype=torch.float32 users=add_5
#052 add_5                          call_function  aten.add.Tensor                               shape=[4096, 4, 1] dtype=torch.float32 users=rsqrt_2
#053 rsqrt_2                        call_function  aten.rsqrt.default                            shape=[4096, 4, 1] dtype=torch.float32 users=mul_4
#054 mul_4                          call_function  aten.mul.Tensor                               shape=[4096, 4, 256] dtype=torch.float32 users=mul_5
#055 _to_copy_7                     call_function  aten._to_copy.default                         shape=[256] dtype=torch.float32 users=add_6
#056 add_6                          call_function  aten.add.Tensor                               shape=[256] dtype=torch.float32 users=mul_5
#057 mul_5                          call_function  aten.mul.Tensor                               shape=[4096, 4, 256] dtype=torch.float32 users=_to_copy_8
#058 _to_copy_8                     call_function  aten._to_copy.default                         shape=[4096, 4, 256] dtype=torch.bfloat16 users=view_4
#059 view_4                         call_function  aten.view.default                             shape=[4096, 1024] dtype=torch.bfloat16 users=view_7
```

## 6. MROPE cos/sin table lookup and axis remap

- process_id: `mrope_table_lookup`
- rule: Index rotary tables with sampled positions, split cos/sin halves, and remap MROPE axis slices.
- node ranges: `60-84`
- target set: `<built-in function getitem>, _tensor_constant0, aten.clone.default, aten.copy_.default, aten.index.Tensor, aten.select.int, aten.slice.Tensor, aten.split.Tensor`

```text
#060 _tensor_constant0              get_attr       _tensor_constant0                             shape=[1048576, 64] dtype=torch.bfloat16 users=index
#061 index                          call_function  aten.index.Tensor                             shape=[3, 4096, 64] dtype=torch.bfloat16 users=split_1
#062 split_1                        call_function  aten.split.Tensor                             shape=- dtype=- users=getitem_5,getitem_6
#063 getitem_5                      call_function  <built-in function getitem>                   shape=[3, 4096, 32] dtype=torch.bfloat16 users=select,select_1,select_2
#064 getitem_6                      call_function  <built-in function getitem>                   shape=[3, 4096, 32] dtype=torch.bfloat16 users=select_3,select_4,select_5
#065 select                         call_function  aten.select.int                               shape=[4096, 32] dtype=torch.bfloat16 users=clone_2
#066 clone_2                        call_function  aten.clone.default                            shape=[4096, 32] dtype=torch.bfloat16 users=slice_2,slice_4,unsqueeze,unsqueeze_2
#067 select_1                       call_function  aten.select.int                               shape=[4096, 32] dtype=torch.bfloat16 users=slice_1
#068 slice_1                        call_function  aten.slice.Tensor                             shape=[4096, 11] dtype=torch.bfloat16 users=copy_
#069 slice_2                        call_function  aten.slice.Tensor                             shape=[4096, 11] dtype=torch.bfloat16 users=copy_
#070 copy_                          call_function  aten.copy_.default                            shape=[4096, 11] dtype=torch.bfloat16 users=-
#071 select_2                       call_function  aten.select.int                               shape=[4096, 32] dtype=torch.bfloat16 users=slice_3
#072 slice_3                        call_function  aten.slice.Tensor                             shape=[4096, 10] dtype=torch.bfloat16 users=copy__1
#073 slice_4                        call_function  aten.slice.Tensor                             shape=[4096, 10] dtype=torch.bfloat16 users=copy__1
#074 copy__1                        call_function  aten.copy_.default                            shape=[4096, 10] dtype=torch.bfloat16 users=-
#075 select_3                       call_function  aten.select.int                               shape=[4096, 32] dtype=torch.bfloat16 users=clone_3
#076 clone_3                        call_function  aten.clone.default                            shape=[4096, 32] dtype=torch.bfloat16 users=slice_6,slice_8,unsqueeze_1,unsqueeze_3
#077 select_4                       call_function  aten.select.int                               shape=[4096, 32] dtype=torch.bfloat16 users=slice_5
#078 slice_5                        call_function  aten.slice.Tensor                             shape=[4096, 11] dtype=torch.bfloat16 users=copy__2
#079 slice_6                        call_function  aten.slice.Tensor                             shape=[4096, 11] dtype=torch.bfloat16 users=copy__2
#080 copy__2                        call_function  aten.copy_.default                            shape=[4096, 11] dtype=torch.bfloat16 users=-
#081 select_5                       call_function  aten.select.int                               shape=[4096, 32] dtype=torch.bfloat16 users=slice_7
#082 slice_7                        call_function  aten.slice.Tensor                             shape=[4096, 10] dtype=torch.bfloat16 users=copy__3
#083 slice_8                        call_function  aten.slice.Tensor                             shape=[4096, 10] dtype=torch.bfloat16 users=copy__3
#084 copy__3                        call_function  aten.copy_.default                            shape=[4096, 10] dtype=torch.bfloat16 users=-
```

## 7. Q RoPE application

- process_id: `q_rope_apply`
- rule: Apply rotary transform to the normalized Q branch and restore flattened Q layout.
- node ranges: `85-101`
- target set: `<built-in function getitem>, aten.add.Tensor, aten.cat.default, aten.mul.Tensor, aten.slice.Tensor, aten.split.Tensor, aten.sub.Tensor, aten.unsqueeze.default, aten.view.default`

```text
#085 view_5                         call_function  aten.view.default                             shape=[4096, 24, 256] dtype=torch.bfloat16 users=slice_10,slice_9
#086 slice_9                        call_function  aten.slice.Tensor                             shape=[4096, 24, 64] dtype=torch.bfloat16 users=split_2
#087 slice_10                       call_function  aten.slice.Tensor                             shape=[4096, 24, 192] dtype=torch.bfloat16 users=cat_1
#088 unsqueeze                      call_function  aten.unsqueeze.default                        shape=[4096, 1, 32] dtype=torch.bfloat16 users=mul_6,mul_8
#089 unsqueeze_1                    call_function  aten.unsqueeze.default                        shape=[4096, 1, 32] dtype=torch.bfloat16 users=mul_7,mul_9
#090 split_2                        call_function  aten.split.Tensor                             shape=- dtype=- users=getitem_7,getitem_8
#091 getitem_7                      call_function  <built-in function getitem>                   shape=[4096, 24, 32] dtype=torch.bfloat16 users=mul_6,mul_9
#092 getitem_8                      call_function  <built-in function getitem>                   shape=[4096, 24, 32] dtype=torch.bfloat16 users=mul_7,mul_8
#093 mul_6                          call_function  aten.mul.Tensor                               shape=[4096, 24, 32] dtype=torch.bfloat16 users=sub
#094 mul_7                          call_function  aten.mul.Tensor                               shape=[4096, 24, 32] dtype=torch.bfloat16 users=sub
#095 sub                            call_function  aten.sub.Tensor                               shape=[4096, 24, 32] dtype=torch.bfloat16 users=cat
#096 mul_8                          call_function  aten.mul.Tensor                               shape=[4096, 24, 32] dtype=torch.bfloat16 users=add_7
#097 mul_9                          call_function  aten.mul.Tensor                               shape=[4096, 24, 32] dtype=torch.bfloat16 users=add_7
#098 add_7                          call_function  aten.add.Tensor                               shape=[4096, 24, 32] dtype=torch.bfloat16 users=cat
#099 cat                            call_function  aten.cat.default                              shape=[4096, 24, 64] dtype=torch.bfloat16 users=cat_1
#100 cat_1                          call_function  aten.cat.default                              shape=[4096, 24, 256] dtype=torch.bfloat16 users=view_6
#101 view_6                         call_function  aten.view.default                             shape=[4096, 6144] dtype=torch.bfloat16 users=view_9
```

## 8. K RoPE application

- process_id: `k_rope_apply`
- rule: Apply rotary transform to the normalized K branch and restore flattened K layout.
- node ranges: `102-118`
- target set: `<built-in function getitem>, aten.add.Tensor, aten.cat.default, aten.mul.Tensor, aten.slice.Tensor, aten.split.Tensor, aten.sub.Tensor, aten.unsqueeze.default, aten.view.default`

```text
#102 view_7                         call_function  aten.view.default                             shape=[4096, 4, 256] dtype=torch.bfloat16 users=slice_11,slice_12
#103 slice_11                       call_function  aten.slice.Tensor                             shape=[4096, 4, 64] dtype=torch.bfloat16 users=split_3
#104 slice_12                       call_function  aten.slice.Tensor                             shape=[4096, 4, 192] dtype=torch.bfloat16 users=cat_3
#105 unsqueeze_2                    call_function  aten.unsqueeze.default                        shape=[4096, 1, 32] dtype=torch.bfloat16 users=mul_10,mul_12
#106 unsqueeze_3                    call_function  aten.unsqueeze.default                        shape=[4096, 1, 32] dtype=torch.bfloat16 users=mul_11,mul_13
#107 split_3                        call_function  aten.split.Tensor                             shape=- dtype=- users=getitem_10,getitem_9
#108 getitem_9                      call_function  <built-in function getitem>                   shape=[4096, 4, 32] dtype=torch.bfloat16 users=mul_10,mul_13
#109 getitem_10                     call_function  <built-in function getitem>                   shape=[4096, 4, 32] dtype=torch.bfloat16 users=mul_11,mul_12
#110 mul_10                         call_function  aten.mul.Tensor                               shape=[4096, 4, 32] dtype=torch.bfloat16 users=sub_1
#111 mul_11                         call_function  aten.mul.Tensor                               shape=[4096, 4, 32] dtype=torch.bfloat16 users=sub_1
#112 sub_1                          call_function  aten.sub.Tensor                               shape=[4096, 4, 32] dtype=torch.bfloat16 users=cat_2
#113 mul_12                         call_function  aten.mul.Tensor                               shape=[4096, 4, 32] dtype=torch.bfloat16 users=add_8
#114 mul_13                         call_function  aten.mul.Tensor                               shape=[4096, 4, 32] dtype=torch.bfloat16 users=add_8
#115 add_8                          call_function  aten.add.Tensor                               shape=[4096, 4, 32] dtype=torch.bfloat16 users=cat_2
#116 cat_2                          call_function  aten.cat.default                              shape=[4096, 4, 64] dtype=torch.bfloat16 users=cat_3
#117 cat_3                          call_function  aten.cat.default                              shape=[4096, 4, 256] dtype=torch.bfloat16 users=view_8
#118 view_8                         call_function  aten.view.default                             shape=[4096, 1024] dtype=torch.bfloat16 users=view_11
```

## 9. vLLM attention and KV cache update

- process_id: `vllm_attention_and_kv_cache`
- rule: View Q/K/V/output buffers by head, update KV cache, then call vLLM attention output custom op.
- node ranges: `119-125`
- target set: `aten.empty.memory_format, aten.view.default, vllm.unified_attention_with_output.default, vllm.unified_kv_cache_update.default`

```text
#119 empty                          call_function  aten.empty.memory_format                      shape=[4096, 6144] dtype=torch.bfloat16 users=view_10
#120 view_9                         call_function  aten.view.default                             shape=[4096, 24, 256] dtype=torch.bfloat16 users=unified_attention_with_output
#121 view_10                        call_function  aten.view.default                             shape=[4096, 24, 256] dtype=torch.bfloat16 users=unified_attention_with_output,view_13
#122 view_11                        call_function  aten.view.default                             shape=[4096, 4, 256] dtype=torch.bfloat16 users=unified_attention_with_output,unified_kv_cache_update
#123 view_12                        call_function  aten.view.default                             shape=[4096, 4, 256] dtype=torch.bfloat16 users=unified_attention_with_output,unified_kv_cache_update
#124 unified_kv_cache_update        call_function  vllm.unified_kv_cache_update.default          shape=[0] dtype=torch.bfloat16 users=unified_attention_with_output
#125 unified_attention_with_output  call_function  vllm.unified_attention_with_output.default    shape=- dtype=- users=-
```

## 10. Attention gate, output projection, and residual

- process_id: `attention_gate_projection_residual`
- rule: Flatten attention output, gate it with the fused Q-side branch, project to Hidden width, and add the pre-attention residual.
- node ranges: `16, 126-132, 135`
- target set: `_param_constant4, aten.add.Tensor, aten.copy_.default, aten.empty_like.default, aten.mm.default, aten.mul.Tensor, aten.sigmoid.default, aten.t.default, aten.view.default`

```text
#016 empty_like                     call_function  aten.empty_like.default                       shape=[4096, 5120] dtype=torch.bfloat16 users=copy__4
#126 view_13                        call_function  aten.view.default                             shape=[4096, 6144] dtype=torch.bfloat16 users=mul_14
#127 sigmoid                        call_function  aten.sigmoid.default                          shape=[4096, 6144] dtype=torch.bfloat16 users=mul_14
#128 mul_14                         call_function  aten.mul.Tensor                               shape=[4096, 6144] dtype=torch.bfloat16 users=mm_1
#129 _param_constant4               get_attr       _param_constant4                              shape=[5120, 6144] dtype=torch.bfloat16 users=t_1
#130 t_1                            call_function  aten.t.default                                shape=[6144, 5120] dtype=torch.bfloat16 users=mm_1
#131 mm_1                           call_function  aten.mm.default                               shape=[4096, 5120] dtype=torch.bfloat16 users=copy__4
#132 copy__4                        call_function  aten.copy_.default                            shape=[4096, 5120] dtype=torch.bfloat16 users=add_9
#135 add_9                          call_function  aten.add.Tensor                               shape=[4096, 5120] dtype=torch.bfloat16 users=_to_copy_9,output
```

## 11. Post-attention RMSNorm

- process_id: `post_attention_rmsnorm`
- rule: Normalize the post-attention residual over the Hidden axis before the MLP projection.
- node ranges: `133-134, 136-145`
- target set: `_param_constant5, aten._to_copy.default, aten.add.Tensor, aten.detach.default, aten.mean.dim, aten.mul.Tensor, aten.pow.Tensor_Scalar, aten.rsqrt.default`

```text
#133 _param_constant5               get_attr       _param_constant5                              shape=[5120] dtype=torch.bfloat16 users=detach_3
#134 detach_3                       call_function  aten.detach.default                           shape=[5120] dtype=torch.bfloat16 users=_to_copy_10
#136 _to_copy_9                     call_function  aten._to_copy.default                         shape=[4096, 5120] dtype=torch.float32 users=mul_15,pow_4
#137 pow_4                          call_function  aten.pow.Tensor_Scalar                        shape=[4096, 5120] dtype=torch.float32 users=mean_3
#138 mean_3                         call_function  aten.mean.dim                                 shape=[4096, 1] dtype=torch.float32 users=add_10
#139 add_10                         call_function  aten.add.Tensor                               shape=[4096, 1] dtype=torch.float32 users=rsqrt_3
#140 rsqrt_3                        call_function  aten.rsqrt.default                            shape=[4096, 1] dtype=torch.float32 users=mul_15
#141 mul_15                         call_function  aten.mul.Tensor                               shape=[4096, 5120] dtype=torch.float32 users=mul_16
#142 _to_copy_10                    call_function  aten._to_copy.default                         shape=[5120] dtype=torch.float32 users=add_11
#143 add_11                         call_function  aten.add.Tensor                               shape=[5120] dtype=torch.float32 users=mul_16
#144 mul_16                         call_function  aten.mul.Tensor                               shape=[4096, 5120] dtype=torch.float32 users=_to_copy_11
#145 _to_copy_11                    call_function  aten._to_copy.default                         shape=[4096, 5120] dtype=torch.bfloat16 users=mm_2
```

## 12. MLP and layer output tuple

- process_id: `mlp_and_layer_output`
- rule: Run fused gate/up projection, SiLU-and-multiply activation, down projection, and package layer outputs.
- node ranges: `146-154`
- target set: `_C.silu_and_mul.default, _param_constant6, _param_constant7, aten.empty.memory_format, aten.mm.default, aten.t.default, output`

```text
#146 _param_constant6               get_attr       _param_constant6                              shape=[34816, 5120] dtype=torch.bfloat16 users=t_2
#147 t_2                            call_function  aten.t.default                                shape=[5120, 34816] dtype=torch.bfloat16 users=mm_2
#148 mm_2                           call_function  aten.mm.default                               shape=[4096, 34816] dtype=torch.bfloat16 users=silu_and_mul
#149 empty_1                        call_function  aten.empty.memory_format                      shape=[4096, 17408] dtype=torch.bfloat16 users=mm_3,silu_and_mul
#150 silu_and_mul                   call_function  _C.silu_and_mul.default                       shape=- dtype=- users=-
#151 _param_constant7               get_attr       _param_constant7                              shape=[5120, 17408] dtype=torch.bfloat16 users=t_3
#152 t_3                            call_function  aten.t.default                                shape=[17408, 5120] dtype=torch.bfloat16 users=mm_3
#153 mm_3                           call_function  aten.mm.default                               shape=[4096, 5120] dtype=torch.bfloat16 users=output
#154 output                         output         output                                        shape=- dtype=- users=-
```
