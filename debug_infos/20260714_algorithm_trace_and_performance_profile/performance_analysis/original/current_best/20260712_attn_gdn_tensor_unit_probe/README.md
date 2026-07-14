# Attention / GDN 是否使用 DCU 矩阵单元

实验日期：2026-07-13（远程容器时区，目录沿用任务开始日 `20260712`）

## 结论

不能用一个笼统的 Yes/No 概括整个 Attention 或 GDN：

| 路径 | 当前主要 kernel | DCU MMAC/MMOP 矩阵单元 |
| --- | --- | --- |
| Full Attention Prefill core | `kernel_unified_attention_2d_gqa6` | **使用** |
| Full Attention Decode stage | `kernel_unified_attention_3d` | **使用** |
| Full Attention Decode reduction | `reduce_segments` | **不使用** |
| GDN Prefill 主矩阵链 | KKT / solve / recompute / delta-h / output | **全部使用** |
| GDN Prefill 辅助算子 | cumsum / L2 norm / gating / causal-conv | **不使用** |
| GDN Decode core | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | **不使用** |
| Attention/GDN 周围的 Linear 投影 | `F.linear -> hipBLAS/rocBLAS` | **使用**，见前一轮 Prefill GEMM 证据 |

这里的“Tensor 单元”准确指 gfx936 的 MMAC/MMOP 矩阵计算单元，不是
NVIDIA Tensor Core。实际指令主要为
`v_mmac_f32_16x16x16_bf16`；`solve_tril64` 的 FP32 块间合并使用
`v_mmac_16x16x8_f32`。

Attention/GDN core 本身不经过 rocBLAS/hipBLAS；其中的 Triton `tl.dot`
直接编译成 MMAC。只有 QKV/O、qkvz/out 等外围 Linear 投影经过
hipBLAS/rocBLAS。

## 动态硬件计数

所有动态实验使用同一 gfx936、同一组 rocprof 原始 counter：

```text
SQ_INSTS_MMOP
SQ_ACTIVE_INST_MMOP
SQ_INSTS_VALU
SQ_ACTIVE_INST_VALU
SQ_WAVES
```

### Attention Prefill

Shape：单序列，`q=512`、`seq=12000`、24 Q heads、4 KV heads、head size
256、BF16，调用当前 H11.5 GQA6 specialization。

| kernel | MMOP instructions | MMOP active | VALU instructions | waves |
| --- | ---: | ---: | ---: | ---: |
| `kernel_unified_attention_2d_gqa6` | `20,686,848` | `41,373,696` | `112,672,416` | `816` |

原始数据：`attention_prefill_mmop.csv`。

### Attention Decode

Shape：单序列，`q=1`、`seq=12000`、24 Q heads、4 KV heads、head size
256、BF16。生产 AITER 路径为 3D segmented attention 加 reduction。

| kernel | MMOP instructions | MMOP active | VALU instructions | waves |
| --- | ---: | ---: | ---: | ---: |
| `kernel_unified_attention_3d` | `384,000` | `768,000` | `3,183,520` | `256` |
| `reduce_segments` | `0` | `0` | `22,752` | `96` |

这直接表明 Decode 是“MMAC QK/PV stage + 普通 VALU reduction”。原始数据：
`attention_decode_mmop.csv`。

### GDN Prefill

Shape：`B=1,T=512,Hg=16,H=48,K=V=128,BT=64`，复用已有 GDN
compiler-config probe 的真实 case builder，并逐个调用当前 production wrapper。
新 profiler 进程触发了 Triton autotune，因此 raw CSV 有 16,860 行；下面报告
每个实际 kernel dispatch 的 MMOP 范围。所有捕获行均为正值。

| kernel | rows | MMOP-positive rows | `SQ_INSTS_MMOP` 范围 |
| --- | ---: | ---: | ---: |
| `chunk_scaled_dot_kkt_fwd_kernel` | `8,431` | `8,431` | `36,864–49,152` |
| `merge_16x16_to_64x64_inverse_kernel` | `1,772` | `1,772` | `24,576–98,304` |
| `recompute_w_u_fwd_kernel` | `1,287` | `1,287` | `98,304` |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | `1,012` | `1,012` | `196,608` |
| `chunk_fwd_kernel_o` | `4,358` | `4,358` | `221,184–344,064` |

原始数据：`gdn_prefill_mmop.csv`。这些计数证明“使用”，不能直接当作最终
production 利用率；后者需要固定 phase、固定选中 config 的无 autotune 采集。

辅助 cumsum/L2Norm 的额外 counter pass 因 one-shot driver 没有调用这两条
wrapper 而捕获 `0 contexts`，因此 `gdn_prefill_aux_mmop.csv` **不作为动态
No 证据**。No 结论来自实际 production code object 全量反汇编：

- `chunk_local_cumsum_scalar_kernel`：4/4 code objects，MMAC=0；
- `l2norm_fwd_kernel2`：MMAC=0；
- `fused_gdn_gating_kernel`：6/6 code objects，MMAC=0；
- `_causal_conv1d_fwd_kernel`：29/29 code objects，MMAC=0。

### GDN Decode

Shape：`B=1,H=16,HV=48,K=V=128`，调用 production packed decode core。

| kernel | MMOP instructions | MMOP active | VALU instructions | waves |
| --- | ---: | ---: | ---: | ---: |
| `fused_recurrent_gated_delta_rule_packed_decode_kernel` | `0` | `0` | `258,048` | `192` |

`WAVES/VALU > 0` 同时 `MMOP = 0`，因此这是一条有效负证据，不是 profiler
没有捕获到 kernel。原始数据：`gdn_decode_mmop.csv`。

## ISA 与生产路径交叉验证

- 当前 Attention Prefill code object 的 QK 与 PV 两段均含
  `v_mmac_f32_16x16x16_bf16`；当前 reference object 有 128 条静态 MMAC。
- `kernel_unified_attention_3d` 的 QK/PV 同样含 BF16 MMAC；
  `reduce_segments` 的 matrix opcode 为 0。
- GDN Prefill production/cache 全量审计：KKT 28/28、solve64 13/13、
  recompute 10/10、delta-h 24/24、output 37/37 code objects 均含 MMAC。
- GDN packed Decode 的 11 个 unique HSACO 均为 matrix opcode 0，普通
  vector FMA 非零；源码也只有 `tl.sum`、elementwise update 和 outer product，
  没有 `tl.dot`。
- phase-labelled production trace 的调用次数与 16 个 full-attention 层、
  48 个 GDN 层严格对应，排除了只反汇编了未命中 cache 变体的风险。
- 历史 trace 中 27 次 `flash_fwd...dim96` 来自 ViT encoder 启动 profiling，
  不是文本 full-attention，未混入本结论。

## 实验边界

- 本轮只回答 kernel 是否使用矩阵单元，不把 counter 值解释为 occupancy 百分比。
- 没有修改模型、权重、scheduler、固定测试脚本或 vLLM 源码。
- 没有启动 vLLM 服务；实验后 GPU 与显存恢复空闲。
