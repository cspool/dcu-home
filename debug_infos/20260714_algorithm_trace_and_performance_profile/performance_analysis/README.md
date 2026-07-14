# 当前最佳 H11.5 + H10.8 性能分析

## 范围

本文只整理当前最高综合分版本 **H11.5 + H10.8**。R4/R11/R23 的 torch/hipprof 历史数据、H10-only 增量以及 LA-RD/LA-RP 等后续候选结果均不作为本目录性能结论。

当前最佳有完整端到端 full×3/accuracy 证据和多组针对现有 H11.5/H10.8 路径的定向硬件采集，但没有一套绑定最终 wheel SHA 的全模型 torch profiler/hipprof。因此本文不会借用 R23 的 kernel 占比，也不会给最终版本虚构热点百分比。

## 1. 权威端到端结果

原始摘要：[`final_identity/FINAL_SUMMARY.md`](original/current_best/final_identity/FINAL_SUMMARY.md)

| Run | 4-8K tok/s | 8-16K tok/s | 16-32K tok/s | Score, K=1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 19.589185273966 | 17.025544511643 | 13.003919636950 | 88.490349137758 |
| 2 | 19.587005633034 | 17.127750783436 | 13.003706763767 | 88.578483694186 |
| 3 | 19.584774728543 | 17.126009556001 | 13.004622851957 | 88.576533680101 |

- 三轮均分：`88.5484555040153`。
- 相对 R24 三轮均值的 20/50/30 加权吞吐提升：`+10.2361569769%`。
- 三轮共 `450/450` 请求成功，`failed=0`，三档 SLA 全部通过。
- Accuracy `K=1.0`，三轮逐请求输出长度和全文一致。
- wheel SHA256：`03568ba87ff64fd0a8aade299026d7ee78cbf40d9c1ed5884fb584250b2031f2`。

完整 benchmark 原件已在相邻 current-best 归档中保存，这里只保留最终身份与摘要，避免重复同一批大文件。

## 2. H10.8 strided LLMM1

### Runtime 验证

入口：[`validation_results.json`](original/current_best/20260712_h10_8_runtime_validation/validation_results.json)

H10.8 的 `(4,640)` pair-reduce 配置在 gfx936/BF16、`N=1`、`K=5120`、16-byte alignment 和限定输出宽度下通过全部正/负门禁。相对旧 `(4,320)` 配置的 standalone median：

| M | old t320 | H10.8 t640 | improvement |
| ---: | ---: | ---: | ---: |
| 14,336 | 120.243 us | 110.854 us | 8.469% |
| 16,384 | 137.498 us | 127.030 us | 8.240% |
| 34,816 | 292.534 us | 271.533 us | 7.734% |

所有记录 seed 都与 t320 bitwise equal，重复运行 bitwise exact；8 个非目标条件均被明确拒绝。该数据是 kernel standalone 验证，不是端到端提升的直接分解。

### 编译资源

入口：[`h10_8_gfx936_kernel_metadata.txt`](original/current_best/20260712_h10_8_single_tu_compile/h10_8_gfx936_kernel_metadata.txt)

- max workgroup：640。
- LDS/group segment：5,200 B。
- VGPR：29；SGPR：11。
- private segment/spill：0。
- wavefront：64。

原始 gfx936 code object、fatbin、源码 diff、build log 和 SHA256 均保留在同目录。

## 3. H11.5 GQA6 attention

### 当前 kernel 的 L4096 reference trace

入口：[`trace_h11_v2`](original/current_best/20260714_h11_5_l4096_reference_profile/trace_h11_v2/)

该采集只运行 H11.5 synthetic attention consumer，没有加载模型或使用正式 fixture。形状为 `seq=4096`、Q24/KV4、head size 256、BF16、page 784，V 使用生产 stride `[14336,256,1]`。5 次 warmup + 20 次 timed invocation；rocprof 中目标 kernel 共 25 次：

| metric | H11.5 `kernel_unified_attention_2d_gqa6` |
| --- | ---: |
| profiler kernel median | 4.472958 ms |
| mean | 4.475377 ms |
| min / p95 / max | 4.414078 / 4.520798 / 4.659678 ms |
| GRD / WGR | 396,288 / 256 |
| workgroups | 1,548 |
| dynamic LDS | 32,768 B |
| Arch_VGPR / SGPR | 216 / 96 |
| scratch | 0 |

Code-object metadata 给出 raw VGPR 216、raw SGPR 87、0 spill、0 private segment。这里没有 achieved/static occupancy 字段，不能从 VGPR/LDS 自行推导 occupancy。

### LDS/MMOP counter

原始数据：[`h11_5_lds.csv`](original/current_best/20260712_h11_5_attention_lds/h11_5_lds.csv)

在 `q=512`、`seq=12000`、Q24/KV4/D256、BF16、page 784 的当前 H11.5 路径中：

| counter | value |
| --- | ---: |
| `SQ_INSTS_LDS` | 54,989,376 |
| `SQ_ACTIVE_INST_LDS` | 69,203,706 |
| `SQ_WAIT_INST_LDS` | 122,376,261 |
| `SQ_LDS_BANK_CONFLICT` | 93,189,120 |
| `SQ_LDS_ADDR_CONFLICT` | 323,232 |
| `SQ_INSTS_MMOP` | 20,686,848 |
| `SQ_ACTIVE_INST_MMOP` | 41,373,696 |

这证明当前 H11.5 同时使用矩阵单元且仍存在显著 LDS wait/conflict；counter 没有兼容的 elapsed-cycle denominator，不能把这些原始数值直接换算为利用率百分比。

## 4. Attention / GDN 矩阵单元归因

入口：[`20260712_attn_gdn_tensor_unit_probe/README.md`](original/current_best/20260712_attn_gdn_tensor_unit_probe/README.md)

当前路径的动态 MMOP 与 code-object 交叉验证结论：

| 路径 | 主要 kernel | gfx936 MMAC/MMOP |
| --- | --- | --- |
| Full Attention Prefill | H11.5 `kernel_unified_attention_2d_gqa6` | 使用 |
| Full Attention Decode stage | `kernel_unified_attention_3d` | 使用 |
| Decode reduction | `reduce_segments` | 不使用 |
| GDN Prefill 主矩阵链 | KKT / solve / recompute / delta-h / output | 使用 |
| GDN Prefill 辅助算子 | cumsum / L2 norm / gating / causal-conv | 不使用 |
| GDN Decode core | packed recurrent delta-rule kernel | 不使用 |

H11.5 prefill 的动态记录为 `SQ_INSTS_MMOP=20,686,848`；GDN prefill 的五类主 kernel 所有被捕获行均为 MMOP-positive。原始 CSV、probe 和 smoke logs 均在该目录。该分析回答“是否使用矩阵单元”，不回答 occupancy。

## 5. 当前 Prefill GEMM 路径

### 代表性 runtime 证据

入口：[`20260712_prefill_tensor_unit_probe/README.md`](original/current_best/20260712_prefill_tensor_unit_probe/README.md)

代表性 `x[512,5120] @ weight[34816,5120].T` BF16 projection 通过 hipBLAS/rocBLAS 到达带 `MAC_MMAC` 的 gfx936 Tensile kernel，反汇编含 `v_mmac_f32_16x16x16_bf16`，动态 `SQ_INSTS_MMOP` 非零。一次 sanity timing 约 `0.715359 ms / 255.17 TFLOP/s`；它只证明矩阵单元被使用，不代表全模型 achieved utilization。

### 生产 trace 归因

原始紧凑证据：[`20260712_r27_prefill_linear_gemm_current_path`](original/current_best/20260712_r27_prefill_linear_gemm_current_path/)

phase-labelled 当前路径证据把 `aten::mm` 连接到 `ISA936/MAC_MMAC` Tensile launch：

- 26 个 `M=4096` prefill chunks 覆盖 all3 输入 token 的 86.75%，是 dense-linear FLOP 主体。
- 六类主 projection 的 per-token FLOP 权重约为 MLP 70.3%、GDN 22.8%、full attention 6.9%。
- 主 prefill linears 已经在用 DCU BF16 矩阵单元，剩余方向是 per-shape rocBLAS/TunableOp solution selection，而不是“开启 MMAC”。

后续 solution audit 的独立确认没有达到预注册晋级门，因此当前最佳仍保留既有自动选择；本归档只收 current-path trace/脚本和 MMOP 证据，没有收候选 solution 的大规模 discovery/confirmation 数据。

## 6. 当前可以下的性能结论

- 端到端：H11.5 + H10.8 是已完整验证的最高综合分版本，三档提升随上下文增加，16-32K 相对 R24 的平均提升最大（`+13.724%`）。
- Decode linear：H10.8 standalone kernel 对三个生产输出宽度比旧配置快约 `7.7%–8.5%`，资源占用低且无 spill。
- Full attention：H11.5 GQA6 确实使用 MMOP；L4096 kernel 约 4.47 ms，当前资源为 216 VGPR/32 KiB dynamic LDS，并存在明确的 LDS wait/conflict 优化空间。
- GDN/Prefill GEMM：主 prefill 链广泛使用矩阵单元；不能再把“未启用 tensor unit”作为首要假设。
- 证据缺口：没有绑定最终 wheel 的全模型 hipprof/torch trace，因而不能给出 H11.5、H10.8、GDN、GEMM 在完整请求中的百分比拆分。

若要补齐最后一项，应在当前 wheel SHA 下重新采集三档 DCU-SMI，以及至少 16-32K n=4 的全模型 hipprof，并记录 wheel、git、输入和采集参数 SHA。现有历史 R23 profile 不应替代这一步。
