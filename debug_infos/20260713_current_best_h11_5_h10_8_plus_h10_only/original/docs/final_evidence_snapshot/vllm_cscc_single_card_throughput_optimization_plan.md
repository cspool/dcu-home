# vllm_cscc 单卡单请求吞吐优化计划与阶段性结论

本文档是优化工作的入口文档，只保留优化路线、当前阶段性结论和后续执行计划。完整实验数据与执行约束拆分到独立文档，通过索引项引用。

配套文档：

- [阶段性实验完整结果](./vllm_cscc_stage_experiment_results.md)
- [优化执行和测试约束](./vllm_cscc_optimization_constraints.md)

## 阶段性结论

### R25 已结束目标（5 小时终止条件）

R24 是 R25 的冻结直接比较基线。R25 的 H11.5 + H10.8 已完成
full x3、SLA 和固定 accuracy 计分闭环；精度系数 `K=1.0`，三轮综合分
均值 `88.5484555040153`，成为当前已闭环的可计分最佳。R25 未达到综合分
`90`，相对 R24 三次 full 均值的 20/50/30 加权吞吐提升
`+10.2361569769%` 也未超过 `20%`。但从 goal start epoch
`1783824849` 到 `1783842849` 已恰满 `18000 s`；终止复验在 epoch
`1783842892` 再次确认累计 `18043 s`。因此第三个 5 小时终止条件已满足，
R25 按时间条件结束；该结论不得误写为达到 90 分或相对 R24 +20%。

当前 R25 最终计分栈已经稳定为 R24 + H11.5 + H10.8。
H10.10 已完成 production 实现和 600 秒小样本，但端到端 TPOT
退化，判定 reject；候选服务在最终 health=200 后停止，源码和运行
wheel 均已回退到 H11.5 + H10.8。该栈的三次固定 full `all`、
TTFT/TPOT SLA 和固定 accuracy 均已完成。最终服务在停止前
models/health 均为 200，已清洁停止且无残留服务；final evidence 与
5 小时终止审计均已完成，冻结结果如下：

- H11.5 对 `cache_block_size=784`、单序列、长 prefill 使用逻辑 56-token
  K/V tile、64-token MFMA padding、`BLOCK_Q=32` 和 causal query 上界；其他
  shape 回退 H11.4。独立 wheel SHA256 为
  `7f26c14ca059826cdeb3bc293411fc662523e88810f23bb834882d87de31e892`。
- H11.5 固定 `all 3` 小样本在 `608 s` 外层窗口内 9/9 成功、最终 health
  通过；相对 R24 小样本三档 TTFT 改善 `15.89% / 20.47% / 24.65%`，TPOT
  变化 `+0.27% / +0.18% / +0.05%`。三档输出 token 数发生漂移，因此 raw
  output throughput 只记录、不用于晋级判断，正确性等待 full 与 accuracy。
- H10.8 保持 H10.7 的三个大投影 exact shape gate，配置改为
  `LLMM1Strided(4,640)` 的 wave-pair reduction。组合 wheel SHA256 为
  `03568ba87ff64fd0a8aade299026d7ee78cbf40d9c1ed5884fb584250b2031f2`；
  三 shape 各 10 seeds 均与 t320 位级一致，正式交错微基准提升
  `8.47% / 8.24% / 7.73%`，无 spill。
- H11.5 + H10.8 固定 `all 3` 小样本在 `602 s` 外层窗口内 9/9 成功、
  最终 health 通过。相对 H11.5，TPOT 下降 `5.23% / 5.03% / 4.91%`，
  TTFT 基本不变，三档 generated text 和 output length 逐项完全一致；因此
  H10.8 没有引入新的输出漂移。当时该结果仅用于晋级 full，不单独计分。
- H11.5 + H10.8 的 full run1 在 epoch `1783833368–1783835766`
  完成，窗口 `2398 s`、status=0、最终 health=200，三档均
  50/50 成功。三档 output throughput 为
  `19.589185 / 17.025545 / 13.003920 tok/s`；假设 `K=1` 的单轮
  pre-accuracy 公式分为 `88.490349`，相对 R24 三轮均值的 20/50/30
  加权相对提升为 `+10.021654%`，均未达 `90` 或 `+20%`。相对
  R24 固定输出，length/text 相同分别仅 `110/150` 和 `97/150`，
  总输出 token `37345 -> 36867`；因此该分数仍是受输出漂移影响的
  provisional raw 结果，不能替代三轮统计和 accuracy。
- full run2 在 epoch `1783835836–1783838224` 完成，窗口
  `2388 s`、status=0、最终 health=200，三档 50/50 成功；吞吐为
  `19.587006 / 17.127751 / 13.003707 tok/s`，pre-K 分
  `88.578484`，相对 R24 加权 `+10.346212%`。run1/run2 三档
  output length 和 generated text 逐请求完全一致；当时的两轮均值为
  pre-K `88.534416`、相对 R24 `+10.183933%`，仍未达 `90/+20%`。
  该跨轮一致性不消除相对 R24 的输出漂移 caveat，也不替代 accuracy。
- full run3 在 epoch `1783838265–1783840653` 完成，窗口
  `2388 s`、status=0、最终 health=200，三档 50/50 成功；吞吐为
  `19.584775 / 17.126010 / 13.004623 tok/s`，pre-K 分
  `88.576534`，相对 R24 加权 `+10.340605%`。三轮平均 pre-K 分
  `88.548456`（95% CI `[88.423426, 88.673485]`），相对 R24 平均
  `+10.236157%`（95% CI `[9.774638%, 10.697676%]`），均未达 `90/+20%`。
  三轮的全局请求 TPOT P99 为
  `47.194475 / 47.184350 / 47.189230 ms`，TTFT/TPOT SLA 全部通过。
  三轮输出签名完全一致，但相对 R24 仍只有 text `97/150`、
  length `110/150` 相同，总 token 少 `478`（`-1.27996%`）。
- 固定 `run_accuracy.sh all` 在 epoch `1783840674–1783841621` 完成，
  窗口 `947 s`、status=0、结束后 models/health=200，output directory 为
  `output/local_accuracy_qwen35/20260712_151818`。固定脚本 Final Results 为
  `77.96 / 33.05 / 100 / 100`，official baseline 为
  `77.96 / 32.96 / 100 / 100`，四项 `k_i=1`，最终 `K=1.0`。
  因此三轮最终综合分与 pre-K 分相同，依次为
  `88.490349137758 / 88.578483694186 / 88.576533680101`，均值
  `88.548455504015`，仍未达 `90` 或相对 R24 `+20%`。
- GDN MFMA 配置探测覆盖 `T=16/32/64/512/4096` 和 `2/4/8` warps。
  在最重要的 `T=4096` 点，缺失数值有效的
  `chunk_delta_h_stateful`；探测 JSON 中 `21.4894%` 时间降幅和
  `1.2737x` 加速是其余四个可用 kernel 重新归一化的上界，不是五
  kernel 的完整加权结果。若保守地把缺失项视为 `1x`，五 kernel
  降幅/加速为 `16.2701% / 1.1943x`，同样低于约 `40%` 的端到端
  `3%` 筛选线；
  `chunk_fwd_o` 单 kernel 虽达到 `36.0648%`，其余有效 kernel 仅
  `5.42%–7.81%`，且 stateful/no-state delta 在该点没有通过数值门槛的
  候选。因此未改生产源码。探测 JSON SHA256 为
  `48b0dd6aca769e10be1aecb8b899205846a102ecd712924f81c439f5a204f7ec`。
- H10.9 曾构建 wheel
  `09501b99beb15c4a3481e398b811b54c25213282293e8fc5240bdae6a64d6ed4`，
  试图强制 PyTorch hipBLAS/rocBLAS；服务 marker 记录调用前 backend 已是
  `_BlasBackend.Cublas`，故该补丁是运行时 no-op。它在进入吞吐测试前即
  停止、回滚并重装 H10.8 wheel，不计作提前结束的有效小样本。
- H11.6 standalone 矩阵在 `q=512, seq=12000` 上确认 H11.5 当前配置为
  `3.9024 ms`、`216 VGPR`、零 spill；所有可执行的 heads2/BQ/warps 变体
  均慢 `1.46x–6.96x`。heads3/6 因 `BLOCK_M=96` 非 2 的幂无法编译，逻辑
  tile 48/64 又不整除 cache block 784。因此没有生产补丁，继续保留 H11.5。
- H10.10 为 gfx936 BF16、`n=1`、无 bias、连续张量、
  `weight[5120,6144]` 增加 `LLMM1Strided(2,768)` exact gate。两轮独立
  `31 groups x 50 calls` 交错微基准均由约 `50.94/50.89 us` 降至
  `45.00/44.97 us`，对应 `+13.1987% / +13.1796%`；K17408 候选只有
  `+2.92% / +2.98%`，未进入生产。H10.10 被拒候选 wheel SHA256 为
  `5c76f909b5fed93ec27fcd2de6555f09d4ba1fdaa5429fd07ce63c73e19272a4`，
  构建、安装和 installed validation 均返回 0，repo/installed `utils.py`
  共同 SHA256 为
  `4329939ee47d417f9af2869ea554993b15973501ec76bd2d1e6b1d100090fcef`。
- H11.5 + H10.8 + H10.10 的固定 `all 3` 小样本在 epoch
  `1783832387–1783832987` 完成，外层窗口恰为 `600 s`，三档
  9/9 成功、status=0、最终 health=200。相对 H10.8，三档 TPOT
  全部回归
  `+1.785% / +1.823% / +1.820%`，20/50/30 weighted raw tok/s 为
  `-0.4393%`。16-32K tok/s 虽为 `+2.744%`，但只来自输出长度
  `[23,259,76] -> [23,284,76]`，不能抵消 paired TPOT 退化；8-16K 也有
  `605 -> 601` 的长度/文本漂移。H10.10 因端到端负结果判定 reject。
  `/public/home/tangyu408/testdata/goal_runs/20260712_restore_h10_8_after_h10_10`
  记录回退 status=0，重装 wheel 为
  `03568ba87ff64fd0a8aade299026d7ee78cbf40d9c1ed5884fb584250b2031f2`，当前 repo 和
  installed 文件均为 H11.5 + H10.8。最终服务目录为
  `/public/home/tangyu408/testdata/goal_runs/20260712_h11_5_h10_8_final_serve`；
  full run1/2/3、SLA 汇总和固定 accuracy 均已完成。最终服务在
  epoch `1783841652–1783841659` 停止；停止前 models/health 均为 200，
  停止后 health=000，无残留服务。

### R24 冻结直接比较基线

R24 作为 R25 的上一可计分最佳和冻结直接对照，其栈为
H6.1c + H4.1/H4.2 + D1 + H11.4 + H10.7。该栈完成三次固定 full all：
三轮 450/450 请求成功，综合分分别为
`85.711178 / 85.704307 / 85.706985`，均值 `85.707490`；相对上一最佳
R23 的 20/50/30 加权吞吐提升三轮均超过 20%，均值 `+20.583336%`。
固定 `run_accuracy.sh all` 得到最终精度系数 `K=1.0`，TTFT 和全局
TPOT SLA 均通过。

| 项目 | official baseline | H6.1c | R23 历史最佳 | R24 上一最佳 |
| --- | ---: | ---: | ---: | ---: |
| 吞吐公式分 | 60.0000 | 69.5665 | 79.0289（三轮均值） | 85.7075（三轮均值） |
| 相对 official 的 20/50/30 加权提升 | 0.0000% | +21.8737% | +50.6725% | +82.2576% |
| 相对 H6.1c 的 20/50/30 加权提升 | — | 0.0000% | +23.7651% | +49.1687% |
| 相对 R23 的 20/50/30 加权提升 | — | — | 0.0000% | +20.5833% |
| 全局 TPOT P99 | 71.8034 ms | 70.3850 ms | 最大 55.6700 ms | 最大 49.5467 ms |
| 精度系数 | 1.0 | 1.0 | 1.0 | 1.0 |
| 综合分 | 60.0000 | 69.5665 | 79.0289（三轮均值） | 85.7075（三轮均值） |

H11.4 通过长 prefill 自适应 launch/MFMA 参数继续降低 attention TTFT；
H10.7 以 gfx936 BF16 两路 FP32 累加的 strided LLMM1 替代三个单 token
大投影，进一步降低 decode TPOT。H10.7 使用严格 shape/device/dtype/
alignment gate，非目标路径保持原 GEMM。三次固定 full 的文本和长度逐条
完全一致；相对 R23 仍有输出漂移及少数重复推理循环样本，但固定 accuracy
四项均为 `k_i=1`。本轮没有另做独立 API token-id/finish-reason 重放，
因此不能宣称改变请求序列后仍逐 token 确定。

R25 进入时的增量工作基线（历史记录）：

- R25 进入时的计分基础为 H6.1c + H4.1/H4.2 + D1 + H11.4 + H10.7；
  H11.5/H10.8 当时只作为 candidate overlay。它们随后已完成 full x3、
  SLA 和 accuracy 闭环并成为 R25 最终最佳；R24 仅继续作为冻结直接对照与
  回退基线。
- 冻结 wheel SHA256：`399c7a847c8607269b41d77f189e96505882094286f6c31a4beedd39194a4fbc`。H10.7 只对 gfx936、BF16、单 token、`k=5120`、无 bias、`m in {96,14336,16384,34816}` 使用 exact shape gate；`m=96` 走原 LLMM1 `rows=4`，三个大投影走 `LLMM1Strided(4,320)`。H11.4 对 `max_seqlen_q >= 128` 使用 `num_warps=2, num_stages=1, waves_per_eu=1, matrix_instr_nonkdim=16, kpack=2`，短 prefill 保留 H11.3 配置。
- 冻结 evidence：R24 build 目录中的 `source_runtime_manifest.sha256` 与 `final_evidence.sha256` 均已完整校验；H11.4 新文件仍为 untracked，迁移时必须显式携带。H10.6 是被 H10.7 取代的中间候选，不属于源码失败；首次 503 仅是代理访问无效测量。
- 可保留的工程基础：`H4.1/H4.2` 的 GDN prefill metadata/no-initial-state 快路径，以及 `D1` 的 GDN decode packed validate 跳过。它们未形成可单独计分的显著收益，但小样本未观察到降速或功能错误，可作为后续源码栈的一部分继续叠加。
- 必须排除：`D2` AITER decode 强制 2D 分支、`H4.3` L2 norm 变体、`H10.5` gfx936 `wvSplitK` 兼容变体。H10.5 首次 BF16 数值对照最大绝对误差 `4.4443`，已回退并重建纯 H11.3 wheel。`H4.4` `wy_fast` tile 变体虽小样本为正，但当前源码已回退；若纳入增量基础，需要单独重放并完成同口径确认。
- 本轮已否决：`H8.1` padded block table fill cache、`H12.1` GDN CUDA graph padding fill cache、`H10.1` 直接启用 ROCm skinny GEMM、`H11.1` AITER descale expand view cache、`H4.5` Qwen3.5 GDN core output `empty` allocation、`H10.2` unquantized GEMM dispatch cache、`H11.2` AITER KV cache view cache。它们均未形成可计入收益，不作为后续基础。

历史通过多请求并发、修改测试脚本、修改数据集、serve 参数扫描、chunk budget 扫描、prefix cache 或未重新编译 wheel 得到的结果，全部不进入结论。

阶段性结论只引用后两个文档中的索引，不在主计划重复展开完整实验日志或约束细节。

| 结论 ID | 结论 | 依据索引 | 后续动作 |
| --- | --- | --- | --- |
| S0 | official baseline 已完成吞吐和精度闭环，可作为固定比较基线。 | [R0](./vllm_cscc_stage_experiment_results.md#r0-official-baseline-full-all-基准), [C0](./vllm_cscc_optimization_constraints.md#c0-赛题口径), [C1](./vllm_cscc_optimization_constraints.md#c1-固定实验契约), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议) | 后续候选统一对齐 R0 的 full `all` 口径。 |
| S1 | H6.1c 是历史可计分基础，也是 R23 的固定增量对照。 | [R1](./vllm_cscc_stage_experiment_results.md#r1-h61c-rocm-aiter-unified-attention), [R2](./vllm_cscc_stage_experiment_results.md#r2-h61c-accuracy-对照), [C4](./vllm_cscc_optimization_constraints.md#c4-正确性门槛) | 作为历史计分基础和增量归因对照保留。 |
| S2 | H5.1 的 upstream ROCm FlashAttention 强制优先改变了输出行为，不能计入收益。 | [R3](./vllm_cscc_stage_experiment_results.md#r3-h51-上游-rocm-flashattention-强制优先), [C4](./vllm_cscc_optimization_constraints.md#c4-正确性门槛) | 只有先修复输出、停止原因和精度后才能重开。 |
| S3 | 早期单条 profile 只能指导定位，不能作为验收结论。 | [R4](./vllm_cscc_stage_experiment_results.md#r4-早期单条-profile), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议), [C5](./vllm_cscc_optimization_constraints.md#c5-profiler-规范) | profiler 结果必须回到固定 full `all` 和 accuracy 闭环验证。 |
| S4 | profiling 后的主优化面是 GEMM/linear、AITER attention2d prefill、Fill/elementwise 和 GDN prefill；decode 影响 TPOT，但不是当前唯一首要热点。 | [R11](./vllm_cscc_stage_experiment_results.md#r11-固定配置-dcuhipprof-profiling), [C2](./vllm_cscc_optimization_constraints.md#c2-允许与禁止), [C5](./vllm_cscc_optimization_constraints.md#c5-profiler-规范), [C6](./vllm_cscc_optimization_constraints.md#c6-源码边界) | 下一轮先做 GEMM/attention2d 的 shape/source 归因，再选一个最小源码候选实施。 |
| S5 | H4.1/H4.2 可作为增量工作基线保留，但不单独计入有效性能突破；H4.3/H4.4 不默认保留。 | [R5](./vllm_cscc_stage_experiment_results.md#r5-h41-gdn-prefill-state-初始化快路径), [R6](./vllm_cscc_stage_experiment_results.md#r6-h42-gdn-prefill-no-initial-state-specialization), [R7](./vllm_cscc_stage_experiment_results.md#r7-h43-fla-l2-norm-kernel-变体), [R8](./vllm_cscc_stage_experiment_results.md#r8-h44-gdn-prefill-wy_fast-tile-变体) | 后续在当前优化栈上叠加；若重开 GDN prefill，必须先有 profiler 归因。 |
| S6 | D1 可作为增量工作基线保留；D2 明确降速并已回滚。 | [R9](./vllm_cscc_stage_experiment_results.md#r9-d1-gdn-decode-packed-validate-跳过), [R10](./vllm_cscc_stage_experiment_results.md#r10-d2-aiter-unified-attention-decode-2d-分支强制) | decode 专项保留，但只有 TPOT 专项 profiling 证明其占主导时才上升为首要候选。 |
| S7 | 后续实验采用“累计优化栈 + 单候选增量归因”的双对照。 | [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议), [C7](./vllm_cscc_optimization_constraints.md#c7-evidence-card-模板) | 每轮同时报告相对 official baseline、H6.1c 和上一轮工作基线的变化。 |
| S8 | 固定配置 profiling 已完成；PCIe 不是瓶颈，显存容量接近打满，主要 kernel 时间集中在 GEMM/linear 与 AITER attention2d。 | [R11](./vllm_cscc_stage_experiment_results.md#r11-固定配置-dcuhipprof-profiling), [C5](./vllm_cscc_optimization_constraints.md#c5-profiler-规范) | 下一轮优先定位 top GEMM 的源 shape/backend 和 attention2d wrapper/kernel。 |
| S9 | H8.1 padded block table fill cache 已实现并验证，但收益为噪声级，已回退。 | [R12](./vllm_cscc_stage_experiment_results.md#r12-h81-padded-block-table-fill-cache), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议), [C4](./vllm_cscc_optimization_constraints.md#c4-正确性门槛) | 不继续做同类 Python cache 补丁；若重开 FillFunctor，先做 stack/source attribution。 |
| S10 | H12.1 GDN CUDA graph padding fill cache 已实现并验证，吞吐 `+0.055%`，属于噪声级。 | [R13](./vllm_cscc_stage_experiment_results.md#r13-h121-gdn-cuda-graph-padding-fill-cache), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议), [C4](./vllm_cscc_optimization_constraints.md#c4-正确性门槛) | 不继续做 metadata padding cache 补丁；下一步转向 GEMM/linear 和 AITER attention2d。 |
| S11 | H10.1 直接启用 ROCm skinny GEMM 不可行，因为当前 wheel 没有注册 `wvSplitK` 扩展。 | [R14](./vllm_cscc_stage_experiment_results.md#r14-h101-rocm-skinny-gemm-可用性探测), [C6](./vllm_cscc_optimization_constraints.md#c6-源码边界) | H10 继续前必须做真实 GEMM shape/source/backend 归因，不能只解除 env gate。 |
| S12 | H11.1 AITER descale expand view cache 已实现并验证，吞吐 `+0.015%`，属于噪声级。 | [R15](./vllm_cscc_stage_experiment_results.md#r15-h111-aiter-descale-expand-view-cache), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议) | AITER attention2d 后续转向 kernel 参数、shape 和 shared-memory bank conflict，不继续 wrapper 小对象缓存。 |
| S13 | H4.5 Qwen3.5 GDN core output 改为 `torch.empty` 会改变输出 token 数，正确性失败。 | [R16](./vllm_cscc_stage_experiment_results.md#r16-h45-qwen35-gdn-core-output-empty-allocation), [C4](./vllm_cscc_optimization_constraints.md#c4-正确性门槛) | 保留 `torch.zeros` 语义；后续不能通过未初始化 output buffer 消除 GDN fill。 |
| S14 | H10.2 unquantized GEMM dispatch cache 已实现并验证，吞吐 `+0.033%`，属于噪声级。 | [R17](./vllm_cscc_stage_experiment_results.md#r17-h102-unquantized-gemm-dispatch-cache), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议) | Python dispatch 不是可见瓶颈；H10 后续必须回到 GEMM shape/source/backend 归因。 |
| S15 | H10.3 已完成 Linear/GEMM source attribution：主目标是 MLP gate/up、MLP down、GDN `qkvz/out` 和 attention `qkv/o`，不是 `in_proj_ba` 小 GEMM；当前 AITER Triton GEMM 与 ROCm skinny GEMM 均不可直接接入。 | [R18](./vllm_cscc_stage_experiment_results.md#r18-h103-lineargemm-source-attribution), [C5](./vllm_cscc_optimization_constraints.md#c5-profiler-规范), [C6](./vllm_cscc_optimization_constraints.md#c6-源码边界) | H10 下一步做这些大投影的真实 backend/kernel 归因，避免继续做 wrapper/dispatch 补丁。 |
| S16 | H11.2 AITER KV cache view cache 已实现并验证，吞吐 `-0.055%`，无收益并已回退。 | [R19](./vllm_cscc_stage_experiment_results.md#r19-h112-aiter-kv-cache-view-cache), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议) | H11 后续不再做 wrapper view/cache 小补丁，转向 attention2d kernel 参数、shape、block table 和 bank conflict。 |
| S17 | H10.4 恢复 `_rocm_C` 并只 gate 实测获胜的 gfx936 LLMM1 shapes；小样本相对 H6.1c 加权 `+13.52%`，三档 TPOT 降至约 `53.47–55.26 ms`。 | [R20](./vllm_cscc_stage_experiment_results.md#r20-h104-gfx936-llmm1-exact-shape-gate), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议), [C4](./vllm_cscc_optimization_constraints.md#c4-正确性门槛) | 作为晋级工作栈的 decode/GEMV 基础保留；其输出变化必须由 accuracy 闭环审计。 |
| S18 | H11.3 消除了 AITER GQA6 prefill attention2d 的重叠行；叠加 H10.4 后，小样本三档相对 H6.1c 均超过 `+20%`，加权 `+22.11%`。 | [R21](./vllm_cscc_stage_experiment_results.md#r21-h113-gqa6-prefill-non-overlap-attention2d), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议) | full、accuracy 与重复性闭环已在 R23 完成。 |
| S19 | H10.5 gfx936 `wvSplitK` 兼容补丁构建成功但 BF16 数值错误，已在进入服务测试前回退。 | [R22](./vllm_cscc_stage_experiment_results.md#r22-h105-gfx936-wvsplitk-correctness-probe), [C4](./vllm_cscc_optimization_constraints.md#c4-正确性门槛) | 不启用 `wvSplitK`；保留 H10.4 的 LLMM1 exact gate。 |
| S20 | H10.4 + H11.3 三次 full 相对 H6.1c 加权提升均值 `+23.7651%`、最差轮 `+23.5860%`；450/450 请求成功，accuracy 系数 `1.0`，公式分均值 `79.0289`。 | [R23](./vllm_cscc_stage_experiment_results.md#r23-h104-和-h113-full-all-accuracy-与重复性闭环), [C0](./vllm_cscc_optimization_constraints.md#c0-赛题口径), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议), [C4](./vllm_cscc_optimization_constraints.md#c4-正确性门槛) | 作为上一可计分最佳与 R24 的直接增量对照保留。 |
| S21 | H11.4 + H10.7 三次 full 共 450/450 成功；综合分均值 `85.7075`，相对 R23 加权提升均值 `+20.5833%`，accuracy `K=1.0`，SLA 全部通过。 | [R24](./vllm_cscc_stage_experiment_results.md#r24-h114-adaptive-gqa6-prefill-和-h107-gfx936-strided-llmm1-闭环), [C0](./vllm_cscc_optimization_constraints.md#c0-赛题口径), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议), [C4](./vllm_cscc_optimization_constraints.md#c4-正确性门槛) | 上一目标闭环，R24 晋升为 R25 的当前可计分基线；R25 终止条件重新按 90 分、相对 R24 +20% 或 5 小时计算。 |
| S22 | R25 已排除 GDN MFMA、H10.9 backend no-op、H11.6 attention 配置矩阵和 H10.10 K6144 exact gate；H10.10 虽 standalone 微基准为正且完成 build/install/validation，但三档端到端 TPOT 均回归。 | [R25](./vllm_cscc_stage_experiment_results.md#r25-h115h108-候选筛选与-h1010-provisional), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议), [C4](./vllm_cscc_optimization_constraints.md#c4-正确性门槛) | H10.10 的 600 秒窗口、final health 和回退已完成；最终栈 H11.5 + H10.8 已完成 full x3/SLA/accuracy，`K=1.0`、三轮均分 `88.548456`，仍未达 90/+20%。5 小时条件已于 epoch `1783842849` 满足；R25 按时间条件结束并冻结全部闭环 evidence。 |

## 文档索引

完整实验结果索引：

| 索引 | 引用 | 用途 |
| --- | --- | --- |
| R0 | [Official baseline full all 基准](./vllm_cscc_stage_experiment_results.md#r0-official-baseline-full-all-基准) | baseline 吞吐、accuracy 与综合基准 |
| R1 | [H6.1c ROCm AITER Unified Attention](./vllm_cscc_stage_experiment_results.md#r1-h61c-rocm-aiter-unified-attention) | 历史可计分基础的构建、吞吐与 SLA |
| R2 | [H6.1c accuracy 对照](./vllm_cscc_stage_experiment_results.md#r2-h61c-accuracy-对照) | 精度系数和正确性阶段结论 |
| R3 | [H5.1 上游 ROCm FlashAttention 强制优先](./vllm_cscc_stage_experiment_results.md#r3-h51-上游-rocm-flashattention-强制优先) | 已排除的无效候选 |
| R4 | [早期单条 profile](./vllm_cscc_stage_experiment_results.md#r4-早期单条-profile) | 仅作定位起点的历史 profile |
| R5 | [H4.1 GDN prefill state 初始化快路径](./vllm_cscc_stage_experiment_results.md#r5-h41-gdn-prefill-state-初始化快路径) | GDN prefill 小样本未达标候选 |
| R6 | [H4.2 GDN prefill no-initial-state specialization](./vllm_cscc_stage_experiment_results.md#r6-h42-gdn-prefill-no-initial-state-specialization) | GDN prefill 小样本未达标候选 |
| R7 | [H4.3 FLA L2 norm kernel 变体](./vllm_cscc_stage_experiment_results.md#r7-h43-fla-l2-norm-kernel-变体) | 输出变化且吞吐下降的回滚候选 |
| R8 | [H4.4 GDN prefill `wy_fast` tile 变体](./vllm_cscc_stage_experiment_results.md#r8-h44-gdn-prefill-wy_fast-tile-变体) | GDN prefill 小样本微弱收益、未达标候选 |
| R9 | [D1 GDN decode packed validate 跳过](./vllm_cscc_stage_experiment_results.md#r9-d1-gdn-decode-packed-validate-跳过) | Decode wrapper 检查开销不是主瓶颈 |
| R10 | [D2 AITER unified attention decode 2D 分支强制](./vllm_cscc_stage_experiment_results.md#r10-d2-aiter-unified-attention-decode-2d-分支强制) | AITER single-query 3D 分支优于强制 2D，候选回滚 |
| R11 | [固定配置 DCU/hipprof profiling](./vllm_cscc_stage_experiment_results.md#r11-固定配置-dcuhipprof-profiling) | 当前瓶颈排序和 PMC 证据 |
| R12 | [H8.1 padded block table fill cache](./vllm_cscc_stage_experiment_results.md#r12-h81-padded-block-table-fill-cache) | 已实现但无收益的 Fill/metadata 候选 |
| R13 | [H12.1 GDN CUDA graph padding fill cache](./vllm_cscc_stage_experiment_results.md#r13-h121-gdn-cuda-graph-padding-fill-cache) | 已实现但无收益的 GDN metadata 候选 |
| R14 | [H10.1 ROCm skinny GEMM 可用性探测](./vllm_cscc_stage_experiment_results.md#r14-h101-rocm-skinny-gemm-可用性探测) | 已排除的 GEMM env gate 候选 |
| R15 | [H11.1 AITER descale expand view cache](./vllm_cscc_stage_experiment_results.md#r15-h111-aiter-descale-expand-view-cache) | 已实现但无收益的 AITER wrapper 候选 |
| R16 | [H4.5 Qwen3.5 GDN core output `empty` allocation](./vllm_cscc_stage_experiment_results.md#r16-h45-qwen35-gdn-core-output-empty-allocation) | 输出 token 数变化的正确性失败候选 |
| R17 | [H10.2 unquantized GEMM dispatch cache](./vllm_cscc_stage_experiment_results.md#r17-h102-unquantized-gemm-dispatch-cache) | 已实现但无收益的 linear dispatch 候选 |
| R18 | [H10.3 Linear/GEMM source attribution](./vllm_cscc_stage_experiment_results.md#r18-h103-lineargemm-source-attribution) | GEMM 热点绑定到 Linear 层 shape/source，排除小 GEMM 优先级 |
| R19 | [H11.2 AITER KV cache view cache](./vllm_cscc_stage_experiment_results.md#r19-h112-aiter-kv-cache-view-cache) | 已实现但无收益的 AITER wrapper view cache 候选 |
| R20 | [H10.4 gfx936 LLMM1 exact-shape gate](./vllm_cscc_stage_experiment_results.md#r20-h104-gfx936-llmm1-exact-shape-gate) | 已显著降低 TPOT、进入晋级工作栈的 decode Linear 候选 |
| R21 | [H11.3 GQA6 prefill non-overlap attention2d](./vllm_cscc_stage_experiment_results.md#r21-h113-gqa6-prefill-non-overlap-attention2d) | 小样本相对 H6.1c 加权超过 20% 的晋级候选 |
| R22 | [H10.5 gfx936 wvSplitK correctness probe](./vllm_cscc_stage_experiment_results.md#r22-h105-gfx936-wvsplitk-correctness-probe) | 构建成功但数值失败并回退的 kernel 候选 |
| R23 | [H10.4 和 H11.3 full all accuracy 与重复性闭环](./vllm_cscc_stage_experiment_results.md#r23-h104-和-h113-full-all-accuracy-与重复性闭环) | 上一可计分最佳和 R24 的直接增量基线 |
| R24 | [H11.4 adaptive GQA6 prefill 和 H10.7 gfx936 strided LLMM1 闭环](./vllm_cscc_stage_experiment_results.md#r24-h114-adaptive-gqa6-prefill-和-h107-gfx936-strided-llmm1-闭环) | R25 的上一最佳/冻结直接基线：源码、10min 筛选、三次 full、SLA、accuracy、输出审计和最终计分 |
| R25 | [H11.5/H10.8 候选筛选与 H10.10 provisional](./vllm_cscc_stage_experiment_results.md#r25-h115h108-候选筛选与-h1010-provisional) | H11.5/H10.8 实现与小样本、GDN/H10.9/H11.6/H10.10 reject、H10.10 回退、full x3/SLA/accuracy/输出审计、服务清理、final evidence 与 5 小时终止闭环 |

执行和测试约束索引：

| 索引 | 引用 | 用途 |
| --- | --- | --- |
| C0 | [赛题口径](./vllm_cscc_optimization_constraints.md#c0-赛题口径) | 指标、SLA、评分公式、精度约束 |
| C1 | [固定实验契约](./vllm_cscc_optimization_constraints.md#c1-固定实验契约) | 固定脚本、固定参数、no-proxy 和 accuracy 命令 |
| C2 | [允许与禁止](./vllm_cscc_optimization_constraints.md#c2-允许与禁止) | 可改源码范围与禁止策略 |
| C3 | [测量协议](./vllm_cscc_optimization_constraints.md#c3-测量协议) | wheel 构建、服务启动、吞吐和精度闭环 |
| C4 | [正确性门槛](./vllm_cscc_optimization_constraints.md#c4-正确性门槛) | 输出、finish reason、hash 与 OpenCompass 门槛 |
| C5 | [Profiler 规范](./vllm_cscc_optimization_constraints.md#c5-profiler-规范) | HIP timeline、counter 与瓶颈分类要求 |
| C6 | [源码边界](./vllm_cscc_optimization_constraints.md#c6-源码边界) | 只读锁定文件与可优化源码地图 |
| C7 | [Evidence Card 模板](./vllm_cscc_optimization_constraints.md#c7-evidence-card-模板) | 候选进入结论前必须补齐的证据 |

## 优化计划

### 目标和边界

目标是在单卡、单请求并发、固定测试脚本的条件下提升 Qwen3.5-27B vLLM CSCC/DCU 在线服务的 `output_throughput`。

不可改变的边界：

- 不修改 `run_throughput.sh`、`run_accuracy.sh`、模型权重、tokenizer、chat template 或评测解析口径。
- 不通过多请求并发、serve 参数调优、修改 batch scheduler、prefix cache、speculative decoding、持久化量化或模型结构变化获得收益。
- 允许的主路径是修改 `remote-home/vllm_cscc` 源码，重新编译 wheel，使用固定启动和固定测试脚本验证。

所有候选必须按 C3 完成源码变更、wheel 构建、服务启动、吞吐 `all`、accuracy `all` 和证据记录闭环。

### Workload 与硬件判断

单请求长上下文包含 prefill 与 decode 两段。

Prefill 阶段处理 prompt/context 并生成 KV cache，是 8K-32K 档 TTFT 的主要候选来源。vLLM 的 `chunked prefill` 是同一请求内部的 context 切分，不改变请求数、不改变上下文语义、不拆数据集。chunked prefill 可以作为路径分析对象，但不能扫描或调整 `MAX_NUM_BATCHED_TOKENS`，也不能修改 batch scheduler 代码来改变 chunk 调度策略。

Decode 阶段每次生成一个或少量 token，是 TPOT P99 的核心来源。decode 优化应优先围绕 Attention、Linear、GDN state update、KV 读取、kernel launch 和 HBM 字节流展开。

模型与硬件先验：

- Qwen3.5-27B 使用 `bfloat16`。
- full-attention `head_dim=256`。
- `layer_types` 共 64 层，其中每 4 层 1 个 `full_attention`，约 16 个 full-attention 层和 48 个 `linear_attention`/GDN 层。
- DCU 目标卡按讲义归纳为海光 DCU `gfx936`，`wavefront=64`。
- 微基准给出的可持续量级：HBM 峰值约 `1206 GB/s`，bf16 算力峰值约 `395 TFLOPS`。
- Roofline 拐点约 `327 FLOP/byte`。算术强度低于该拐点时优先按 memory-bound 分析，高于该拐点时优先按 compute-bound 分析。

这些先验只用于决定先测哪里；进入结论必须以 C5 的 profiler 证据和固定脚本结果为准。

### Backend 证据矩阵

| 路径 | 预期源码 | 必需证据 | 边界 | 决策 |
| --- | --- | --- | --- | --- |
| full-attention prefill | `selector.py`、`flash_attn.py`、`rocm_aiter_fa.py`、`triton_prefill_attention.py` | selector 日志、kernel timeline、chunk `q_len` | 不改 batch scheduler 或 chunk 参数 | 证实热点后优化 wrapper/Triton/HIP kernel |
| full-attention decode | `paged_attn.py`、`triton_decode_attention.py`、`csrc/rocm/attention.cu` | paged decode kernel 名称与耗时 | gfx936 custom paged attention 只支持 head 64/128 | 未命中时降低 `attention.cu` 优先级 |
| KV cache allocation/block manager | `kv_cache_manager.py`、`single_type_kv_cache_manager.py`、`block_pool.py`、`kv_cache_utils.py` | block 分配/释放次数、碎片率、HBM 占用、cache miss | 不改锁定参数或 scheduler 策略 | 若开销高，优化内部实现 |
| KV cache write/update | `triton_reshape_and_cache_flash.py`、`cache_kernels.cu`、`cache_kernels_fused.cu` | reshape/cache kernel 次数、字节流、slot mapping shape | 不改变 KV 语义或输出口径 | 若 memory-bound，减少 HBM 往返 |
| Q/K norm + RoPE | `qk_norm_rope_fusion.py`、`fused_qknorm_rope_kernel.cu`、`torch_bindings.cpp` | custom op 命中、kernel 数下降、数值对比 | 不生成可复用模型图或改变模型结构 | 只在 full-attention 路径成立时优化 |
| GDN/linear attention prefill | `gdn_attn.py`、`qwen3_next.py`、`layers/fla/ops/` | GDN kernel timeline、48 层占比、state/update shape | `--gdn-prefill-backend` 只是已有开关，不能单独计收益 | 若占比高，减少中间张量和 launch |
| GDN/linear attention decode | `layers/fla/ops/`、`gdn_attn.py` | per-token state update kernel 与 HBM 读写 | 不改变 recurrent state 语义 | 若 TPOT 热，优化 state update |
| Decode 算子融合 | `gdn_attn.py`、`layers/fla/ops/`、`qwen3_next.py`、`linear.py`、`gpu_model_runner.py` | decode token 级 kernel 序列、launch gap、相邻 elementwise/norm/reshape/copy 证据 | 不改采样、stop、输出 token、batch scheduler 或请求间状态复用 | 只融合数学等价子路径 |
| Linear/GEMM/GEMV | `linear.py`、`kernels/linear/`、`_aiter_ops.py`、`platforms/rocm.py` | rocBLAS/hipBLASLt/AITER/Triton 命中证据 | 不做持久化权重量化或权重重排压缩 | 只修正真实 fallback |
| 非持久化运行时量化 | `quantization/kv_cache.py`、`torch_utils.py`、attention/linear kernels | KV/activation 动态 scale、临时 dtype、精度审计 | 禁止生成量化权重文件、压缩缓存或持久化转换 | 只有精度系数通过时保留 |
| 非 batch-scheduler runtime overhead | `gpu_model_runner.py`、`gpu_input_batch.py`、`workspace.py` | Python timeline、launch gap、metadata 构造次数 | 禁止修改 batch scheduler 相关代码 | 仅缓存/消除执行路径重复 work |

### 假设 Backlog

优先级由当前增量工作基线上的 profiler 结果决定。每个候选一次只验证一个假设，避免把 backend 切换、fusion、调度变更和环境开关混成不可归因数字。新增候选不从 official baseline 重新开始，而是在已有可行优化栈上叠加；归因时同时记录相对上一轮工作基线和相对 H6.1c 的差异。

| ID | 触发条件 | 源码路径 | 预期瓶颈 | 最小改动 | 验证信号 | 失败判据 |
| --- | --- | --- | --- | --- | --- | --- |
| H0 路径表征 | 尚未证明固定脚本实际命中哪些 backend | `selector.py`、`registry.py`、`model_runner.py`、相关 backend | 结论风险来自路径假设错误 | 增加一次性日志或轻量插桩 | backend 矩阵填完整 | 日志影响吞吐或无法关联到 EngineCore |
| H1 Decode 专项：Attention/GDN/算子融合 | H6.1c 后 TPOT 只小幅改善 | `paged_attn.py`、`triton_decode_attention.py`、`gdn_attn.py`、`layers/fla/ops/`、`linear.py`、`qwen3_next.py` | HBM 带宽、GEMV、GDN state update、launch-bound | 数学等价融合、减少读写字节、修正 backend fallback | TPOT P99/Mean TPOT 下降，output throughput 提升 | 输出 token、finish reason、OpenCompass 精度或 SLA 失败 |
| H1.1 full-attention decode | attention decode 占 TPOT 主体 | `paged_attn.py`、`triton_decode_attention.py`、`rocm_aiter_unified_attn.py`、`attention.cu` | KV 读取、block table address、softmax/reduction、非合并访存 | shape/capability gating、减少 wrapper/copy、优化 block table 访问 | attention decode kernel 时间下降 | head_dim=256 路径不命中目标 kernel，或输出异常 |
| H1.2 GDN/linear attention decode | 48 个 GDN/linear 层 state update 累计占比高 | `gdn_attn.py`、`layers/fla/ops/`、`qwen3_next.py` | recurrent state 读写、gate/update 中间张量、reshape/copy、launch-bound | 融合 gate/state/update/norm 等价子路径 | GDN decode kernel 数、HBM bytes、TPOT 下降 | state 语义改变或精度下降 |
| H1.3 decode 算子融合 | 单 token decode 小 kernel 多且 gap 明显 | `qwen3_next.py`、`linear.py`、`gpu_model_runner.py`、`workspace.py`、fusion/custom op | launch-bound、metadata 重建、中间 tensor HBM 往返 | 融合相邻无副作用算子，缓存固定 shape 元数据 | kernel 数和 EngineCore gap 下降 | 触碰 scheduler、改变采样/stop 或引入持久化缓存 |
| H2 KV cache allocation/block manager | 显存碎片、block 管理或 cache 分配开销可观 | `kv_cache_manager.py`、`single_type_kv_cache_manager.py`、`block_pool.py`、`kv_cache_utils.py` | 块管理开销、碎片率、无效 block 访问 | 优化内部块管理和 layout | HBM 占用/碎片下降，TTFT/TPOT 不退 | 改变上下文容量、scheduler 或请求语义 |
| H3 KV cache write/update | cache reshape/write 在 prefill 或 decode 中高占比 | `triton_reshape_and_cache_flash.py`、`cache_kernels.cu`、`cache_kernels_fused.cu` | 非合并访存、重复地址计算、额外 copy | 合并 slot mapping/address 计算，向量化 K/V store | cache kernel 时间和 HBM bytes 下降 | 破坏 KV 正确性或输出哈希 |
| H4 GDN/linear attention prefill | 48 个 GDN/linear 层合计占 TTFT 主体 | `gdn_attn.py`、`qwen3_next.py`、`layers/fla/ops/` | 中间张量 materialization、HBM 往返、小 kernel launch | 融合安全的 gate/state/norm 子路径 | GDN kernel 数和字节下降，TTFT P99 不退 | GDN 占比低，或精度下降 |
| H5 full-attention prefill backend | full-attention chunk kernel 是 TTFT 热点 | `flash_attn.py`、`rocm_aiter_fa.py`、`triton_prefill_attention.py`、`prefix_prefill.py` | IO-bound attention、fallback、wrapper/copy | 对实际命中 path 做 shape 特化或减少 wrapper/copy | TTFT P99 下降，output throughput 提升 | head_dim=256 路径未命中目标 kernel |
| H6 GEMM/GEMV backend gating | profiler 显示 Triton/eager fallback 或 AITER 未命中 | `_aiter_ops.py`、`platforms/rocm.py`、`linear.py`、`kernels/linear/` | backend 选择错误、API 能力检测不足、权重带宽受限 | 基于实际 API 和 dtype/shape 做 capability gating | backend 稳定命中，TPOT 或 TTFT 改善 | 伪装设备能力、fallback 抖动或权重持久化变更 |
| H7 非持久化运行时量化 | KV/activation 带宽是主瓶颈且精度有余量 | `quantization/kv_cache.py`、`torch_utils.py`、attention/linear kernels | KV 或 activation 字节流过大 | 动态 scale、临时低精度、kernel 内部转换 | output throughput 提升，OpenCompass 通过 | 生成持久化量化产物或精度下降不可接受 |
| H8 非 batch-scheduler runtime overhead | HIP timeline 存在 Python gap、launch gap 或 metadata 重建 | `gpu_model_runner.py`、`gpu_input_batch.py`、`workspace.py` | Python overhead、metadata 重建、workspace resize | 缓存固定 shape 元数据，减少重复对象构造 | EngineCore gap 下降，TTFT/E2E 改善 | 修改 scheduler 代码或锁定参数 |
| H9 Q/K norm + RoPE fusion | full-attention 层中 norm/rope 小 kernel 多 | `qk_norm_rope_fusion.py`、`fused_qknorm_rope_kernel.cu`、`torch_bindings.cpp` | launch-bound 与中间张量写回 | 源码级 custom op/fusion | kernel 数下降，数值/精度通过 | 被判定为模型图重构或数值不稳定 |
| H10 GEMM/linear source attribution | R11 显示 `linear_gemm` 占 `53.221%`，top GEMM PMC 低 L2 hit/低有效 GFLOPS | `linear.py`、`kernels/linear/`、`_aiter_ops.py`、`qwen3_next.py`、ROCm backend gating | 小/瘦 GEMM、GEMV、权重流式访存、backend 选择或融合不足 | 给 top rocBLAS kernel 加 shape/source 归因，再做单点 gating/fusion 候选 | top GEMM 时间下降，TTFT/TPOT 至少一项改善 | 无法绑定到源码调用点，或候选改变权重/模型结构 |
| H11 AITER attention2d prefill | R11 显示 `kernel_unified_attention_2d` 占 `26.707%`，PMC 共享内存 bank conflict `28.134%` | `rocm_aiter_unified_attn.py`、AITER wrapper、attention metadata | attention2d tiling/shared-memory、wrapper tensor 构造、descale/copy | 保持 H6.1c backend，减少 wrapper/copy 或做 shape-safe gating | attention2d kernel 或 wrapper 总时间下降，TTFT 改善 | 重复 D2 式强制 2D/3D 分支或改变输出行为 |
| H12 Fill/metadata source attribution | R11 显示 `FillFunctor<int>` 占 `8.386%`，H8.1/H12.1 两类 padding cache 无收益 | `gpu_model_runner.py`、attention metadata、block table 构造、临时 tensor 初始化点 | GPU tensor fill、metadata 初始化、图捕获 padding | 先用 stack/source attribution 锁定具体 fill，再决定是否合并或消除 | FillFunctor 时间下降且输出/finish 不变 | 未归因前继续做缓存补丁，或触碰 batch scheduler |

R25 最终状态：H10.8 t640 wave-pair reduction 与 H11.5 wide-causal tile
已完成各自和组合小样本；GDN MFMA、H10.9 backend selector、H11.6 attention
配置矩阵和 H10.10 K6144 exact gate 均已否决。H10.10 虽完成两轮微基准、
production build/install/validation，但固定 all3 的三档 TPOT 全部回归；其
600 秒外层窗口、最终 health、服务停止和 H10.8 wheel/source 回退均已完成。
最终计分栈为 H11.5 + H10.8。三次固定 full `all`、SLA 和 accuracy
均已完成；`K=1.0`，最终平均分 `88.548456`、相对 R24 平均加权
`+10.236157%`，未达两个性能终止条件。SLA 全部通过，三轮输出
一致，但仍有相对 R24 的漂移；固定 accuracy 未观察到计分精度下降。
服务已清洁停止。R24 wheel/evidence 继续冻结作为直接比较与回退基线；
5 小时终止条件已满足，R25 按时间条件结束，final evidence 已冻结并校验。

### 终止后冻结与交接

1. 保留 R24 wheel、源码快照和 evidence 为 R25 的冻结直接对照/回退基线；
   冻结 R25 full x3、SLA、accuracy、输出审计和服务停止 evidence。
2. H11.5、H10.8 的独立/组合构建、数值和至少 600 秒小样本已完成；继续保留其 evidence，不重复把已完成步骤写成待办。
3. H10.10 的 600 秒窗口、最终 health 和回退证据已完成；保留被拒
   wheel/源码/小样本 evidence，不得用 K6144 standalone 微基准覆盖端到端
   负结果。运行时已重装 H10.8 wheel，repo 与 installed hashes 已对齐。
4. accuracy 已完成且 `K=1.0`；保留固定脚本 Final Results、OpenCompass
   output directory、原生 aggregation `0.00` 与固定 Counter 复算 `100.00`
   的口径说明。5 小时终止 epoch `1783842849` 已到达，R25 按时间条件结束；
   固定 full x3/SLA/输出审计、服务清理、accuracy 与 final evidence manifests
   均已冻结，R25 下不再启动新的性能实验。
5. 提交或迁移时必须同时携带 R23/R24 累计栈 evidence、R25 tracked diff、完整 untracked GQA6 文件、source/runtime manifest 和 final evidence manifest；不能只携带 tracked diff。
6. 保留 R24 相对 R23 的输出漂移记录。任何 R25 候选必须继续做 paired comparison、停止原因/长度审计和固定 OpenCompass accuracy。
7. H10.8 只替换 H10.7 三个大投影的等价 config，保持 gfx936、BF16、`n=1`、`k=5120`、无 bias 和精确 m gate；`m=96` 保留原 LLMM1，不重新启用失败的 wvSplitK。
8. H11.5 只对 gfx936、BF16、head256、GQA6、单序列长 prefill 使用 wide-causal path；短 query、多序列、decode 和非目标 shape 回退 H11.4/AITER 原路径。
9. D2、H4.3、H4.5、H10.5 以及 H8.1/H11.1/H11.2/H12.1 等无收益或错误候选继续排除，不作为 R25 基础。
10. 固定 `run_throughput.sh`、`run_accuracy.sh`、`start_vllm.sh`、模型权重、tokenizer、chat template、serve 参数和 scheduler 边界继续保持只读；每轮小样本除明确报错外外层窗口至少 600 秒。

### 研究循环

Inner loop：

1. 选择最高优先级的未验证假设。
2. 写实验协议：触发条件、源码路径、预测、失败判据。
3. 做最小源码改动。
4. 重新编译并安装 wheel。
5. 用固定启动和固定吞吐脚本运行。
6. 记录 evidence card，必须列出“累计优化栈”和“本轮新增 diff”。
7. 若失败，写明它排除了什么；若成功，再进入消融。

Outer loop：

1. 每 3-5 个候选或遇到矛盾结果时，重新综合 backend 矩阵和瓶颈归因。
2. 调整 H1-H9 及 H1.x decode 子项优先级。
3. 删除没有路径证据、SLA 证据或精度证据的候选。
4. 只有完整 evidence card 支持的结果才能进入最终结论。
