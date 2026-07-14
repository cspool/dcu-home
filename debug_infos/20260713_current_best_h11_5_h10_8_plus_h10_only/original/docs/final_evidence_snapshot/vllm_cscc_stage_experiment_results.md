# vllm_cscc 阶段性实验完整结果

本文档保存阶段性实验的完整结果、路径、指标和结论。主计划只引用这里的索引，不重复展开明细。

## 索引

| ID | 主题 | 结论 |
| --- | --- | --- |
| R0 | Official baseline full all 基准 | 吞吐公式基准分 `60.00/100`，baseline 精度系数 `1.0` |
| R1 | H6.1c ROCm AITER Unified Attention | 上一可计分最佳；完整 `all` 加权吞吐提升 `+21.87%`，综合分约 `69.57/100` |
| R2 | H6.1c accuracy 对照 | 相对 official baseline 未观察到精度下降 |
| R3 | H5.1 上游 ROCm FlashAttention 强制优先 | 输出行为异常，实验无效并排除 |
| R4 | 早期单条 profile | 只作为定位起点，不作为验收结论 |
| R5 | H4.1 GDN prefill state 初始化快路径 | 小样本相对 H6.1c 加权 `+0.05%`，未达新增 `+20%` 目标 |
| R6 | H4.2 GDN prefill no-initial-state specialization | 小样本相对 H6.1c 加权 `+0.047%`，未达新增 `+20%` 目标 |
| R7 | H4.3 FLA L2 norm kernel 变体 | 小样本加权 `-2.87%`，且输出 token 数变化；回滚 |
| R8 | H4.4 GDN prefill `wy_fast` tile 变体 | 小样本相对 H6.1c 加权 `+0.134%`，未达新增 `+20%` 目标；不晋级 |
| R9 | D1 GDN decode packed validate 跳过 | 小样本相对 H6.1c 加权 `+0.049%`，噪声级；不作为有效收益 |
| R10 | D2 AITER unified attention decode 2D 分支强制 | 小样本相对 H6.1c 加权 `-15.56%`，明确降速；回滚 |
| R11 | 固定配置 DCU/hipprof profiling | PCIe 非瓶颈，显存容量接近打满；热点为 GEMM、AITER attention 2D、Fill/elementwise、GDN prefill |
| R12 | H8.1 padded block table fill cache | 已实现、编译、小样本验证；吞吐无实质提升，源码和运行 wheel 已恢复到工作基线 |
| R13 | H12.1 GDN CUDA graph padding fill cache | 已实现、强制重编 wheel、小样本验证；吞吐 `+0.055%`，噪声级，不作为有效收益 |
| R14 | H10.1 ROCm skinny GEMM 可用性探测 | `wvSplitK` 扩展未注册，不能源码默认启用该路径 |
| R15 | H11.1 AITER descale expand view cache | 已实现、强制重编 wheel、小样本验证；吞吐 `+0.015%`，噪声级，不作为有效收益 |
| R16 | H4.5 Qwen3.5 GDN core output `empty` allocation | 已实现、强制重编 wheel、小样本验证；输出 token 数大幅变化，正确性失败并回滚 |
| R17 | H10.2 unquantized GEMM dispatch cache | 已实现、强制重编 wheel、小样本验证；吞吐 `+0.033%`，噪声级，不作为有效收益 |
| R18 | H10.3 Linear/GEMM source attribution | 主目标是 MLP、GDN 和 attention 的大投影；`in_proj_ba` 小 GEMM 不是主目标 |
| R19 | H11.2 AITER KV cache view cache | 小样本吞吐 `-0.055%`；无收益并回滚 |
| R20 | H10.4 `_rocm_C` + gfx936 LLMM1 exact gate | all3 小样本相对 H6.1c 加权 `+13.5204%`；8-16K/16-32K 输出有变化，只作为晋级信号 |
| R21 | H11.3 GQA6 prefill non-overlap kernel | all3 小样本相对 H6.1c 加权 `+22.1103%`、相对 H10.4 `+7.6875%`；最终闭环见 R23 |
| R22 | H10.5 gfx936 `wvSplitK` | `96x5120` 直接对照数值错误，正确性失败；已回退并清洁重建 wheel |
| R23 | H10.4 和 H11.3 full all、accuracy 与重复性闭环 | 三次 full 相对 H6.1c 加权均值 `+23.7651%`，accuracy 系数 `1.0`，综合分均值 `79.0289/100`；作为 R24 的上一可计分最佳 |
| R24 | H11.4 adaptive GQA6 prefill 和 H10.7 gfx936 strided LLMM1 闭环 | 三次 full 综合分均值 `85.7075/100`，相对 R23 加权提升 `+20.5833%`，accuracy 系数 `1.0`；同时命中 85 分和相对当前最佳 +20% 终止条件 |
| R25 | H11.5/H10.8 候选筛选与 H10.10 provisional | H11.5/H10.8 已完成 full x3/SLA/accuracy，`K=1.0`，最终三轮均分 `88.548456`、相对 R24 平均 `+10.236157%`；GDN MFMA、H10.9 backend no-op、H11.6 attention 配置和 H10.10 均已否决；服务已无残留；5 小时条件于 epoch `1783842849` 满足，R25 按时间条件结束，final evidence 已冻结 |

## R0 Official baseline full all 基准

baseline wheel 状态：

- `pip show vllm`：`0.18.1+das.dtk2604`，安装位置 `/usr/local/lib/python3.10/dist-packages`。
- installed marker：`vllm/platforms/rocm.py` 中 `ROCM_AITER_UNIFIED_ATTN` 计数为 `3`；`rocm_aiter_unified_attn.py` 仍包含 `output_scale=`，符合保存的 official baseline wheel 形态。

吞吐基准：

- 路径：`/public/home/tangyu408/testdata/goal_runs/20260710_040000_official_baseline_full_all/throughput_all`
- 命令：固定 `run_throughput.sh all`，未传第二参数。
- 三档均 `completed=50`、`failed=0`。

| 档位 | output throughput | TTFT P99 | TPOT P99 | 单档公式基准分 |
| --- | ---: | ---: | ---: | ---: |
| `4-8K` | `12.2076` | `4792.48 ms` | `68.96 ms` | `12.00 / 20` |
| `8-16K` | `8.8108` | `24886.19 ms` | `70.37 ms` | `30.00 / 50` |
| `16-32K` | `5.3902` | `28740.84 ms` | `71.82 ms` | `18.00 / 30` |

baseline 相对自身提升率为 `0`，因此吞吐公式基准分为 `60.00/100`。

准确率基准：

- 路径：`/public/home/tangyu408/testdata/goal_runs/20260710_061800_official_baseline_full_accuracy_all_dtk_env`
- 命令：固定 `run_accuracy.sh all`，未传第二参数。
- 输出目录：`/public/home/tangyu408/testdata/accuracy_debug/output/local_accuracy_qwen35/20260710_141706`
- 口径：以 `run_accuracy.sh` 最终表为准；OpenCompass 原始 summary 中 RULER 聚合任务可能未按脚本重算，不能替代最终表。

| 数据集 | metric | baseline accuracy |
| --- | --- | ---: |
| `hotpotqa` | `score` | `77.96` |
| `gov_report` | `score` | `32.96` |
| `retrieval_multi_point` | `accuracy` | `100.00` |
| `aggregation_keyword_aggregation` | `accuracy` | `100.00` |

baseline 精度系数按定义为 `1.0`。若综合分按“吞吐公式分乘精度系数”计算，则 official baseline 综合基准分为 `60.00/100`。

## R1 H6.1c ROCm AITER Unified Attention

实验时间：`2026-07-09`。

源码变更：

- `vllm/platforms/rocm.py`：当 `aiter.ops.triton.unified_attention` 可导入时，将 `ROCM_AITER_UNIFIED_ATTN` 放在默认 `TRITON_ATTN` 前。
- `vllm/v1/attention/backends/rocm_aiter_unified_attn.py`：适配当前容器内 AITER unified attention 函数签名，移除不支持的 `sinks` 与 `output_scale` 关键字，并将 `supports_sink()` 置为 `False`。
- `setup.py`、`vllm/version.py`：保留 wheel 构建所需的版本生成修正。

路径证据：

- 服务日志确认命中 `Using ROCM_AITER_UNIFIED_ATTN attention backend out of potential backends: ['ROCM_AITER_UNIFIED_ATTN', 'TRITON_ATTN']`。
- 固定 `start_vllm.sh` 未修改。
- no-proxy API 快速检查返回正常中文短答，`finish_reason=stop`，未出现 H5.1 的重复输出失控现象。

构建与结果路径：

- build/install：`/public/home/tangyu408/testdata/goal_runs/20260709_225549_candidate_aiter_unified_no_output_scale_build`
- serve：`/public/home/tangyu408/testdata/goal_runs/20260709_225731_candidate_aiter_unified_no_output_scale_serve`
- 小样本 throughput：`/public/home/tangyu408/testdata/goal_runs/20260709_230206_candidate_aiter_unified_no_output_scale_all3`
- 完整 throughput：`/public/home/tangyu408/testdata/goal_runs/20260709_234926_h6_1c_full_throughput_all`

小样本吞吐：

| 档位 | baseline output throughput | H6.1c output throughput | 相对提升 | baseline total output tokens | H6.1c total output tokens | TTFT P99 对比 | TPOT P99 对比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4-8K` | `7.4017` | `9.5594` | `+29.15%` | `172` | `241` | `4344.36 -> 3187.06 ms` | `68.78 -> 68.36 ms` |
| `8-16K` | `7.9994` | `9.7455` | `+21.83%` | `600` | `610` | `13227.32 -> 7869.08 ms` | `70.03 -> 69.14 ms` |
| `16-32K` | `3.6039` | `5.2747` | `+46.36%` | `388` | `364` | `28405.19 -> 15154.81 ms` | `71.68 -> 70.31 ms` |

小样本 20/50/30 加权相对提升约 `+30.65%`。该结果只作为筛选信号，不能替代完整 `all`。

完整 `all` 吞吐：

- official baseline full all：`/public/home/tangyu408/testdata/goal_runs/20260710_040000_official_baseline_full_all/throughput_all`
- H6.1c full all：`/public/home/tangyu408/testdata/goal_runs/20260709_234926_h6_1c_full_throughput_all`
- 两组均使用固定 `start_vllm.sh` 与固定 `run_throughput.sh all`，未传第二参数，`MAX_CONCURRENCY=1`，三档均 `completed=50`、`failed=0`。

| 档位 | official baseline output throughput | H6.1c output throughput | 相对提升 | TTFT P99 对比 | TPOT P99 对比 | SLA |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `4-8K` | `12.2076` | `12.7816` | `+4.70%` | `4792.48 -> 3459.79 ms` | `68.96 -> 68.44 ms` | 通过 |
| `8-16K` | `8.8108` | `10.5395` | `+19.62%` | `24886.19 -> 10020.25 ms` | `70.37 -> 69.39 ms` | 通过 |
| `16-32K` | `5.3902` | `7.3887` | `+37.08%` | `28740.84 -> 15255.00 ms` | `71.82 -> 70.40 ms` | 通过 |

完整 `all` 的 20/50/30 加权相对提升为 `+21.87%`，按赛题单档公式汇总得分约 `69.57/100`。相比小样本 `+30.65%`，全量结果下修 `8.78` 个百分点；后续结论以完整 `all` 为准。

## R2 H6.1c accuracy 对照

H6.1c accuracy 路径：

- `/public/home/tangyu408/testdata/goal_runs/20260710_030800_h6_1c_full_accuracy_all_dtk_env`
- output dir：`/public/home/tangyu408/testdata/accuracy_debug/output/local_accuracy_qwen35/20260710_110647`

| 数据集 | baseline | H6.1c | 观察到的下降 |
| --- | ---: | ---: | ---: |
| `hotpotqa` | `77.96` | `77.96` | `0.00` |
| `gov_report` | `32.96` | `32.97` | `0.00` |
| `retrieval_multi_point` | `100.00` | `100.00` | `0.00` |
| `aggregation_keyword_aggregation` | `100.00` | `100.00` | `0.00` |

结论：以固定脚本最终表为口径，H6.1c 相对 official baseline 未观察到精度下降。若精度系数按无下降计为 `1.0`，H6.1c 综合分约为 `69.57/100`。

正确性补充：

- 小样本 `generated_texts` 与 baseline 逐请求文本哈希比对：`9` 条中 `5` 条完全一致，`4` 条不一致。
- 不一致样本没有出现重复 token 或明显失控输出，但存在措辞、长度和局部内容差异；最终正确性结论以 OpenCompass/accuracy 固定脚本结果为准。

## R3 H5.1 上游 ROCm FlashAttention 强制优先

实验时间：`2026-07-09`。

源码变更：

- `vllm/platforms/rocm.py`：在 ROCm backend priority 中将 upstream `flash_attn` 放在 `TRITON_ATTN` 前。
- `vllm/model_executor/models/config.py`：为 upstream ROCm FlashAttention 将 hybrid attention/mamba cache block 对齐到 `64`，实际 attention block size 从 `784` 变为 `832`。
- `vllm/v1/attention/backends/flash_attn.py`：增加 ROCm upstream `flash_attn_varlen_func` 调用路径，并把 `cu_seqlens_k` 从 forward 中的临时构造移动到 metadata build 阶段，以避免 HIP graph capture 中的 GPU 写入。

有效路径证据：

- 服务日志确认命中 `Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'TRITON_ATTN']`。
- 服务日志确认 `Setting attention block size to 832 tokens`，并成功完成启动。
- 固定 `start_vllm.sh` 未修改；固定 `run_throughput.sh all 3` 运行完成。

构建与结果路径：

- build/install：`/public/home/tangyu408/testdata/goal_runs/20260709_220658_candidate_flashattn_rocm_cuseq_build`
- serve：`/public/home/tangyu408/testdata/goal_runs/20260709_220921_candidate_flashattn_rocm_cuseq_serve`
- throughput：`/public/home/tangyu408/testdata/goal_runs/20260709_221245_candidate_flashattn_rocm_cuseq_all3`

小样本吞吐现象：

| 档位 | baseline output throughput | H5.1 output throughput | baseline total output tokens | H5.1 total output tokens | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `4-8K` | `7.4017` | `14.5549` | `172` | `3072` | 输出长度异常 |
| `8-16K` | `7.9994` | `14.3017` | `600` | `3072` | 输出长度异常 |
| `16-32K` | `3.6039` | `13.9549` | `388` | `3072` | 输出长度异常 |

失败判据：

- 三个档位均输出满 `3 * 1024 = 3072` tokens，而有效 baseline 的输出 token 数分别为 `172/600/388`。
- no-proxy API 快速检查中，简单中文问答输出大量重复感叹号，说明 deterministic 输出已明显偏离。
- 因输出行为和停止行为改变，该候选的 `output_throughput` 提升不能进入有效优化结论。

后续规则：

- H5 的 upstream ROCm FlashAttention 路径只有在先修复输出正确性、finish reason、stop reason 和输出哈希后才能重新进入候选。
- 任何依赖“输出提前停止变成输出满长”得到的吞吐提升都必须判为无效。

## R4 早期单条 profile

已有单条 profile 只作为定位起点，不作为验收结果。

| Context | Input tokens | Output tokens | Output throughput | TTFT ms | TPOT ms | E2E ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `4-8K` | `7574` | `88` | `7.70` | `4350.36` | `69.76` | `10419.54` |
| `8-16K` | `13962` | `92` | `4.65` | `12303.98` | `71.01` | `18765.48` |
| `16-32K` | `20574` | `23` | `0.85` | `24631.13` | `71.79` | `26210.50` |

这些 profile 的实际输出 token 数远小于 `custom-output-len=1024`，说明输出提前停止会显著影响官方主指标。正式比较必须逐请求检查输出 token 数、finish reason、stop reason 和输出哈希，并用 OpenCompass 做精度验收。

## R5 H4.1 GDN prefill state 初始化快路径

实验时间：`2026-07-10`。

源码变更：

- `vllm/v1/attention/backends/gdn_attn.py`：在 GDN prefill metadata 中增加 `has_initial_state_any` 和 `has_initial_state_all`，用已有 CPU metadata 判断 non-spec prefill 是否全部没有历史 recurrent state，避免在模型 forward 热路径做 GPU->CPU 同步。
- `vllm/model_executor/models/qwen3_next.py`：当 `has_initial_state_any is False` 时，用 `ssm_state.new_zeros(...)` 直接构造初始 state；当 `has_initial_state_all is True` 时跳过 mask 清零。

构建与运行证据：

- build：`/public/home/tangyu408/testdata/goal_runs/20260710_151814_candidate_gdn_prefill_statefast_build`
- wheel：`/public/home/tangyu408/vllm_cscc/dist/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl`
- wheel sha256：`fea33d9a835c0310af1dfdc0edb560d20a4777358561cfb5f0b0464cd16c76bd`
- serve：`/public/home/tangyu408/testdata/goal_runs/20260710_152203_candidate_gdn_prefill_statefast_serve2`
- 小样本 throughput：`/public/home/tangyu408/testdata/goal_runs/20260710_152634_candidate_gdn_prefill_statefast_all3`
- 固定脚本 hash：`run_throughput.sh = adf0cf91266745b37df916926c7d495ec79f00a11be653c219d1d5df4d93c681`，`start_vllm.sh = 7c3e8c5ecdf02109e02af8c3b5ba05050b26339c7f50869b5288eea359364fad`
- 小样本运行命令：`./run_throughput.sh all 3`；脚本耗时 `322s`，包装命令补足到 `10min+` 后读取结果。

小样本对比 H6.1c 小样本最佳：

| 档位 | H6.1c output throughput | H4.1 output throughput | 相对 H6.1c |
| --- | ---: | ---: | ---: |
| `4-8K` | `9.5594` | `9.5629` | `+0.04%` |
| `8-16K` | `9.7455` | `9.7511` | `+0.06%` |
| `16-32K` | `5.2747` | `5.2771` | `+0.04%` |

20/50/30 加权 throughput 为 `8.3712`，相对 H6.1c 小样本最佳 `8.3670` 提升 `+0.05%`，距离新增 `+20%` 目标 `10.0405` 明显不足。

结论：H4.1 是正确性风险较低的源码快路径，但 gather/mask 清零不是当前 GDN prefill 的主要瓶颈；不进入有效收益结论。下一步应继续查看 FLA chunk kernel 是否支持 `initial_state=None` 的零初始状态路径，或转向 GDN prefill 主体 kernel 的 launch/HBM 归因。

## R6 H4.2 GDN prefill no-initial-state specialization

实验时间：`2026-07-10`。

源码变更：

- 在 H4.1 基础上，`qwen3_next.py` 的 FLA/Triton GDN prefill backend 增加 `supports_none_initial_state` 标记。
- 当 metadata 判定 non-spec prefill 全部没有历史 recurrent state 时，传 `initial_state=None`，使 `chunk_gated_delta_rule_fwd_h` 走 Triton `USE_INITIAL_STATE=False` specialization，跳过初始 state tensor 分配和 kernel 内 state load。
- `_warmup_prefill_kernels()` 增加 `(initial_state=None, output_final_state=True)` warmup case，覆盖 no-initial-state specialization。

构建与运行证据：

- build：`/public/home/tangyu408/testdata/goal_runs/20260710_154128_candidate_gdn_prefill_none_state_build`
- wheel sha256：`95b68517ca7425db15d85e51dd479ae1e7abb8286aac4f44c0402c059ebe7c4f`
- serve：`/public/home/tangyu408/testdata/goal_runs/20260710_154410_candidate_gdn_prefill_none_state_serve`
- 小样本 throughput：`/public/home/tangyu408/testdata/goal_runs/20260710_154849_candidate_gdn_prefill_none_state_all3`
- 小样本运行命令：`./run_throughput.sh all 3`；脚本耗时 `322s`，包装命令补足到 `10min+` 后读取结果。

小样本对比 H6.1c 小样本最佳：

| 档位 | H6.1c output throughput | H4.2 output throughput | 相对 H6.1c |
| --- | ---: | ---: | ---: |
| `4-8K` | `9.5594` | `9.5611` | `+0.02%` |
| `8-16K` | `9.7455` | `9.7504` | `+0.05%` |
| `16-32K` | `5.2747` | `5.2785` | `+0.07%` |

20/50/30 加权 throughput 为 `8.3710`，相对 H6.1c 小样本最佳 `8.3670` 提升 `+0.047%`；相对 H4.1 加权结果 `8.3712` 略低 `-0.003%`。距离新增 `+20%` 目标 `10.0405` 明显不足。

结论：no-initial-state specialization 能命中并保持 `failed=0`，但初始 state load/清零路径不是主要瓶颈。后续 GDN prefill 优化应转向 `chunk_local_cumsum`、`chunk_scaled_dot_kkt_fwd`、`solve_tril`、`recompute_w_u_fwd`、`chunk_gated_delta_rule_fwd_h`、`chunk_fwd_o` 等主体 kernel 的 launch/HBM 归因。

## R7 H4.3 FLA L2 norm kernel 变体

实验时间：`2026-07-10`。

源码变更：

- `vllm/model_executor/layers/fla/ops/l2norm.py`：将 `USE_DEFAULT_FLA_NORM` 的源码默认值从 `0` 改为 `1`，使 GDN/KDA prefill 的 q/k L2 norm 走现有 autotuned FLA norm kernel 分支，而非当前默认 `l2norm_fwd_kernel2`。

构建与运行证据：

- build：`/public/home/tangyu408/testdata/goal_runs/20260710_160215_candidate_gdn_prefill_default_fla_norm_build`
- wheel sha256：`5e6bf105a1c372a0b972b8430aa5139e7e3ecd4dc9126b0ba119ad535ec8bed1`
- serve：`/public/home/tangyu408/testdata/goal_runs/20260710_160502_candidate_gdn_prefill_default_fla_norm_serve`
- 小样本 throughput：`/public/home/tangyu408/testdata/goal_runs/20260710_160907_candidate_gdn_prefill_default_fla_norm_all3`
- 小样本运行命令：`./run_throughput.sh all 3`；脚本耗时 `318s`，包装命令补足到 `10min+` 后读取结果。

小样本对比 H6.1c 小样本最佳：

| 档位 | H6.1c output throughput | H4.3 output throughput | 相对 H6.1c | H6.1c total output tokens | H4.3 total output tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `4-8K` | `9.5594` | `8.3836` | `-12.30%` | `241` | `172` |
| `8-16K` | `9.7455` | `9.7450` | `-0.005%` | `610` | `610` |
| `16-32K` | `5.2747` | `5.2598` | `-0.28%` | `364` | `362` |

20/50/30 加权 throughput 为 `8.1272`，相对 H6.1c 小样本最佳 `8.3670` 下降 `-2.87%`。

结论：该 kernel 变体没有吞吐收益，并且 4-8K 与 16-32K 输出 token 数发生变化，不能进入有效收益结论。该变更应回滚；后续若重新研究 L2 norm，只能在先完成逐请求输出/精度审计的前提下进入候选。

## R8 H4.4 GDN prefill `wy_fast` tile 变体

实验时间：`2026-07-10`。

源码变更：

- `vllm/model_executor/layers/fla/ops/wy_fast.py`：将 `recompute_w_u_fwd()` 中的 `BK/BV` 从 `64/64` 调整为 `128/128`。
- 该变更只改变 `A @ v` 和 `A @ k` 的列向量分块大小，数学表达不变。目标是让 Qwen3.5-27B 的 GDN `K/V=128` 场景把两个分块循环各从 2 次降到 1 次。

构建与运行证据：

- build：`/public/home/tangyu408/testdata/goal_runs/20260710_163212_candidate_gdn_prefill_wy128_build`
- wheel sha256：`c31633e1cc7ee6356e0052c4bc2e2d28fb3770092a96d730ca8f8e556288213f`
- wheel copy：`/public/home/tangyu408/testdata/goal_runs/20260710_163212_candidate_gdn_prefill_wy128_build/vllm_h4_4_wy128.whl`
- serve：`/public/home/tangyu408/testdata/goal_runs/20260710_163535_candidate_gdn_prefill_wy128_serve`
- 小样本 throughput：`/public/home/tangyu408/testdata/goal_runs/20260710_163851_candidate_gdn_prefill_wy128_all3`
- 小样本运行命令：`./run_throughput.sh all 3`；脚本耗时 `339s`，包装命令补足到 `600s` 后读取结果。

小样本对比 H6.1c 小样本最佳：

| 档位 | H6.1c output throughput | H4.4 output throughput | 相对 H6.1c | H4.4 total output tokens | completed | failed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `4-8K` | `9.5594` | `9.5727` | `+0.1389%` | `241` | `3` | `0` |
| `8-16K` | `9.7455` | `9.7569` | `+0.1167%` | `610` | `3` | `0` |
| `16-32K` | `5.2747` | `5.2843` | `+0.1820%` | `364` | `3` | `0` |

20/50/30 加权 throughput 为 `8.3783`，相对 H6.1c 小样本最佳 `8.3670` 提升 `+0.134%`；新增 `+20%` 目标线为 `10.0405`。

结论：H4.4 没有形成有效性能突破。输出 token 数与 H6.1c 小样本一致，`failed=0`，说明该 tile 变体未暴露明显功能错误；但收益只有噪声级别，不值得进入 full `all` 和 accuracy 晋级。该变更不作为当前有效候选保留，后续 GDN prefill 优化应转向 profiler 证明的主体瓶颈，而不是继续盲扫 `BK/BV`。

## R9 D1 GDN decode packed validate 跳过

实验时间：`2026-07-10`。

源码变更：

- `vllm/model_executor/layers/fla/ops/fused_recurrent.py`：为 `fused_recurrent_gated_delta_rule_packed_decode()` 增加 `validate: bool = True`，用显式参数控制 Python 侧 shape/dtype/device 检查。
- `vllm/model_executor/models/qwen3_next.py`：在 Qwen3Next packed non-spec decode 快路径中传入 `validate=False`，避免每 token decode 重复执行已由上层保证的不变量检查。

正确性快检：

- same-input GPU 对照：`validate=True` 与 `validate=False` 的 `max_out_diff=0.0`，`max_state_diff=0.0`。
- 该快检只能证明局部 kernel wrapper 数值一致；进入有效收益仍必须依赖固定脚本吞吐和后续 accuracy。

构建与运行证据：

- build：`/public/home/tangyu408/testdata/goal_runs/20260710_170553_candidate_decode_packed_validate_skip_build`
- wheel sha256：`7a995a16f92a0ef29568c63781a6fb32b924f80157a0bd74a0e9069999acea08`
- serve：`/public/home/tangyu408/testdata/goal_runs/20260710_170728_candidate_decode_packed_validate_skip_serve`
- 小样本 throughput：`/public/home/tangyu408/testdata/goal_runs/20260710_171056_candidate_decode_packed_validate_skip_all3`
- 小样本运行命令：`./run_throughput.sh all 3`；脚本结束后补足总等待到 `600s`。

小样本对比 H6.1c 小样本最佳：

| 档位 | H6.1c output throughput | D1 output throughput | 相对 H6.1c | total output tokens | completed | failed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `4-8K` | `9.5594` | `9.5615` | `+0.0216%` | `241` | `3` | `0` |
| `8-16K` | `9.7455` | `9.7503` | `+0.0494%` | `610` | `3` | `0` |
| `16-32K` | `5.2747` | `5.2791` | `+0.0823%` | `364` | `3` | `0` |

20/50/30 加权 throughput 为 `8.3712`，相对 H6.1c 小样本最佳 `8.3670` 提升 `+0.0493%`；decode 目标新增 `+10%` 门槛为 `9.2038`。

结论：D1 逻辑正确性风险低，但收益只有噪声级，说明 Python 侧 packed decode 参数检查不是当前单请求 decode 的主要瓶颈。该结果不进入有效性能提升结论；后续 decode 优化不能继续围绕类似 wrapper 检查做微调。

## R10 D2 AITER unified attention decode 2D 分支强制

实验时间：`2026-07-10`。

源码变更：

- `vllm/v1/attention/backends/rocm_aiter_unified_attn.py`：尝试在 `max_query_len == 1` 的 decode 场景中把传给 AITER `unified_attention()` 的 `max_seqlen_q` 从 `1` 改为 `2`，使 AITER Python 分支选择 2D kernel，避开其 single-query segmented 3D 分支的临时 tensor 分配。
- 该变更只作为 D2 候选测试；测试后已回滚源码，不作为后续基础。

构建与运行证据：

- build：`/public/home/tangyu408/testdata/goal_runs/20260710_173124_candidate_decode_aiter_2d_build`
- wheel sha256：`09df2ec5d2280af5ad9bc58c186763209ae0e5f848dbc95bf2577fce23a3472f`
- serve：`/public/home/tangyu408/testdata/goal_runs/20260710_173257_candidate_decode_aiter_2d_serve`
- 小样本 throughput：`/public/home/tangyu408/testdata/goal_runs/20260710_173600_candidate_decode_aiter_2d_all3`
- 小样本运行命令：`./run_throughput.sh all 3`；完整运行并等待超过 `10min` 后记录。

小样本对比 H6.1c 小样本最佳：

| 档位 | H6.1c output throughput | D2 output throughput | 相对 H6.1c | total output tokens | completed | failed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `4-8K` | `9.5594` | `8.5380` | `-10.6847%` | `241` | `3` | `0` |
| `8-16K` | `9.7455` | `8.0793` | `-17.0975%` | `612` | `3` | `0` |
| `16-32K` | `5.2747` | `4.3933` | `-16.7102%` | `378` | `3` | `0` |

20/50/30 加权 throughput 为 `7.0652`，相对 H6.1c 小样本最佳 `8.3670` 下降 `-15.5589%`，相对 D1 下降 `-15.6005%`。decode 目标新增 `+10%` 门槛为 `9.2038`，未达标。

结论：AITER 的 single-query segmented 3D decode 分支虽然存在临时分配，但在当前 gfx936/Qwen3.5-27B 形状下明显快于强制 2D 分支。该实验排除“仅通过 Python branch selector 强制 2D decode 获益”的方向；后续若继续 attention decode，必须基于 kernel timeline 和 counter 找到真实瓶颈，而不是继续替换 AITER 分支。

## R11 固定配置 DCU/hipprof profiling

实验时间：`2026-07-10` 至 `2026-07-11`。

目标：在固定 `start_vllm.sh` 与固定 `run_throughput.sh` 的单请求口径下，区分当前工作栈的主要瓶颈，避免继续依赖猜测或只围绕 decode wrapper 做微调。

DCU SMI full `all`：

- 路径：`/public/home/tangyu408/testdata/profile_runs/dcu_smi_fixed_working_stack_20260710_225924`
- 命令：固定 `run_throughput.sh all`，未传第二参数。
- 结果：三档均 `completed=50`、`failed=0`。

| 档位 | output throughput | mean TTFT | mean TPOT |
| --- | ---: | ---: | ---: |
| `4-8K` | `12.7727` | `2575.09 ms` | `68.226 ms` |
| `8-16K` | `10.5370` | `7120.617 ms` | `68.951 ms` |
| `16-32K` | `7.3900` | `14422.295 ms` | `70.091 ms` |

`dcu_smi_summary.md` 关键结论：

- 采样数 `1873`。
- HCU mean `88.362%`，p50 `96.200%`，p95/p99/max `100%`。
- HCU memory mean `96.470%`。
- PCIe BW mean `0.152 MB/s`，p50 `0`，p95 `0.711 MB/s`，max `2.958 MB/s`。
- VRAM mean `63100 MiB`，p50/p95/p99/max `63446 MiB`。

解释：PCIe/host-device 传输不是当前瓶颈；HCU 大部分时间较高但仍有调度/launch 或 kernel gap；显存容量接近打满，vLLM 启动日志中的 KV cache 容量和 graph capture 限制仍是背景约束，但本轮不通过改 serve 参数获益。

Hipprof trace 小样本：

- 路径：`/public/home/tangyu408/testdata/profile_runs/hipprof_fixed_trace_clean_16_32K_n4_20260711_003100`
- 命令：固定脚本小样本 `run_throughput.sh 16-32K 4`，仅用于 kernel attribution，不作为最终吞吐验收。
- 小样本结果：`completed=4`、`failed=0`、output throughput `8.82 tok/s`、mean TTFT `14519.04 ms`、mean TPOT `70.91 ms`。

按 kernel class 聚合：

| class | total time | 占比 |
| --- | ---: | ---: |
| `linear_gemm` | `116260.001 ms` | `53.221%` |
| `attention` | `66669.504 ms` | `30.520%` |
| `elementwise_or_reduction` | `20191.996 ms` | `9.243%` |
| `gdn_linear_attention` | `12017.869 ms` | `5.501%` |
| `kv_cache_or_copy` | `1366.924 ms` | `0.626%` |
| `other` | `1697.801 ms` | `0.777%` |
| `norm_rope` | `178.921 ms` | `0.082%` |
| `sampling` | `65.842 ms` | `0.030%` |

Top kernels：

| 排名 | kernel | time | 占比 |
| ---: | --- | ---: | ---: |
| 1 | rocBLAS `Cijk_Alik_Bljk_BBH_MT64x32x32...` | `62808.717 ms` | `28.752%` |
| 2 | `kernel_unified_attention_2d` | `58341.847 ms` | `26.707%` |
| 3 | rocBLAS `MT32x16x4...` | `21078.050 ms` | `9.649%` |
| 4 | `FillFunctor<int>` | `18320.211 ms` | `8.386%` |
| 8 | `chunk_fwd_kernel_o` | `4537.044 ms` | `2.077%` |
| 9 | `kernel_unified_attention_3d` | `4464.445 ms` | `2.044%` |
| 11 | `flash_fwd_kernel_16x64_prefetch...` | `3863.212 ms` | `1.768%` |
| 17 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | `1031.816 ms` | `0.472%` |
| 40 | `reshape_and_cache_flash_kernel` | `144.324 ms` | `0.066%` |

PMC attention2d：

- 路径：`/public/home/tangyu408/testdata/profile_runs/hipprof_fixed_pmc_attention2d_16_32K_n1_20260711_005039`
- filter：`kernel_unified_attention_2d`
- rows `512`，total kernel time `27450.012 ms`
- weighted ALU instr `%` `56.587`，perf `7350.812 GFLOPS`，shared memory bank conflict `28.134%`，L1 active `18.595%`，L1 stalled `0.007%`，L2 write stalled `0.001%`，L2 hit `96.210%`

PMC top GEMM：

- 路径：`/public/home/tangyu408/testdata/profile_runs/hipprof_fixed_pmc_topgemm_16_32K_n1_20260711_005732`
- filter：trace top-1 rocBLAS Cijk kernel
- rows `7952`，total kernel time `3081.030 ms`
- weighted ALU instr `%` `4.956`，perf `427.657 GFLOPS`，shared memory bank conflict `20.254%`，L1 active `41.958%`，L1 stalled `0.270%`，L2 write stalled `0`，L2 hit `1.477%`

解释：attention2d 更像 compute/shared-memory tiling 问题，不是 PCIe 或外部拷贝问题；top GEMM 在 PMC 中呈现低 L2 hit 与低有效 GFLOPS，更像小/瘦 GEMM 或 GEMV 形状下的缓存和流式访存效率问题。hipprof 文本没有可靠给出完整 L2 read/write KB 字段，因此不把它表述为已量化的 HBM 带宽上限。

结论：

- 当前最大热点是 Linear/GEMM 与 AITER unified attention 2D prefill，合计超过一半时间。
- Decode 确实影响 TPOT，但显式 decode kernel 在本轮 trace 中不是最大耗时；`kernel_unified_attention_3d` 和 packed recurrent decode 的占比低于 GEMM、attention2d、Fill/elementwise 和 GDN prefill 主体 kernel。
- 下一阶段优先级应调整为：GEMM/linear backend 与形状归因；AITER attention2d prefill kernel/wrapper；Fill/metadata elementwise 的真实来源；GDN prefill 主体 kernel。decode 专项保留，但不再作为唯一首要方向。

## R12 H8.1 padded block table fill cache

实验时间：`2026-07-11`。

触发条件：R11 trace 中 `FillFunctor<int>` 排名第 4，占总 kernel 时间 `8.386%`。源码检查后怀疑 `vllm/v1/worker/gpu_model_runner.py` 的 `_build_attention_metadata()` 在每轮构造 block table 时反复执行 padded row 的 `fill_(-1)`。

源码变更：

- `vllm/v1/worker/gpu_model_runner.py`：增加 `_cscc_block_table_pad_cache`，并新增 `_fill_padded_block_table_rows(...)`，尝试只填充新增或变化的 padded row 区间。
- 对 `for_cudagraph_capture` 和 encoder-only 新建 tensor 路径保留原始 fill 行为，避免图捕获和异构 cache group 语义风险。

构建与安装：

- build/install：`/public/home/tangyu408/testdata/goal_runs/20260711_011527_padfill_cache_build`
- 安装态验证：site-packages 中 `has_pad_cache=True`

小样本对照：

- 有效 pre-padfill baseline：`/public/home/tangyu408/testdata/goal_runs/20260711_010847_pre_padfill_baseline_16_32K_n4_noproxy`
- padfill candidate：`/public/home/tangyu408/testdata/goal_runs/20260711_011753_padfill_cache_16_32K_n4_noproxy`
- 命令：固定 `start_vllm.sh`，固定 `run_throughput.sh 16-32K 4`，仅加 no-proxy 修正本机 API 路由。

| 指标 | pre-padfill baseline | padfill candidate | 变化 |
| --- | ---: | ---: | ---: |
| completed / failed | `4 / 0` | `4 / 0` | 无变化 |
| output throughput | `8.88 tok/s` | `8.88 tok/s` | 无实质变化 |
| duration | `156.30 s` | `156.26 s` | `-0.04 s` |
| total input tokens | `86623` | `86623` | 无变化 |
| total output tokens | `1388` | `1388` | 无变化 |
| mean TTFT | `14521.67 ms` | `14519.89 ms` | `-1.78 ms` |
| p99 TTFT | `15247.48 ms` | `15238.29 ms` | `-9.19 ms` |
| mean TPOT | `70.10 ms` | `70.09 ms` | `-0.01 ms` |
| p99 TPOT | `70.40 ms` | `70.37 ms` | `-0.03 ms` |
| mean E2EL | `38863.67 ms` | `38853.31 ms` | `-10.36 ms` |

无效数据记录：

- `/public/home/tangyu408/testdata/goal_runs/20260711_010457_pre_padfill_baseline_16_32K_n4` 因未绕过代理，4 个请求均返回 503/Squid 路径；该结果不能进入任何 baseline 或优化结论。

回退：

- padfill cache 未达到可见收益，未进入 full `all` 和 accuracy。
- 源码已回退，重新 build/install：`/public/home/tangyu408/testdata/goal_runs/20260711_012537_restore_working_stack_build`
- 安装态验证：`vllm_version=0.18.1+das.dtk2604`，`has_pad_cache=False`，`gpu_model_runner=/usr/local/lib/python3.10/dist-packages/vllm/v1/worker/gpu_model_runner.py`

结论：该候选排除了“通过 Python 侧缓存 padded block table fill 区间即可显著降低 FillFunctor 热点”的假设。`FillFunctor<int>` 仍需要进一步定位到具体 tensor 和调用栈；没有 stack attribution 前，不继续做同类 Python cache 补丁。

## R13 H12.1 GDN CUDA graph padding fill cache

实验时间：`2026-07-11`。

触发条件：R11 trace 中 `FillFunctor<int>` 排名第 4，占总 kernel 时间 `8.386%`。H8.1 已排除 `gpu_model_runner.py` 的 padded block table Python cache 假设后，继续检查 GDN metadata builder，发现 `vllm/v1/attention/backends/gdn_attn.py` 在非 speculative decode full CUDA graph 路径中每步对 padding tail 执行：

- `non_spec_state_indices_tensor[num_decodes:].fill_(PAD_SLOT_ID)`
- `non_spec_query_start_loc[num_decodes + 1:].fill_(non_spec_num_query_tokens)`

源码变更：

- 在 `GDNAttentionMetadataBuilder.__init__()` 中预初始化 `spec_state_indices_tensor`、`non_spec_state_indices_tensor`、`spec_sequence_masks`、`non_spec_query_start_loc` 和 `num_accepted_tokens` 的 padding 默认值。
- 在非 speculative decode full CUDA graph 分支中记录上次 active decode 区间，只在 active prefix 变短或 query-start tail 值变化时补齐 padding tail。
- 不修改 batch scheduler，不改变有效 token 的 state index 或 query start prefix；speculative decode 路径暂不消除 fill。

构建与安装：

- 首次普通 `bdist_wheel`：`/public/home/tangyu408/testdata/goal_runs/20260711_h12_1_gdn_cg_padfill_build`
- 该 wheel 的 site-packages marker 为 `False`，说明复用了旧 `build/lib` Python 产物，不能作为有效测试依据。
- 强制重编并安装：`/public/home/tangyu408/testdata/goal_runs/20260711_h12_1_gdn_cg_padfill_rebuild_force`
- 安装态验证：`vllm_file=/usr/local/lib/python3.10/dist-packages/vllm/__init__.py`，`h12_1_marker=True`
- 因收益为噪声级，源码已回退到工作基线并重新强制重编安装：`/public/home/tangyu408/testdata/goal_runs/20260711_restore_after_h12_1_gdn_cg_padfill`
- 恢复态验证：`h12_1_marker=False`，`h41_h42_marker=True`

小样本对照：

- 对照 baseline：`/public/home/tangyu408/testdata/goal_runs/20260711_010847_pre_padfill_baseline_16_32K_n4_noproxy`
- H12.1 candidate：`/public/home/tangyu408/testdata/goal_runs/20260711_h12_1_gdn_cg_padfill_16_32K_n4_fixed_retry1`
- 命令：固定 `start_vllm.sh`，固定 `run_throughput.sh 16-32K 4`，仅加 no-proxy 修正本机 API 路由。
- 有效性：candidate 主体 benchmark `status=0`，`bench_elapsed_sec=227`，成功后额外等待 `373` 秒，满足小样本至少 10 分钟等待约束。

| 指标 | baseline | H12.1 candidate | 变化 |
| --- | ---: | ---: | ---: |
| completed / failed | `4 / 0` | `4 / 0` | 无变化 |
| output throughput | `8.880139 tok/s` | `8.885060 tok/s` | `+0.055%` |
| duration | `156.3039 s` | `156.2173 s` | `-0.0866 s` |
| total input tokens | `86623` | `86623` | 无变化 |
| total output tokens | `1388` | `1388` | 无变化 |
| mean TTFT | `14521.666 ms` | `14515.625 ms` | `-6.041 ms` |
| mean TPOT | `70.096 ms` | `70.047 ms` | `-0.048 ms` |
| median TPOT | `70.203 ms` | `70.116 ms` | `-0.086 ms` |

结论：

- H12.1 证明 GDN decode CUDA graph padding fill cache 在单请求固定脚本下不产生可计入的吞吐收益。
- `FillFunctor<int>` 热点不是通过这一处 tail fill 消除即可解决，或者该 fill 在无 profiler 环境下被其它 GEMM/attention 时间完全淹没。
- 后续不继续做同类 metadata padding cache 补丁。若重开 FillFunctor，必须先做 kernel-to-source stack attribution；主优化方向转回 R11 占比更高的 GEMM/linear 和 AITER attention2d。

## R14 H10.1 ROCm skinny GEMM 可用性探测

实验时间：`2026-07-11`。

触发条件：R11 显示 `linear_gemm` 占 `53.221%`，且 top GEMM PMC 呈现低 L2 hit 与低有效 GFLOPS。源码中 `vllm/model_executor/layers/utils.py` 已存在 ROCm skinny GEMM 路径：

- `ops.wvSplitK(weight, x_view, cu_count, bias)`：`0 < n <= 4` 的 skinny GEMM。
- `ops.LLMM1(weight, x_view, 4)`：`n == 1`、`k <= 8192`、无 bias 的路径。
- 这些路径默认受 `envs.VLLM_ROCM_USE_SKINNY_GEMM` 控制，固定 `start_vllm.sh` 不打开该环境变量。

微基准探测：

- 命令路径：`cd /tmp` 后直接导入 installed wheel 的 `vllm._custom_ops`。
- 探测形状：Qwen3.5 decode 主要线性形状，包括 `(m,k)=(14336,5120)`、`(5120,6144)`、`(34816,5120)`、`(5120,17408)`。
- 结果：首个形状调用 `ops.wvSplitK` 即失败：

```text
AttributeError: '_OpNamespace' '_rocm_C' object has no attribute 'wvSplitK'
```

结论：

- 当前 wheel/ROCm 扩展没有注册 `torch.ops._rocm_C.wvSplitK`。源码默认打开 skinny GEMM gate 会导致运行时崩溃，不能作为候选实施。
- 下一步若继续 H10，必须先确认可用 backend：hipBLASLt/rocBLAS shape attribution、AITER GEMM API 是否可在当前 DTK 环境稳定编译，或新增可编译的自定义 kernel；不能只解除现有 env gate。

## R15 H11.1 AITER descale expand view cache

实验时间：`2026-07-11`。

触发条件：R11 显示 `kernel_unified_attention_2d` 占 `26.707%`。`rocm_aiter_unified_attn.py` 每次 forward 都构造 `layer._k_scale.expand(descale_shape)` 与 `layer._v_scale.expand(descale_shape)` 后传入 AITER unified attention。该操作主要是 Python/view 开销，不直接改变 GPU kernel。

源码变更：

- `vllm/v1/attention/backends/rocm_aiter_unified_attn.py`：在 `RocmAiterUnifiedAttentionImpl` 中增加 `_descale_cache_shape`、`_descale_cache_scale_ids`、`_cached_k_descale`、`_cached_v_descale`。
- 当 `descale_shape` 或 scale tensor id 变化时刷新 expanded view；否则复用已有 view。
- 保持 H6.1c 的 AITER backend、`supports_sink=False`、不传 `sinks/output_scale`；不改变 attention 数学或 KV cache。

构建与安装：

- build/install：`/public/home/tangyu408/testdata/goal_runs/20260711_h11_1_aiter_descale_cache_build`
- 安装态验证：`h11_1_marker=True`，`supports_sink=False`，`passes_sinks=False`，`passes_output_scale=False`
- 因收益为噪声级，源码已回退到工作基线并重新强制重编安装：`/public/home/tangyu408/testdata/goal_runs/20260711_restore_after_h11_1_descale_cache`
- 恢复态验证：`h11_1_marker=False`，`supports_sink_false=True`，`passes_sinks=False`，`passes_output_scale=False`

小样本对照：

- 对照 baseline：`/public/home/tangyu408/testdata/goal_runs/20260711_010847_pre_padfill_baseline_16_32K_n4_noproxy`
- H11.1 candidate：`/public/home/tangyu408/testdata/goal_runs/20260711_h11_1_aiter_descale_cache_16_32K_n4_fixed`
- 命令：固定 `start_vllm.sh`，固定 `run_throughput.sh 16-32K 4`，仅加 no-proxy 修正本机 API 路由。
- 有效性：candidate 主体 benchmark `status=0`，`bench_elapsed_sec=227`，成功后额外等待 `373` 秒，满足小样本至少 10 分钟等待约束。

| 指标 | baseline | H11.1 candidate | 变化 |
| --- | ---: | ---: | ---: |
| completed / failed | `4 / 0` | `4 / 0` | 无变化 |
| output throughput | `8.880139 tok/s` | `8.881475 tok/s` | `+0.015%` |
| duration | `156.3039 s` | `156.2803 s` | `-0.0235 s` |
| total output tokens | `1388` | `1388` | 无变化 |
| mean TTFT | `14521.666 ms` | `14517.670 ms` | `-3.996 ms` |
| mean TPOT | `70.096 ms` | `70.121 ms` | `+0.025 ms` |

结论：

- H11.1 只减少 wrapper 侧 view 构造，未降低 attention kernel 主体时间，吞吐变化为噪声级。
- 不进入 full `all` 或 accuracy；该候选不作为后续基础保留。
- AITER attention2d 后续应转为 kernel/shape 级归因，例如 q/k/v shape、block table、shared-memory bank conflict 和真实 kernel 参数，而不是继续做 Python wrapper 小对象缓存。

## R16 H4.5 Qwen3.5 GDN core output `empty` allocation

实验时间：`2026-07-11`。

触发条件：R11 中 `FillFunctor<int>` 和 elementwise/fill 类 kernel 占比明显，且实际 Qwen3.5 命中 `vllm/model_executor/models/qwen3_5.py`。该文件的 `Qwen3_5GatedDeltaNet.forward()` 在每个 GDN 层分配：

```python
core_attn_out = torch.zeros((num_tokens, num_v_heads, head_v_dim), ...)
```

源码中原注释提示不要使用 `torch.empty`，但 `_forward_core()` 的主分支和 packed decode 分支看起来都会写入 `core_attn_out[:num_actual_tokens]`。因此做最小源码验证：把该分配改为 `torch.empty(...)`，尝试消除中间张量清零。

构建与安装：

- build/install：`/public/home/tangyu408/testdata/goal_runs/20260711_h4_5_qwen35_gdn_empty_core_out_build`
- 安装态验证：`h4_5_marker=True`，`no_old_zero_comment=True`
- 因正确性失败，源码已回退到工作基线并重新强制重编安装：`/public/home/tangyu408/testdata/goal_runs/20260711_restore_after_h4_5_qwen35_gdn_empty_core_out`
- 恢复态验证：`h4_5_marker=False`，`uses_zeros=True`，`old_zero_comment=True`

小样本对照：

- 对照 baseline：`/public/home/tangyu408/testdata/goal_runs/20260711_010847_pre_padfill_baseline_16_32K_n4_noproxy`
- H4.5 candidate：`/public/home/tangyu408/testdata/goal_runs/20260711_h4_5_qwen35_gdn_empty_core_out_16_32K_n4_fixed`
- 命令：固定 `start_vllm.sh`，固定 `run_throughput.sh 16-32K 4`，仅加 no-proxy 修正本机 API 路由。

| 指标 | baseline | H4.5 candidate | 变化 |
| --- | ---: | ---: | ---: |
| completed / failed | `4 / 0` | `4 / 0` | 无请求失败 |
| output throughput | `8.880139 tok/s` | `6.402967 tok/s` | `-27.896%` |
| duration | `156.3039 s` | `106.5131 s` | `-49.7907 s` |
| total output tokens | `1388` | `682` | `-706` |
| output_lens | `[23, 265, 76, 1024]` | `[23, 289, 76, 294]` | 第 2/4 条变化，长输出提前停止 |
| mean TTFT | `14521.666 ms` | `14510.384 ms` | `-11.282 ms` |
| mean TPOT | `70.096 ms` | `70.024 ms` | `-0.071 ms` |

结论：

- H4.5 正确性失败。它降低 E2E 主要来自输出提前停止，并不是有效吞吐提升。
- 原注释“不要使用 `torch.empty`”在 Qwen3.5 固定负载下被验证为必要约束；`gdn_attention_core` 的有效写入不能作为省略初始化的充分条件。
- 该候选不进入 full `all` 或 accuracy，必须回滚。后续 GDN fill 优化不能再通过把 core output buffer 直接改成 uninitialized allocation 实现。

## R17 H10.2 unquantized GEMM dispatch cache

实验时间：`2026-07-11`。

触发条件：R11 显示 `linear_gemm` 占总 kernel 时间 `53.221%`。H10.1 已证明当前安装的 ROCm 扩展没有注册 `wvSplitK`，不能直接打开 existing skinny GEMM gate；在尚未完成 shape/source 归因前，先验证 linear 热路径上是否存在可见 Python dispatch 开销。

源码变更：

- `vllm/model_executor/layers/linear.py`：在 `UnquantizedLinearMethod.__init__()` 中缓存一次 `dispatch_unquantized_gemm()` 返回的函数句柄。
- `UnquantizedLinearMethod.apply()` 保持 `linear_batch_invariant` 分支原样；普通路径从每次调用 `dispatch_unquantized_gemm()(...)` 改为 `self._unquantized_gemm(...)`。
- 不改变权重、tensor shape、GEMM backend gate、batch scheduler、serve 参数或输出语义。

构建与安装：

- build/install：`/public/home/tangyu408/testdata/goal_runs/20260711_h10_2_linear_dispatch_cache_build`
- 构建命令包含 `python3 setup.py build_py --force` 和 `python3 setup.py bdist_wheel`。
- wheel：`/public/home/tangyu408/vllm_cscc/dist/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl`
- wheel sha256：`7ac51ceb89fbcfc8865af840ec10237c57bd31949122aaef635e5a9c9e0bd80a`
- 从 `/tmp` 导入 installed package 的验证：`vllm_path=/usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/linear.py`，`dispatch_cached=True`，`init_cached=True`。
- 因收益为噪声级，源码已回退到工作基线并重新强制重编安装：`/public/home/tangyu408/testdata/goal_runs/20260711_restore_after_h10_2_linear_dispatch_cache`
- 恢复 wheel sha256：`50c7d4c57aa8f8a58695d1b174a06c7036cceddbfbe0f6550de79cdbb2af30e7`
- 恢复态验证：`linear_path=/usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/linear.py`，`h10_2_marker=False`，`dispatch_call=True`

服务和 API：

- 运行路径：`/public/home/tangyu408/testdata/goal_runs/20260711_h10_2_linear_dispatch_cache_16_32K_n4_fixed`
- 启动：固定 `start_vllm.sh`，未传额外 serve 参数。
- API ready：`models_ready_after_seconds=205`
- `/v1/models` 返回 `Qwen3.5-27B`；短问答 `finish_reason=stop`。

小样本对照：

- 对照 baseline：`/public/home/tangyu408/testdata/goal_runs/20260711_010847_pre_padfill_baseline_16_32K_n4_noproxy`
- H10.2 candidate：`/public/home/tangyu408/testdata/goal_runs/20260711_h10_2_linear_dispatch_cache_16_32K_n4_fixed`
- 命令：固定 `start_vllm.sh`，固定 `run_throughput.sh 16-32K 4`，仅加 no-proxy 修正本机 API 路由。
- 有效性：candidate 主体 benchmark `status=0`，`elapsed_before_wait=220`，成功后额外等待 `380` 秒，`elapsed_total=600`，满足小样本至少 10 分钟等待约束。

| 指标 | baseline | H10.2 candidate | 变化 |
| --- | ---: | ---: | ---: |
| completed / failed | `4 / 0` | `4 / 0` | 无变化 |
| output throughput | `8.880139 tok/s` | `8.883034 tok/s` | `+0.033%` |
| total output tokens | `1388` | `1388` | 无变化 |
| output_lens | `[23, 265, 76, 1024]` | `[23, 265, 76, 1024]` | 无变化 |
| mean TTFT | `14521.666 ms` | `14514.580 ms` | `-7.086 ms` |
| p99 TTFT | `15247.480 ms` | `15237.987 ms` | `-9.494 ms` |
| mean TPOT | `70.096 ms` | `70.059 ms` | `-0.037 ms` |
| p99 TPOT | `70.397 ms` | `70.385 ms` | `-0.012 ms` |
| mean E2EL | `38863.667 ms` | `38851.128 ms` | `-12.539 ms` |

结论：

- H10.2 没有改变输出 token 数或请求成功率，但吞吐提升只有 `+0.033%`，属于噪声级。
- 该结果排除了“缓存 Python 侧 unquantized GEMM dispatch 可以显著改善单请求吞吐”的假设。
- 后续 H10 必须继续做真实 GEMM shape/source/backend 归因，而不是继续围绕 Python dispatch 小开销做补丁。

## R18 H10.3 Linear/GEMM source attribution

实验时间：`2026-07-11`。

触发条件：R11 显示 `linear_gemm` 占总 kernel 时间 `53.221%`，H10.1 排除了直接启用 `wvSplitK`，H10.2 排除了 Python dispatch cache。需要把 GEMM 热点绑定到具体 Linear 层形状，避免继续围绕错误的小 GEMM 或 wrapper 开销做补丁。

AITER GEMM 可用性探测：

- `aiter.ops.triton.gemm_a16w16` 可以 import。
- `torch.ops._rocm_C.wvSplitK` 和 `torch.ops._rocm_C.LLMM1` 均未注册。
- 对 Qwen3.5 相关形状做微基准时，`gemm_a16w16` 在当前 DTK/ROCm 环境报错：

```text
RuntimeError: cannot get address for 'hipGetDeviceProperties' from libamdhip64.so
```

探测形状包括：

- GDN `in_proj_ba`：`(96, 5120)`
- attention/GDN QKV：`(8192, 5120)`、`(14336, 5120)`、`(16384, 5120)`
- MLP gate/up：`(34816, 5120)`
- MLP down：`(5120, 17408)`

结论：当前不能把 AITER Triton GEMM 作为源码候选直接接入，也不能解除 skinny GEMM gate。

静态 Linear 层归因：

- 采用构造期日志记录 `LinearBase` 子类的 prefix、class、输入/输出 shape 和 partition 信息。
- 首次 forward-hook 方案因 TorchDynamo fullgraph 拒绝调用 `torch.compiler.disable()` 函数而失败；后续改为构造期静态记录，服务可启动并完成 API 检查。
- 运行中解析到 `462` 个 Linear 层。服务退出时 `atexit` 被无层进程覆盖成空文件；后续若重做该插桩，必须使用 PID 分文件，或只在 `rows > 0` 时覆盖目标文件。

按每 token FLOP 估算的主要 Linear 来源：

| 来源 | class | count | shape `(out, in)` | 估算 GFLOP/token | 结论 |
| --- | --- | ---: | --- | ---: | --- |
| MLP `gate_up_proj` | `MergedColumnParallelLinear` | `64` | `(34816, 5120)` | `22.817` | 第一优先级 |
| MLP `down_proj` | `RowParallelLinear` | `64` | `(5120, 17408)` | `11.409` | 第一优先级 |
| GDN `in_proj_qkvz` | `MergedColumnParallelLinear` | `48` | `(16384, 5120)` | `8.053` | 第二优先级 |
| GDN `out_proj` | `RowParallelLinear` | `48` | `(5120, 6144)` | `3.020` | 第二优先级 |
| full-attention `qkv_proj` | `QKVParallelLinear` | `16` | `(14336, 5120)` | `2.349` | 第三优先级 |
| full-attention `o_proj` | `RowParallelLinear` | `16` | `(5120, 6144)` | `1.007` | 第三优先级 |
| GDN `in_proj_ba` | `ColumnParallelLinear` | `48` | `(96, 5120)` | `0.047` | 不是主目标 |

附加源码观察：

- Qwen3Next MLP 命中 `Qwen2MoeMLP`。
- `gate_up_proj -> SiluAndMul -> down_proj` 已经使用融合激活路径；简单的 SiLU/mul Python 侧融合不是可用突破点。

结论：

- GEMM 优化不能再优先追 `in_proj_ba` 这类小 GEMM；其 FLOP 权重远低于 MLP 与 GDN 大投影。
- 下一步 H10 若继续，应面向 MLP gate/up、MLP down、GDN `qkvz/out` 和 attention `qkv/o` 的实际 backend/kernel 做归因。
- 在当前环境下，AITER Triton GEMM 与现有 ROCm skinny GEMM 都不能直接作为候选实施。

## R19 H11.2 AITER KV cache view cache

实验时间：`2026-07-11`。

触发条件：R11 显示 `kernel_unified_attention_2d` 占 `26.707%`，H11.1 已排除 descale expand view cache。继续验证 attention wrapper 中反复执行 `kv_cache.unbind(0)` 和 FP8 `view` 是否有可见开销。

源码变更：

- `vllm/v1/attention/backends/rocm_aiter_unified_attn.py`：在 `RocmAiterUnifiedAttentionImpl` 中新增 `_get_kv_cache_views()`，按 `id(kv_cache)` 缓存每层 key/value cache view。
- `forward()`、`do_kv_cache_update()`、`do_rope_and_kv_cache_update()` 改用该 helper。
- 保持 H6.1c 的 AITER backend、`supports_sink=False`、不传 `sinks/output_scale`；不改变 KV cache 内容、attention 数学、serve 参数或测试脚本。

构建与安装：

- candidate build/install：`/public/home/tangyu408/testdata/goal_runs/20260711_h11_2_kv_cache_view_build`
- 构建命令包含 `python3 setup.py build_py --force` 和 `python3 setup.py bdist_wheel`。
- candidate wheel sha256：`06711d8dba1636cb1a2c25557af1954570c5551b3aeb9c8a7084c75de59565a0`
- 从 `/tmp` 导入 installed package 的验证：`h11_2_marker=True`，`linear_static_marker=False`，`h10_2_marker=False`。
- 因收益为负，源码已回退到工作基线并重新强制重编安装：`/public/home/tangyu408/testdata/goal_runs/20260711_restore_after_h11_2_build`
- 恢复 wheel sha256：`9466a8822fc5fad2c9e8bad0368784358cac2ec4d1aea641f0208da8e12973ba`
- 恢复态验证：`h11_2_marker=False`，`h6_1c_sink_marker=True`，`linear_static_marker=False`，`h10_2_marker=False`

服务和 API：

- 运行路径：`/public/home/tangyu408/testdata/goal_runs/20260711_h11_2_kv_cache_view_start`
- 启动：固定 `start_vllm.sh`，未传额外 serve 参数。
- API ready：`models_ready_after_seconds=159`
- `/v1/models` 返回 `Qwen3.5-27B`；短问答 `finish_reason=stop`。

小样本对照：

- 对照 baseline：`/public/home/tangyu408/testdata/goal_runs/20260711_010847_pre_padfill_baseline_16_32K_n4_noproxy`
- H11.2 candidate：`/public/home/tangyu408/testdata/goal_runs/20260711_h11_2_kv_cache_view_16_32K_n4`
- 命令：固定 `start_vllm.sh`，固定 `run_throughput.sh 16-32K 4`，仅加 no-proxy 修正本机 API 路由。
- 有效性：candidate 主体 benchmark `status=0`，`elapsed_before_wait=220`，成功后额外等待 `380` 秒，`elapsed_total=600`，满足小样本至少 10 分钟等待约束。

| 指标 | baseline | H11.2 candidate | 变化 |
| --- | ---: | ---: | ---: |
| completed / failed | `4 / 0` | `4 / 0` | 无变化 |
| output throughput | `8.880139 tok/s` | `8.875221 tok/s` | `-0.055%` |
| total token throughput | `563.076320 tok/s` | `562.764444 tok/s` | `-0.055%` |
| total input tokens | `86623` | `86623` | 无变化 |
| total output tokens | `1388` | `1388` | 无变化 |
| output_lens | `[23, 265, 76, 1024]` | `[23, 265, 76, 1024]` | 无变化 |
| mean TTFT | `14521.666 ms` | `14529.182 ms` | `+7.516 ms` |
| p99 TTFT | `15247.480 ms` | `15244.904 ms` | `-2.576 ms` |
| mean TPOT | `70.096 ms` | `70.174 ms` | `+0.078 ms` |
| p99 TPOT | `70.397 ms` | `70.437 ms` | `+0.040 ms` |
| mean E2EL | `38863.667 ms` | `38885.548 ms` | `+21.880 ms` |

结论：

- H11.2 没有改变输出 token 数或请求成功率，但吞吐下降 `-0.055%`，属于无收益候选。
- 该结果进一步排除 attention wrapper 小对象/view cache 方向；H11 后续应转向 AITER `kernel_unified_attention_2d` 的 kernel 参数、shape、block table 访问和 shared-memory bank conflict。
- H11.2 不进入 full `all` 或 accuracy，不作为后续基础保留。

## R20 H10.4 gfx936 LLMM1 exact-shape gate

实验时间：`2026-07-11`。

候选与瓶颈归因：

- 候选 ID：`H10.4`。
- R11/R18 已将 Linear/GEMM 定位为主要热点，并确认单 token decode 会重复命中 Qwen3.5-27B 的大投影。H10.4 因此面向 gfx936 单 token BF16 GEMM，而不是 Python dispatch 开销。
- `setup.py` 恢复构建仓库已有的 `vllm._rocm_C` 扩展；`vllm/platforms/rocm.py` 增加精确的 `on_gfx936()` 判定。
- `vllm/model_executor/layers/utils.py` 只在以下条件全部满足时调用 `ops.LLMM1(weight, x_view, 8)`：`on_gfx936()`、`n == 1`、`k == 5120`、`m in (96, 14336, 16384, 34816)`、输入和权重均为 BF16、`bias is None`、权重 contiguous、输入最后一维 stride 为 `1`。任一条件不满足时保留原 GEMM dispatch。
- 该 exact gate 排除了 `k=6144`、LM head、非单 token、非 BF16、有 bias、非连续权重以及非 gfx936 设备；未修改模型权重、scheduler、固定 serve 参数或吞吐脚本。

构建与路径证据：

- build/install：`/public/home/tangyu408/testdata/goal_runs/20260711_h10_4_rocm_ext_llmm1_gate_build`
- wheel sha256：`514ae2e1cb28505170277f99529ca3e9e480aa4b4ec6655cd97c760366f37d0a`
- 安装态验证：`gfx936=True`，`vllm._rocm_C.abi3.so` 已安装，`wvSplitK=True`、`LLMM1=True`，源码 exact gate marker 为 `True`。
- serve：`/public/home/tangyu408/testdata/goal_runs/20260711_h10_4_llmm1_gate_serve`
- 小样本 throughput：`/public/home/tangyu408/testdata/goal_runs/20260711_h10_4_llmm1_gate_all3_10m`
- 固定小样本命令：`./run_throughput.sh all 3`；主体运行 `298s`，随后等待 `302s`，总时长 `600s`。这是候选筛选口径，不是最终可计分的 full `all`。

小样本性能对比：

| 档位 | H6.1c output throughput | H10.4 output throughput | 相对 H6.1c | H10.4 mean TTFT | H10.4 mean TPOT | completed / failed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `4-8K` | `9.559390354` | `11.127260615` | `+16.4014%` | `2710.067 ms` | `53.4747 ms` | `3 / 0` |
| `8-16K` | `9.745495152` | `11.317036241` | `+16.1258%` | `6648.893 ms` | `54.2126 ms` | `3 / 0` |
| `16-32K` | `5.274733352` | `5.657544798` | `+7.2575%` | `14257.140 ms` | `55.2626 ms` | `3 / 0` |

三档相对提升按 `20/50/30` 加权为 `+13.5204%`。

输出审计：

| 档位 | H6.1c output_lens | H10.4 output_lens | 逐请求文本相等 |
| --- | --- | --- | --- |
| `4-8K` | `[88, 62, 91]` | `[88, 62, 91]` | `[true, true, true]` |
| `8-16K` | `[100, 12, 498]` | `[95, 12, 497]` | `[false, true, false]` |
| `16-32K` | `[23, 265, 76]` | `[23, 259, 76]` | `[true, false, true]` |

结论：H10.4 显著降低了小样本 mean TPOT，但相对 H6.1c 的 8-16K 与 16-32K 输出已发生变化，且加权提升未达到新增 `+20%`。该 all3 结果只能作为继续组合优化和执行 full `all`/accuracy 的晋级信号，不能写入最终可计分结论。

## R21 H11.3 GQA6 prefill non-overlap attention2d

实验时间：`2026-07-11`。

候选与源码变更：

- 候选 ID：`H11.3`；候选基线包含已验证可运行的 H10.4 exact gate。
- AITER 原 `BLOCK_M=16` 的 GQA6 行映射会跨越相邻 query block。新增 `vllm/v1/attention/ops/rocm_aiter_unified_attention_gqa6.py`，把每个 KV head 对应的 `6` 个 query heads 分成 `3` 组、每组 `2` heads；每个 program 精确覆盖 `8 tokens x 2 heads = 16` 行，从映射上消除相邻 query block 的行重叠。
- 新 kernel 保留原 online-softmax、KV cache block 选择和 causal mask；第三个 launch grid 轴只表示 GQA head group，不是 segmented 3D decode，也不引入 segment 临时 buffer 或归约 kernel。
- `rocm_aiter_unified_attn.py` 的构造期 gate 精确限制为 gfx936、`24` query heads、`4` KV heads、head size `256`、`kv_cache_dtype=auto`、无 ALiBi/sliding window/logits soft cap/sinks、decoder attention。调用期还要求 `max_seqlen_q > 1`、q/k/v/output 均为 BF16、query 尾部 shape 为 `(24, 256)`、K/V cache 尾部 shape 为 `(4, 256)`；decode 与所有非目标调用继续走原 AITER path。

构建与路径证据：

- 清洁 build/install：`/public/home/tangyu408/testdata/goal_runs/20260711_h11_3_gqa6_prefill_pure_build`
- wheel sha256：`95b2aa5f43eb40ffec849ea2550e611e7534205634bbd2bf13fde79a2d371cf7`
- 安装态验证：H10.4 LLMM1 gate、H11.3 wrapper 与 `rocm_aiter_unified_attention_gqa6.py` 均来自 installed wheel。
- serve：`/public/home/tangyu408/testdata/goal_runs/20260711_h11_3_gqa6_prefill_serve`
- server log 明确记录：`H11.3 enabled for gfx936 BF16 head256 GQA6 prefill; non-target and decode calls keep the original AITER path`。
- 小样本 throughput：`/public/home/tangyu408/testdata/goal_runs/20260711_h11_3_gqa6_prefill_all3_10m`
- 固定小样本命令：`./run_throughput.sh all 3`；主体运行 `279s`，随后等待 `321s`，总时长补足到 `600s`。这是候选筛选口径，不是最终可计分的 full `all`。

小样本性能对比：

| 档位 | H6.1c output throughput | H10.4 output throughput | H11.3 output throughput | H11.3 相对 H6.1c | H11.3 mean TTFT | H11.3 mean TPOT | completed / failed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4-8K` | `9.559390354` | `11.127260615` | `11.624868885` | `+21.6068%` | `2399.671 ms` | `53.4918 ms` | `3 / 0` |
| `8-16K` | `9.745495152` | `11.317036241` | `11.987605740` | `+23.0066%` | `5688.001 ms` | `54.0973 ms` | `3 / 0` |
| `16-32K` | `5.274733352` | `5.657544798` | `6.379901432` | `+20.9521%` | `11869.088 ms` | `55.2292 ms` | `3 / 0` |

三档相对 H6.1c 的提升按 `20/50/30` 加权为 `+22.1103%`；相对直接工作基线 H10.4 的三档提升按相同权重加权为 `+7.6875%`。

正确性快检：

- H11.3 的三档 `output_lens` 分别为 `[88, 62, 91]`、`[95, 12, 497]`、`[23, 259, 76]`，与 H10.4 完全相同。
- 三档共 `9` 条 `generated_texts` 与 H10.4 逐请求文本完全一致。
- 该一致性只说明 H11.3 相对 H10.4 未引入新的可见输出变化；H10.4 相对 H6.1c 在 8-16K/16-32K 已有变化，因此不能据此宣称通过最终精度门槛。

结论：H10.4 + H11.3 的 all3 小样本首次给出相对 H6.1c 加权超过 `+20%` 的晋级信号，同时 H11.3 增量保持 H10.4 的逐条输出。后续三次 full `all`、accuracy、SLA 与输出审计均已完成，最终闭环见 R23。

## R22 H10.5 gfx936 wvSplitK correctness probe

实验时间：`2026-07-11`。

候选变更与失败证据：

- 候选 ID：`H10.5`。
- 候选尝试在 `csrc/rocm/skinny_gemms.cu` 中将普通 `wvSplitK` 编译路径扩展到 gfx936，并为 gfx936 的 FP16 dot-product 指令形式增加条件分支。
- candidate build/install：`/public/home/tangyu408/testdata/goal_runs/20260711_h11_3_gqa6_prefill_h10_5_ext_build`
- candidate wheel sha256：`330c77c12ad26f7acaecb1c6ca8566898620701575c7cbe9f63601e3ff8c237d`
- 对 `96x5120` 目标形状做 direct correctness 对照时，结果为 `max_abs=4.4443`、`mean_abs=1.1473`。该误差不是可接受的 BF16 舍入噪声，正确性 gate 直接失败。

回退与恢复证据：

- H10.5 的 gfx936 `wvSplitK` 源码变更已回退；未进入服务吞吐、full `all` 或 accuracy。
- 回退后执行清洁 wheel 重建并安装：`/public/home/tangyu408/testdata/goal_runs/20260711_h11_3_gqa6_prefill_pure_build`
- 恢复 wheel sha256：`95b2aa5f43eb40ffec849ea2550e611e7534205634bbd2bf13fde79a2d371cf7`
- 恢复 wheel 保留 H10.4 LLMM1 exact gate 与 H11.3 GQA6 prefill kernel，但不包含 H10.5 的 gfx936 `wvSplitK` 实现变更，避免复用旧 native build 产物。

结论：H10.5 在最小直接数值门槛即失败，不能以其任何潜在性能作为优化证据。gfx936 `wvSplitK` 只有在修复 kernel 数学并覆盖目标 dtype/shape 的 same-input 对照后才能重新进入候选；当前工作栈以回退后的清洁 wheel 为准。

## R23 H10.4 和 H11.3 full all accuracy 与重复性闭环

实验时间：2026-07-11。

最终结论：H6.1c + H4.1/H4.2 + D1 + H10.4 + H11.3 已完成三次固定
full all、OpenCompass accuracy、TTFT/TPOT SLA 和独立 API 输出审计。三次
full 相对 H6.1c 的 20/50/30 加权提升均值为 +23.7651%，最差一轮仍为
+23.5860%；最终 accuracy 系数为 1.0。该栈替代 H6.1c，成为当前可计分最佳。

### 源码、构建和服务证据

累计源码栈：

- H6.1c：ROCm AITER Unified Attention backend gating 和 wrapper 签名适配。
- H4.1/H4.2：GDN prefill metadata/no-initial-state 快路径。
- D1：GDN decode packed validate 跳过。
- H10.4：打包已有的 _rocm_C，并对 gfx936 BF16 单 token 的
  m={96,14336,16384,34816}, k=5120 使用 LLMM1(rows=8) exact gate。
- H11.3：gfx936、head256、GQA6 prefill non-overlap attention2d kernel。
- 不包含已回退的 H10.5 gfx936 wvSplitK 变更。

权威产物：

- clean build/install：
  /public/home/tangyu408/testdata/goal_runs/20260711_h11_3_gqa6_prefill_pure_build
- wheel SHA256：
  95b2aa5f43eb40ffec849ea2550e611e7534205634bbd2bf13fde79a2d371cf7
- tracked source diff：上述目录的 `source_final_tracked.diff`，SHA256
  `96dac45a018d670d4dc9482b242f79400ecb9f70cf44cca761aaa0343dbd3a08`
- H11.3 新文件补充 diff：上述目录的
  `source_untracked_h11_3_gqa6_repo_relative.diff`，其 patch 路径为仓库相对路径，
  SHA256 `e280ccfb4ba0165195bf91baa35f77e4b855f4ac1a9a33a92746c23aa82bf325`
- 两份最终源码 diff 的可移植校验清单：上述目录的
  `source_final_evidence.sha256`；其条目均为文件名，可在该目录执行
  `sha256sum -c source_final_evidence.sha256`，manifest 自身 SHA256 为
  `d048f56d68d88dd0f3ed81214385b9db497d6d38214afe9eca7fd0fe914196f7`。
  仓库相对的新文件 diff 已在空临时目录应用并与实测源码逐字节比对通过。
- H11.3 源码与 installed wheel 文件共同 SHA256：
  2d27f778c5295da1d77a2452e1de91f3190ee883d8f40191a383b7ac05bc2cbf
- serve：
  /public/home/tangyu408/testdata/goal_runs/20260711_h11_3_gqa6_prefill_serve
- server log 同时确认 ROCM_AITER_UNIFIED_ATTN 和
  H11.3 enabled for gfx936 BF16 head256 GQA6 prefill。

### 三次固定 full all

三次均使用未修改的 run_throughput.sh all，不传第二参数；默认
MAX_CONCURRENCY=1、REQUEST_RATE=1、CUSTOM_OUTPUT_LEN=1024、NUM_WARMUPS=2，
并显式绕过本机代理。每档 50 条，三次合计 450/450 completed、failed=0。
固定脚本 SHA256 分别为：run_throughput.sh
adf0cf91266745b37df916926c7d495ec79f00a11be653c219d1d5df4d93c681，
run_accuracy.sh
2e641672a45ac96318c2118df8df4dae2babf87c16afd49cbe4b037ff9beed4e，
start_vllm.sh
7c3e8c5ecdf02109e02af8c3b5ba05050b26339c7f50869b5288eea359364fad。

| run | 4-8K tok/s | 8-16K tok/s | 16-32K tok/s | 相对 H6.1c 加权 | 相对 official 加权 | 全局 TPOT P99 | 公式分，K=1 | wrapper elapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| repeat1 | 16.013011890 | 13.087815856 | 8.974868230 | +23.586039% | +50.457339% | 55.663000 ms | 78.970059 | 3168 s |
| repeat2 | 16.006059993 | 13.145676012 | 8.975153785 | +23.850812% | +50.775888% | 55.670008 ms | 79.056908 | 3169 s |
| repeat3 | 16.009989712 | 13.145788851 | 8.975399178 | +23.858493% | +50.784332% | 55.649035 ms | 79.059617 | 3162 s |

证据目录：

- repeat1：
  /public/home/tangyu408/testdata/goal_runs/20260711_h11_3_gqa6_prefill_full_all
- repeat2：
  /public/home/tangyu408/testdata/goal_runs/20260711_h11_3_gqa6_prefill_full_all_repeat2
- repeat3：
  /public/home/tangyu408/testdata/goal_runs/20260711_h11_3_gqa6_prefill_full_all_repeat3

每个目录的 status.txt 均为 0。每档 benchmark 主体均超过 600 秒。

以三次结果做统计；95% CI 使用 n=3、df=2 的双侧 t 区间：

| 档位 | throughput mean | sample std | 95% CI half-width | mean 相对 H6.1c |
| --- | ---: | ---: | ---: | ---: |
| 4-8K | 16.009687198 | 0.003485807 | ±0.008659225 | +25.255721% |
| 8-16K | 13.126426906 | 0.033438198 | ±0.083065090 | +24.544992% |
| 16-32K | 8.975140398 | 0.000265727 | ±0.000660103 | +21.471583% |

- 相对 H6.1c 的加权提升：mean +23.765115%，sample std 0.155132 个百分点，
  95% CI half-width ±0.385368 个百分点，最差轮 +23.586039%。
- 相对 official baseline 的加权提升均值：+50.672520%。
- 吞吐公式分：mean 79.028861，sample std 0.050943，
  95% CI half-width ±0.126548。
- H6.1c 公式分为 69.566475；候选均值相对 H6.1c 公式分再提升
  +13.601934%，超过目标要求的 +10%。

### SLA

official baseline 的全局 TPOT P99 为 71.803421 ms，SLA 上限为
107.705132 ms。三次候选全局 TPOT P99 的最大值仅 55.670008 ms。

| 档位 | official TTFT P99 | 1.5x 上限 | 三次候选最大 TTFT P99 | 结论 |
| --- | ---: | ---: | ---: | --- |
| 4-8K | 4792.478682 ms | 7188.718023 ms | 3030.067658 ms | 通过 |
| 8-16K | 24886.185346 ms | 37329.278019 ms | 8865.360782 ms | 通过 |
| 16-32K | 28740.837379 ms | 43111.256069 ms | 12641.404063 ms | 通过 |

### Accuracy 与精度系数

固定 run_accuracy.sh all 返回 0，证据目录为：

- run：
  /public/home/tangyu408/testdata/goal_runs/20260711_h11_3_gqa6_prefill_full_accuracy_all
- OpenCompass output：
  /public/home/tangyu408/testdata/accuracy_debug/output/local_accuracy_qwen35/20260711_210118

| 数据集 | official baseline | candidate | 相对下降 Delta | k |
| --- | ---: | ---: | ---: | ---: |
| hotpotqa | 77.96 | 77.96 | 0.00000% | 1.00 |
| gov_report | 32.961006236 | 32.938380849 | 0.068643% | 1.00 |
| retrieval_multi_point | 100.00 | 100.00 | 0.00000% | 1.00 |
| aggregation_keyword_aggregation | 100.00 | 100.00 | 0.00000% | 1.00 |

官方规则规定单类 Delta <= 1% 时 k=1.00，因此四类 k 均为 1.00，最终
accuracy 系数 K=1.00，无精度扣分。gov_report 的细微差异来自实际生成文本，
不是数据错位、汇总或四舍五入错误。

### 输出与停止原因审计

- repeat2、repeat3 的三档 output_lens 和 generated_texts 与 repeat1
  逐请求完全相同；三次 full 的 150 个固定输入输出具有完全重复性。
- repeat1 相对 H6.1c 的 output_lens 相同 125/150，generated_texts 相同
  114/150；总输出 token 为 36803 对 37271。该输出变化由上述
  OpenCompass K=1.00 结果按 C4 完成豁免闭环。
- 独立 API 审计选取每档前三条和首个 1024-token 样本，共 12 条；路径：
  /public/home/tangyu408/testdata/goal_runs/20260711_h11_3_gqa6_prefill_api_output_audit_12_10m
- 审计 wrapper 的 status.txt 为 0，elapsed_total.txt 为 600，满足本轮最短测试时长。
- 审计请求 12/12 返回 token IDs；finish_reason 为 10 条 stop、2 条 length，
  stop_reason 均为 null，未出现异常停止原因。
- 独立重放与 full repeat1 的文本相同 5/12、长度相同 7/12。该结果只说明
  改变请求序列后未能逐 token 复现，原因未单独定位，可能涉及调度或数值
  非确定性；不能把单次 API 重放误写成逐 token 确定性。固定 full 的三次
  同序列重复性和 OpenCompass K=1.00 才是最终正确性依据。

### 阶段结论

H10.4 主要降低 decode TPOT，H11.3 主要降低 prefill TTFT；组合栈在三次
full 中稳定复现，SLA、完成率和 accuracy 系数均通过。当前应冻结 wheel、
源码及证据，不继续扩大 LLMM1 shape gate 或 H11.3 kernel；后续新优化必须
以本 R23 的三次汇总值作为新的可计分基线。

## R24 H11.4 adaptive GQA6 prefill 和 H10.7 gfx936 strided LLMM1 闭环

实验时间：2026-07-12。

最终结论：以 R23 为直接增量基线，H11.4 + H10.7 已完成源码实现、构建安装、
数值验证、两次有效 10min 总窗口筛选、三次固定 full all、SLA 和固定
`run_accuracy.sh all` 闭环。三次 full 共 450/450 请求成功，综合分分别为
`85.711178 / 85.704307 / 85.706985`，均值 `85.707490`；相对 R23 三次
均值的 20/50/30 加权吞吐提升分别为
`+20.595366% / +20.573414% / +20.581227%`，均值 `+20.583336%`。
该栈同时命中“综合分达到 85”与“相对当前最佳性能再次提升超过 20%”两项
终止条件，替代 R23 成为新的可计分最佳。

### 累计源码栈与本轮新增机制

累计栈保留 H6.1c、H4.1/H4.2、D1，并以 R23 的 H10.4/H11.3 为直接工作
基线。本轮新增：

- H11.4：在
  `vllm/v1/attention/ops/rocm_aiter_unified_attention_gqa6.py` 中对长
  prefill 增加自适应 kernel 配置；`max_seqlen_q >= 128` 时使用
  `num_warps=2`、`num_stages=1`、`waves_per_eu=1`、
  `matrix_instr_nonkdim=16`、`kpack=2`，短 prefill 保留 H11.3 配置。
  gfx936、BF16、head256、GQA6、prefill-only 等精确 gate 不变，decode 和
  非目标 shape 继续走原 AITER 路径。
- H10.7：在 `csrc/rocm/skinny_gemms.cu` 新增
  `LLGemm1_strided_kernel` 和公开绑定 `LLMM1Strided`。gfx936 BF16
  内层把 packed BF16 转为 `float2`，以两路 FP32 `fmaf` 累加后写回
  BF16；生产配置固定为 `rows=4, threads=320`。
- `csrc/rocm/ops.h`、`csrc/rocm/torch_bindings.cpp`、
  `vllm/_custom_ops.py` 增加声明、绑定和 Python wrapper；
  `vllm/model_executor/layers/utils.py` 只在 gfx936、BF16、`n=1`、
  `k=5120`、无 bias 且 `m in {96,14336,16384,34816}` 时启用。
  `m=96` 使用原 LLMM1 `rows=4`，其余三个大投影使用
  `LLMM1Strided(4,320)`；条件不满足时回退原 GEMM。
- C++ 接口显式检查 rank、CUDA/同设备、连续性、dtype/K 一致性、M/K
  合法范围、16-byte 对齐和唯一受支持配置。

H10.7 的 shape gate 是精确的，但其 FP32 累加结果不是与旧 LLMM1
`rows=8` 逐位相同；正确性依据是 BF16 数值容差、固定 workload 重复性和
最终 accuracy，而不是 bitwise-equal 声明。

### 构建、安装和可复现证据

- 最终 build/install：
  `/public/home/tangyu408/testdata/goal_runs/20260712_h11_4_h10_7_final_build`
- wheel：
  `/public/home/tangyu408/vllm_cscc/dist/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl`
- wheel SHA256：
  `399c7a847c8607269b41d77f189e96505882094286f6c31a4beedd39194a4fbc`
- 本轮 6 个 tracked 增量文件的 `source_tracked.diff` SHA256：
  `acb942830f6373ab6690938f76b5cd17060c4f979b1d84e708796e9a39cb4afd`
- H11.4 GQA6 源码、build copy 和 installed 文件共同 SHA256：
  `693ff7ef74db4bda4ce1bbc43e6d0289eb583c9b7dbcc65fcd4edec423f46690`
- `utils.py` 仓库与 installed 文件共同 SHA256：
  `ee03026af2d22e92b9f3a3c1a0af7d555a08ba69e76f8715ff594d48fb2652dc`
- 累计源码和固定脚本清单 `source_runtime_manifest.sha256` 已执行
  `sha256sum -c` 全部通过，manifest SHA256：
  `6e9e9ca62f27d6d88e47ab40d9abc3e5e1156e2b601732404f675677c6538bfc`
- build、wheel、accuracy 和预测产物统一清单 `final_evidence.sha256` 已执行
  `sha256sum -c` 全部通过，manifest SHA256：
  `134bd35238c3c4dcd46f3002341c1e93a195dec14f93911c988fbd9286566844`
- serve：
  `/public/home/tangyu408/testdata/goal_runs/20260712_h11_4_h10_7_serve`
  的 server log 确认 ROCM AITER Unified Attention 和 H11.4 marker 均命中。
  accuracy 完成后最终健康检查为 HTTP 200，并于 epoch
  `1783800951–1783800958` 正常停止；记录见 `shutdown_record.txt`。

H11.4 文件在当前工作树中仍为 untracked；迁移或提交时必须显式携带该完整
文件。R24 的增量 diff 只覆盖本轮 6 个 tracked 文件，累计复现还必须携带
R23 已冻结的源码 evidence。

固定脚本 SHA256 未变化：

- `run_throughput.sh`：
  `adf0cf91266745b37df916926c7d495ec79f00a11be653c219d1d5df4d93c681`
- `run_accuracy.sh`：
  `2e641672a45ac96318c2118df8df4dae2babf87c16afd49cbe4b037ff9beed4e`
- `start_vllm.sh`：
  `7c3e8c5ecdf02109e02af8c3b5ba05050b26339c7f50869b5288eea359364fad`

### H10.7 微基准与数值门槛

证据为最终 build 目录中的 `final_microbench.log` 和
`validation.log`：

| m，k=5120，n=1 | median latency | effective bandwidth |
| ---: | ---: | ---: |
| 14336 | 121.3278 us | 1.210 TB/s |
| 16384 | 138.7819 us | 1.209 TB/s |
| 34816 | 294.7050 us | 1.210 TB/s |

三个大投影相对 H10.6 合计估算每个 decode token 进一步减少约
`3.11 ms`；`m=96` 继续使用原 LLMM1 `rows=4`，约 `10.83 us`。
多 seed/shape 的 9 组 same-input 验证全部通过：H10.7 相对
`torch.nn.functional.linear` 的最大绝对误差不超过 `0.00390625`，
mean absolute error 不超过 `2.4e-7`；5 组非法接口输入均被拒绝。

### 10min 总窗口筛选与无效代理尝试

首次 H11.4 + H10.6 尝试
`/public/home/tangyu408/testdata/goal_runs/20260712_h11_4_h10_6_all3_10m`
的 4-8K、8-16K 两档均为 0/3，日志明确记录
`Service Unavailable` 和 `All requests failed`，第三档未产生结果；
`status.txt`、`run_end_epoch.txt` 为空且无窗口结束文件。诊断确认本机
localhost 请求走代理时返回 503，清除代理变量后同一健康端点返回 200。
该次属于明确错误而提前中止，不纳入有效测量。

| 候选 | 总窗口 | 固定 all3 主体 | 4-8K tok/s，TPOT | 8-16K tok/s，TPOT | 16-32K tok/s，TPOT | 完成 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H11.4 + H10.6 retry | 611 s | 254 s | 12.923551，50.661 ms | 13.665765，51.249 ms | 8.073505，52.431 ms | 9/9，failed=0 |
| H11.4 + H10.7 | 613 s | 251 s | 13.500518，47.336 ms | 14.680806，47.956 ms | 8.558906，49.141 ms | 9/9，failed=0 |

有效路径分别为：

- `/public/home/tangyu408/testdata/goal_runs/20260712_h11_4_h10_6_all3_retry1_10m`
- `/public/home/tangyu408/testdata/goal_runs/20260712_h11_4_h10_7_all3_10m`

两次均以测试后等待补足总窗口到 600 秒以上；不能误写成 workload 主体连续
运行 10 分钟。H10.7 三档均优于 H10.6，因此晋级 full。

### 三次固定 full all

三次均使用未修改的 `run_throughput.sh all`，未传第二参数并显式绕过
本机代理。每次三档各 50 条，合计 450/450 completed、failed=0；每一轮的
每一档 benchmark 主体均超过 600 秒。

| run | 4-8K tok/s | 8-16K tok/s | 16-32K tok/s | 相对 R23 加权 | 综合分，K=1 | wrapper |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| repeat1 | 18.353870994415 | 15.605704715609 | 11.435246853256 | +20.595366284% | 85.711177920 | 2671 s |
| repeat2 | 18.345542243199 | 15.602994180320 | 11.434880896856 | +20.573413685% | 85.704306763 | 2671 s |
| repeat3 | 18.348966838444 | 15.604415084332 | 11.434319170690 | +20.581226611% | 85.706984666 | 2672 s |

证据目录：

- `/public/home/tangyu408/testdata/goal_runs/20260712_h11_4_h10_7_full_run1`
- `/public/home/tangyu408/testdata/goal_runs/20260712_h11_4_h10_7_full_run2`
- `/public/home/tangyu408/testdata/goal_runs/20260712_h11_4_h10_7_full_run3`

三轮统计使用 `n=3`、`df=2`、`t=4.3026527299` 的双侧 95% CI：

| 指标 | mean | sample std | 95% CI |
| --- | ---: | ---: | ---: |
| 4-8K tok/s | 18.349460025353 | 0.004186221351 | [18.339060875024, 18.359859175682] |
| 8-16K tok/s | 15.604371326753 | 0.001355797342 | [15.601003339446, 15.607739314060] |
| 16-32K tok/s | 11.434815640267 | 0.000467271391 | [11.433654873784, 11.435976406750] |
| 综合分，K=1 | 85.707489782785 | 0.003463315726 | [85.698886429582, 85.716093135987] |
| 相对 R23 加权提升 | +20.583335526872% | 0.011127209758 pp | [+20.555694005484%, +20.610977048260%] |

三轮均值相对 official baseline 的加权吞吐提升为 `+82.257604%`，
相对 H6.1c 为 `+49.168718%`；相对 R23 各档均值提升分别为
`+14.614732% / +18.877524% / +27.405424%`。

### SLA

| 指标 | official 1.5x 上限 | 三轮候选最大值 | 结论 |
| --- | ---: | ---: | --- |
| 4-8K TTFT P99 | 7188.718023 ms | 2372.534232 ms | 通过 |
| 8-16K TTFT P99 | 37329.278019 ms | 5571.223128 ms | 通过 |
| 16-32K TTFT P99 | 43111.256069 ms | 8749.062950 ms | 通过 |
| 全局 TPOT P99 | 107.705132 ms | 49.546679 ms | 通过 |

三轮从 150 个请求原始 ITL 重建的全局 TPOT P99 分别为
`49.535646 / 49.546679 / 49.542734 ms`。

### C7 Evidence Card 补充

- 候选 ID：H11.4 + H10.7；变更文件和精确 gate 见“累计源码栈与本轮
  新增机制”。
- baseline wheel：R23 冻结 wheel
  `vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl`，mtime
  `2026-07-11 19:42:23 +0800`，大小 `57,026,430 B`，SHA256
  `95b2aa5f43eb40ffec849ea2550e611e7534205634bbd2bf13fde79a2d371cf7`；
  R23 的 `install_verify.txt` 记录版本、site-packages 路径、H10.4/H11.3
  marker 和 native extension 路径。
- 新 wheel：同名文件，mtime `2026-07-12 01:14:49 +0800`，大小
  `57,074,786 B`，SHA256 为本节前述 `399c7a...`；当前
  `vllm --version` 和 `pip show vllm` 均为
  `0.18.1+das.dtk2604`，位置为
  `/usr/local/lib/python3.10/dist-packages`。
- 构建日志明确记录 `running bdist_wheel`、`running build_py`、
  `running build_ext` 和 `MAX_JOBS=16`，安装日志记录从
  `./dist/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl`
  卸载重装成功。该次 build 的完整外层 shell 命令与是否另行执行
  `build_py --force` 没有单独留存，不能事后声称已记录；最终 wheel、
  repo 和 site-packages 的关键源码哈希一致，证明实际测试 wheel 包含本轮
  变更。这是可复现记录的非阻断缺口。
- 环境变量：没有新增性能控制变量。外层命令清除 `MODEL_DIR`、
  `SERVED_MODEL_NAME`、`VLLM_HOST`、`VLLM_PORT`、
  `MAX_CONCURRENCY`、`REQUEST_RATE`、`CUSTOM_OUTPUT_LEN` 和
  `NUM_WARMUPS`，以恢复固定脚本默认值；
  同时清除 `http_proxy/https_proxy/HTTP_PROXY/HTTPS_PROXY` 并设置
  `NO_PROXY/no_proxy=127.0.0.1,localhost`，后者仅用于防止 localhost
  请求走代理。
- 固定命令：服务由未修改的 `./start_vllm.sh` 启动；筛选使用未修改的
  `./run_throughput.sh all 3` 并将总窗口补足到 600 秒以上，full 使用
  未修改的 `./run_throughput.sh all` 且不传第二参数；最终精度使用
  未修改的 `./run_accuracy.sh all`，不传第二参数。启动后的
  `/v1/models` 与 chat smoke test 产物分别为 serve evidence 中的
  `models.json` 和 `chat_check.json`。
- 路径命中：serve log 的 H11.4 marker、仓库/installed 双哈希以及
  H10.7 microbench/validation 共同证明目标 backend 命中。
- 瓶颈归因：H10.7 针对单 token 大投影的权重带宽受限 GEMV；
  H11.4 针对长 prefill GQA6 attention2d 的 launch/MFMA 配置。

三轮均值性能字段如下；完整逐轮值保存在三个 full 目录的
`results/*/result.json`：

| 档位 | request throughput | mean TTFT | mean TPOT | mean E2E | output throughput | completed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4-8K | 0.072470221269 req/s | 1836.192563 ms | 47.322297 ms | 13783.194219 ms | 18.349460025 tok/s | 150/150 |
| 8-16K | 0.056759680368 req/s | 4431.298265 ms | 48.040840 ms | 17602.563374 ms | 15.604371327 tok/s | 150/150 |
| 16-32K | 0.052266274981 req/s | 8349.433349 ms | 49.193631 ms | 19117.220659 ms | 11.434815640 tok/s | 150/150 |

评分、SLA、三次方差/CI 和精度表分别见相邻小节。合规上未修改固定脚本、
模型、tokenizer、chat template、serve 参数或 scheduler，未使用并发调优、
prefix cache、持久化量化、模型结构变化或数据集改写。结论为保留并冻结。

### Accuracy 与精度系数

固定 `run_accuracy.sh all` 返回 0，运行 1008 秒：

- evidence：
  `/public/home/tangyu408/testdata/goal_runs/20260712_h11_4_h10_7_accuracy_all`
- OpenCompass output：
  `/public/home/tangyu408/testdata/accuracy_debug/output/local_accuracy_qwen35/20260712_034936`
- 权威最终表：上述 evidence 的 `run_accuracy.log`

| 数据集 | official baseline | candidate | 相对下降 Delta | k |
| --- | ---: | ---: | ---: | ---: |
| hotpotqa | 77.959706960 | 77.959706960 | 0.000000% | 1.00 |
| gov_report | 32.961006236 | 33.196573970 | -0.714686% | 1.00 |
| retrieval_multi_point | 100.00 | 100.00 | 0.000000% | 1.00 |
| aggregation_keyword_aggregation | 100.00 | 100.00 | 0.000000% | 1.00 |

四类均为 `k_i=1.00`，故 `K=1.00`；三轮综合分等于上表吞吐公式分，
均值 `85.707490/100`。

OpenCompass 原生 `AccEvaluator` 的中间 summary 把 aggregation 记为
`0.0`，因为 30 条预测列表与 gold 的元素相同但顺序不同，原生 evaluator
按完整字符串顺序敏感比较。固定脚本按 C1 使用 `target_list` 与预测列表的
`Counter` 多重集合等价重算，独立复算 30/30，最终为 `100.00`；official
baseline、R23 也同样是原生 0、固定表 100。按 C1，原生 summary 不能替代
固定脚本最终表。

### 输出重复性、漂移与停止条件

- 三次 H10.7 full 的 150/150 `output_lens` 和 150/150
  `generated_texts` 逐请求完全一致；每轮总输出 token 均为 37345，
  分档为 `12660 / 13746 / 10939`。
- repeat1 相对 R23 repeat1：长度相同 118/150，文本相同 103/150；总输出
  token 从 36803 增至 37345，即 `+1.473%`；1024-token cap 样本从
  12 增至 16；输出审计在其中至少两条 cap 样本中观察到明显重复计数/
  推理循环。
- 本轮未另做独立 API token-id、finish_reason、stop_reason 重放，因此只
  声明固定 full 的文本/长度重复性和 OpenCompass accuracy 闭环，不宣称已
  采集全部输出字段或改变请求序列后仍逐 token 确定。
- 上述输出漂移由固定 accuracy 的 `K=1.00` 按 C4 完成计分豁免，但仍作为
  后续候选必须保留的风险记录。

最终综合分三轮全部超过 85；同时相对进入本轮前当前最佳 R23 的加权吞吐提升
三轮全部超过 20%。目标已双重达成，应冻结 R24 wheel、源码和 evidence，
不继续扩大 H10.7 shape/config 或 H11.4 kernel gate。

## R25 H11.5/H10.8 候选筛选与 H10.10 provisional

本节记录 R25 的候选筛选、full x3、SLA、accuracy、输出审计和服务清理。
R24 是冻结直接比较基线；R25 H11.5 + H10.8 已得到 `K=1.0` 和最终三轮
均分 `88.5484555040153`。standalone/microbench 仍只用于归因，不替代已完成的
固定吞吐/accuracy 证据。R25 未达到 90 或相对 R24 +20%；但从 epoch
`1783824849` 到 `1783842849` 已满 `18000 s`，终止复验 epoch
`1783842892` 对应累计 `18043 s`，第三个终止条件已经满足。因此 R25
按时间条件结束；该结论不表示两个性能条件达成。

### H11.5 wide-causal GQA6 prefill：实现、数值与 608 秒小样本

证据目录：

- build/standalone validation：
  `/public/home/tangyu408/testdata/goal_runs/20260712_h11_5_wide_causal_build`
- service：
  `/public/home/tangyu408/testdata/goal_runs/20260712_h11_5_wide_causal_serve`
- 固定 `run_throughput.sh all 3`：
  `/public/home/tangyu408/testdata/goal_runs/20260712_h11_5_wide_causal_all3_10m`

H11.5 只对 gfx936、BF16、head dimension 256、GQA6、
`cache_block_size=784`、单序列且 `max_seqlen_q >= 128` 的 prefill 使用
wide-causal kernel：`BLOCK_Q=32`、每 CTA 两个 query heads、逻辑
56-token K/V tile、64-token MFMA padding，并按 query block 计算 causal KV 上界。
其他 shape、多序列和 decode 保留 H11.4/AITER 原路径。build 和 validation
均返回 0，候选 wheel SHA256 为
`7f26c14ca059826cdeb3bc293411fc662523e88810f23bb834882d87de31e892`。
backend 和完整 GQA6 source SHA256 分别为
`2d038efffe89cec4e3101bd95505f520e1a49c753c77044adb35ec14b92e2c92`、
`82a52c2b263de9e557ec933b39debc121b4b73f9abddc58cfb6d78153b128cfd`。

standalone same-input validation 覆盖长 query/sequence 和多序列回退：所有输出
finite，最坏 `max_abs=4.8828125e-4`；目标点相对 H11.4 的 raw kernel
加速为 `1.6755x–3.1083x`，多序列回退与对照位级一致。固定
all3 外层窗口为 epoch `1783827057–1783827665`，共 `608 s`，
body status=0、最终 health status=0，9/9 请求成功：

| 档位 | output tok/s | mean TTFT | 相对 R24 TTFT | mean TPOT | 相对 R24 TPOT |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4-8K | `12.553916` | `1621.651 ms` | `-15.887%` | `47.465 ms` | `+0.273%` |
| 8-16K | `15.196309` | `3364.224 ms` | `-20.472%` | `48.044 ms` | `+0.184%` |
| 16-32K | `9.657658` | `6239.829 ms` | `-24.652%` | `49.167 ms` | `+0.052%` |

每档各有 1/3 请求相对 R24 的 generated text/output length 变化，总输出
token 也因此漂移。所以 raw output tok/s 只记录，不作为计分或正确性
结论；候选仍需 full 输出审计和固定 accuracy 闭环。

### H10.8 t640 wave-pair LLMM1：实现、数值与 602 秒组合小样本

证据目录：

- 单 TU gfx936 编译：
  `/public/home/tangyu408/testdata/goal_runs/20260712_h10_8_single_tu_compile`
- isolated runtime validation：
  `/public/home/tangyu408/testdata/goal_runs/20260712_h10_8_runtime_validation`
- H11.5 + H10.8 build/install：
  `/public/home/tangyu408/testdata/goal_runs/20260712_h11_5_h10_8_build`
- service 与固定 `run_throughput.sh all 3`：
  `/public/home/tangyu408/testdata/goal_runs/20260712_h11_5_h10_8_serve`、
  `/public/home/tangyu408/testdata/goal_runs/20260712_h11_5_h10_8_all3_10m`

H10.8 保留 H10.7 的 gfx936/BF16/`n=1`/`k=5120`/无 bias 与三个精确
`m in {14336,16384,34816}` gate，把大投影从
`LLMM1Strided(4,320)` 改为 `LLMM1Strided(4,640)` wave-pair reduction；
`m=96` 仍走原 LLMM1。单 TU 编译返回 0，gfx936 code object 为
`29 VGPR / 11 SGPR / 5200 B LDS / 0 spill`。源码 SHA256：

- `csrc/rocm/skinny_gemms.cu`：
  `fb7635f3ce0dcaedb050ad41f85eeb99d6d334dd3e408dcc8a92d57c68966884`
- `vllm/model_executor/layers/utils.py`：
  `ddb9ee5dafa9ba3b3cd524b8a3fe63706812fdff3d7baae65517e8b069f1f661`

isolated validation 字段 `all_passed=true`；三 shape 各 10 seeds 的 t640 都与
H10.7 t320 位级一致，重复执行也位级一致。zeros 精确，NaN/Inf
分类一致，FP16、非目标 K/M、非连续/未对齐输入和 `n!=1` 均被拒绝。
`31 groups x 50 calls` 随机交错结果为：

| M | H10.7 t320 | H10.8 t640 | improvement |
| ---: | ---: | ---: | ---: |
| 14336 | `120.243 us` | `110.854 us` | `+8.469%` |
| 16384 | `137.498 us` | `127.030 us` | `+8.240%` |
| 34816 | `292.534 us` | `271.533 us` | `+7.734%` |

组合 build/install 均返回 0，wheel SHA256 为
`03568ba87ff64fd0a8aade299026d7ee78cbf40d9c1ed5884fb584250b2031f2`。
固定 all3 外层窗口为 epoch `1783828554–1783829156`，共 `602 s`，
body status=0、最终 health status=0，9/9 请求成功：

| 档位 | output tok/s | mean TTFT | mean TPOT | 相对 H11.5 TPOT |
| --- | ---: | ---: | ---: | ---: |
| 4-8K | `12.948555` | `1619.977 ms` | `44.981 ms` | `-5.233%` |
| 8-16K | `15.771697` | `3363.076 ms` | `45.626 ms` | `-5.033%` |
| 16-32K | `9.889236` | `6236.690 ms` | `46.755 ms` | `-4.906%` |

三档 generated text 和 output length 都与 H11.5 逐请求完全相同，因此
H10.8 没有引入新的输出漂移。完整契约、数值、指标和 artifact hashes
还汇总在该目录的 `evidence_card.md`。这只是当时的 provisional 筛选结果；
后续 full x3、SLA 和 accuracy 闭环见本节末尾。

### GDN MFMA 配置探测：reject

证据目录：
`/public/home/tangyu408/testdata/goal_runs/20260712_gdn_mfma_config_probe`。
探测覆盖 `T=16/32/64/512/4096`、`num_warps=2/4/8`，每侧 warmup 3 次、
随机交错 15 次。脚本与完整 JSON 的 SHA256 分别为：

- `gdn_mfma_config_probe.py`：
  `75c070c93754a5f1bed5dbc77e7b6c73a5c63246e2380e099173a0144d597179`
- `full_w2_w4_w8.json`：
  `48b0dd6aca769e10be1aecb8b899205846a102ecd712924f81c439f5a204f7ec`

在与长输入最相关的 `T=4096` 点，数值有效的最佳结果为：

| kernel | 最佳配置 | time reduction | speedup |
| --- | --- | ---: | ---: |
| `chunk_scaled_dot_kkt` | BK128, w4 | `5.4241%` | `1.0574x` |
| `solve_tril64` | w2 | `7.8125%` | `1.0847x` |
| `recompute_w_u` | BK64/BV64, w2 | `5.8363%` | `1.0620x` |
| `chunk_fwd_o` | BK128/BV128, w4 | `36.0648%` | `1.5641x` |

`chunk_delta_h_stateful` 在 `T=4096` 没有数值有效候选，summary 也把
它列为 missing kernel。探测脚本因此从分子和分母同时排除了该项：
JSON 中 `21.4894%` weighted time reduction 和 `1.2737x` weighted speedup
是其余四个可用 kernel 按 R23 profile 权重重新归一化后的上界，不是五
kernel 的完整加权改善。若保守地将缺失项视为 `1x`，完整五 kernel
估算仅为 `16.2701%` 时间降幅、`1.1943x` 加速；两种口径都低于约
`40%` 的端到端 `3%` 筛选线，也低于约 `55%` 的 90 分 bridge
筛选线。因此该估算不足以支持生产改动。结论为不修改 GDN 生产源码、
不进入 600 秒小样本。

### H10.9 PyTorch BLAS backend selector：no-op 并回滚

backend 探测目录：
`/public/home/tangyu408/testdata/goal_runs/20260712_prefill_gemm_backend_probe`。
最初扫描中的 `torch_default` 标签不能解释为进程原始默认值：调用
`preferred_blas_library("default")` 本身会切换到 CublasLt。新进程和服务
marker 的可核验证据表明生产运行在调用补丁前已经是
`_BlasBackend.Cublas`，即 PyTorch 的 hipBLAS/rocBLAS 路径。对行投影的
复核也得到 K6144 rocBLAS `49.5935 us`、K17408 rocBLAS `150.2319 us`，
与既有生产路径一致。

H10.9 build 目录：
`/public/home/tangyu408/testdata/goal_runs/20260712_h11_5_h10_9_build`。
build/install 均返回 0，候选 wheel SHA256 为
`09501b99beb15c4a3481e398b811b54c25213282293e8fc5240bdae6a64d6ed4`。
服务目录：
`/public/home/tangyu408/testdata/goal_runs/20260712_h11_5_h10_9_serve`；
`server.log` 明确记录 `previous=_BlasBackend.Cublas`，`shutdown.txt` 明确记录
`rejection=H10.9 runtime no-op`。因此在进入吞吐测试前停止，删除 H10.9
源码块并重装 H10.8 wheel
`03568ba87ff64fd0a8aade299026d7ee78cbf40d9c1ed5884fb584250b2031f2`。
这不是一个提前结束的有效小样本，也没有产生可计收益。

关键探测产物 SHA256：

- `benchmark_prefill_gemm_backends.py`：
  `1e9db77591c25218e35e29c5dfd6dce7e14d996d3a63682e96869f6bc05b902f`
- `torch_full_results.json`：
  `1aea563040233cab92dbd6ae00ef93c89cf31e7dc395f0ff950730176cd634ae`
- `row_batched_event_recheck.log`：
  `89d9f0f057cd2ce94bba74b7eb6adc8b68adbd89819570c77ddafe7a18a91935`

### H11.6 GQA6 head-group/query-tile 矩阵：reject

证据目录：
`/public/home/tangyu408/testdata/goal_runs/20260712_h11_6_gqa6_headgroup_probe`。
该实验只 standalone raw-launch H11.5 kernel，未修改生产源码。固定长 prefill
点 `q=512, seq=12000` 上，当前 H11.5 heads2/BQ32/w4/tile56 为
`3.9024 ms`、`216 VGPR`、`32768 B` shared、零 spill。可执行变体结果为：

| 变体 | latency | candidate/current | spill |
| --- | ---: | ---: | ---: |
| heads2, BQ16/w2 | `5.6843 ms` | `1.458x` | 0 |
| heads2, BQ16/w4 | `7.6291 ms` | `1.950x` | 0 |
| heads2, BQ32/w2 | `11.8446 ms` | `3.042x` | 141 |
| heads2, BQ64/w2 | `27.1702 ms` | `6.960x` | 521 |
| heads2, BQ64/w4 | `9.3045 ms` | `2.385x` | 102 |

所有可执行 heads2 变体与当前输出 `max_abs=0`，但都更慢。heads3/BQ32 和
heads6/BQ16 都会令 `BLOCK_M=96`，在 `tl.arange` 处因非 2 的幂编译失败；
tile48/64 不满足 `784 % TOKENS_PER_BLOCK == 0`。对 heads3/6 做 128-row
padding 已超出参数扫描，而且 BQ64 的资源结果已显示严重 spill，故停止扫描
并保留 H11.5。`probe.py` 和 `q512_s12000.json` SHA256 分别为
`efa1d2df151c183927b3999148fcc82bcbb2d8e1dbb20c08437824ac851630d4`、
`3ea0746c1389e5eb64d1c5858584ddd07a63f42d45975f8c417c9f4f740d7829`。

### H10.10 K6144 output projection：实现、构建与 installed validation

standalone 探测目录：
`/public/home/tangyu408/testdata/goal_runs/20260712_h10_10_row_projection_probe`。
候选只对 gfx936 BF16 单 token 行投影进行对照，baseline 是新进程显式设置
`preferred_blas_library("hipblas")` 的 `F.linear`。两轮相互独立的
`31 groups x 50 calls` 随机交错结果为：

| shape/候选 | run1 baseline -> candidate | run1 improvement | run2 baseline -> candidate | run2 improvement | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| K6144 `r2_b384_p1` | `50.9375 -> 44.9984 us` | `+13.1987%` | `50.8928 -> 44.9664 us` | `+13.1796%` | 进入候选 production gate，后被端到端否决 |
| K17408 `r4_b512_p2` | `150.9023 -> 146.6239 us` | `+2.9179%` | `150.9407 -> 146.5727 us` | `+2.9801%` | 低于 5%，否决 |

K6144 六个 seed 全部通过 BF16 默认 `assert_close`，每个候选重复运行均位级
一致；相对 rocBLAS 的最坏 `max_abs=0.5`、最坏 mean absolute error
`9.765625e-5`。选中 kernel 使用 2 行、768 threads、两个 384-thread cohort，
standalone gfx936 metadata 为 `28 VGPR / 16 SGPR / 3120 B LDS / 0 spill`。
两轮结果 SHA256 分别为
`d4ddd949d378bdf03930f997780223de2866d6c8880bd42550b3fa89bf0f94c2`、
`7116a2fd5c75ba36ddd7d02218ea2d3db61f440e05d037ac4f71d7e08ad1d92d`。

production 改动只涉及：

- `csrc/rocm/skinny_gemms.cu`：增加
  `LLGemm1_k6144_pairreduce768_r2_kernel` 和 `(rows=2, threads=768)` 的
  K6144/M5120 BF16 参数检查与 launch；
- `vllm/model_executor/layers/utils.py`：增加 gfx936、BF16、`n=1`、
  `m=5120`、`k=6144`、无 bias、weight/input 连续的 exact gate；所有非目标
  shape 保持原 backend。

被拒候选 build/install 目录：
`/public/home/tangyu408/testdata/goal_runs/20260712_h10_10_k6144_production_build`。
单 TU、wheel、install 和 installed validation 均返回 0：

- 被拒候选 wheel SHA256：
  `5c76f909b5fed93ec27fcd2de6555f09d4ba1fdaa5429fd07ce63c73e19272a4`
- `production_source.diff` SHA256：
  `2f3aff28d072e2e19387f9c81723c0ebb1aef60b8c6daaf3b92801dbf4ff90fe`
- repo `skinny_gemms.cu` SHA256：
  `3cb19823071bbb61f08f6056ddc19c84003392ba5e1c61d4a8801009c9c8b11b`
- repo/installed `utils.py` 共同 SHA256：
  `4329939ee47d417f9af2869ea554993b15973501ec76bd2d1e6b1d100090fcef`
- installed `_rocm_C.abi3.so` SHA256：
  `51cfdcb7bc195f8f29837302f50c3c84cb982abdb589d222cf219b36258d2dc8`

installed validation 的两个普通 seed 均通过 BF16 默认 `assert_close` 且重复
位级一致；zeros、low-amplitude、NaN/Inf 分类通过，4 个 ABI negative case
全部拒绝，dispatch exact positive gate 命中，bias/FP16/`n!=1`/非连续输入
均不命中。`validation_results.json` SHA256 为
`6e7e6754c80392f1704eba824d923b0f072836718589a3869f2ef08e1de0bdda`，
字段 `all_passed=true`。

### H10.10 600 秒小样本、最终 health 与回退：reject

H10.10 候选服务目录：
`/public/home/tangyu408/testdata/goal_runs/20260712_h11_5_h10_10_final_serve`。
固定脚本 SHA256 与 R24 相同，`/v1/models`、health 和 chat smoke 均成功；
H11.5 marker 命中。固定小样本目录：
`/public/home/tangyu408/testdata/goal_runs/20260712_h11_5_h10_10_all3_10m`，
`window_start_epoch.txt` 为 `1783832387`。

workload 主体在 epoch `1783832387–1783832612` 完成，外层窗口于
epoch `1783832987` 结束，总长恰为 `600 s`。固定三档均
completed=3、failed=0，body status=0；窗口结束后 `/v1/models` HTTP 200，
服务停止前 final health HTTP 200：

| 档位 | H10.8 tok/s -> H10.10 | throughput 变化 | H10.8 TPOT -> H10.10 | TPOT 变化 | 输出审计 |
| --- | ---: | ---: | ---: | ---: | --- |
| 4-8K | `12.948555 -> 12.823116` | `-0.97%` | `44.981431 -> 45.784435 ms` | `+1.79%` | 3/3 文本和长度完全一致 |
| 8-16K | `15.771697 -> 15.540940` | `-1.46%` | `45.625695 -> 46.457451 ms` | `+1.82%` | 总长度 `605 -> 601`，文本变化 |
| 16-32K | `9.889236 -> 10.160632` | `+2.74%` | `46.754698 -> 47.605460 ms` | `+1.82%` | 长度 `[23,259,76] -> [23,284,76]`，文本变化 |

相对 H10.8 的精确 TPOT 变化为 `+1.785% / +1.823% / +1.820%`，三档
全部退化。20/50/30 weighted raw output throughput 为 `-0.4393%`；16-32K
的局部 `+2.744%` 只来自额外生成 25 tokens，不能覆盖 paired TPOT 负结果。
因此 H10.10 已判定 reject。候选服务于 epoch `1783833008–1783833015`
停止，停止后 health HTTP 000，符合服务已退出。回退证据目录：
`/public/home/tangyu408/testdata/goal_runs/20260712_restore_h10_8_after_h10_10`。
该步骤在 epoch `1783833130–1783833154` 完成且 status=0，重装 H11.5 + H10.8
wheel SHA256
`03568ba87ff64fd0a8aade299026d7ee78cbf40d9c1ed5884fb584250b2031f2`。
回退后 repo/installed `utils.py` SHA256 均为
`ddb9ee5dafa9ba3b3cd524b8a3fe63706812fdff3d7baae65517e8b069f1f661`，
repo 已无 H10.10 K6144 symbol/gate，H11.5 GQA6 文件保持不变。

最终 H11.5 + H10.8 服务目录：
`/public/home/tangyu408/testdata/goal_runs/20260712_h11_5_h10_8_final_serve`。
health、`/v1/models` 和 chat smoke 均通过，H11.5 marker 命中。第一次固定
full `all` 目录为
`/public/home/tangyu408/testdata/goal_runs/20260712_h11_5_h10_8_full_run1`，
full x3、SLA 汇总和固定 `run_accuracy.sh all` 均已完成，详细结果
见下节。
H10.10 只能记为已完成实现/构建/installed validation、600 秒小样本但端到端
失败并已回退的候选，不能记为工作基线或可计分结果。

### H11.5 + H10.8 固定 full x3/SLA/accuracy：计分闭环完成

run1 证据目录：
`/public/home/tangyu408/testdata/goal_runs/20260712_h11_5_h10_8_full_run1`。
未修改的 `run_throughput.sh all` 于 epoch
`1783833368–1783835766` 运行，wrapper 窗口为 `2398 s`，status=0，
结束后 health/models HTTP 200。三档均 completed=50、failed=0，共 150/150
请求成功。三档 `result.json` SHA256 依次为
`e840de7ec8e42d8ea06f1ffe1231040aedb40e33788e5b4dd81abcdd95936a60`、
`76d81cb2d890eeff5df8c7168deac4aec66b541f06c0e72bce55a59b4c8603b7`、
`af4da4cc68128f1ecb4612e685ff341f210505b5887c73f4a624d21f70d92fde`。

| 档位 | output tok/s | 相对 R24 三轮均值 | mean TTFT | TTFT P99 | mean TPOT | TPOT P99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4-8K | `19.589185274` | `+6.756195%` | `1547.389 ms` | `1956.479 ms` | `44.995726 ms` | `45.276733 ms` |
| 8-16K | `17.025544512` | `+9.107532%` | `3596.663 ms` | `6129.860 ms` | `45.716261 ms` | `46.255274 ms` |
| 16-32K | `13.003919637` | `+13.722163%` | `6280.184 ms` | `6548.072 ms` | `46.870334 ms` | `47.201300 ms` |

按赛题请求口径，先对 150 个请求分别计算
`sum(itls) / (output_len - 1)`，再取 P99，run1 全局请求 TPOT P99 为
`47.194474909 ms`；先前写入的 `47.770285 ms` 是把所有 token-level ITL
展平后取 P99，不是 SLA 的请求 TPOT 口径，现已纠正。本轮 TTFT/TPOT
均未触发 SLA 上限。以 R24 三次 full 每档吞吐均值
`18.349460025353 / 15.604371326753 / 11.434815640267` 为直接对照，
20/50/30 加权相对提升为 `+10.02165396%`。假设 accuracy 系数
`K=1` 的 pre-accuracy 公式分为 `88.4903491377583`。因此本轮既未达
`90` 分，也未达相对 R24 `+20%`；仅就 run1 而言又只有一轮数据、尚无 accuracy，
不能计为新的可计分最佳。

输出审计也要求保留强 caveat。R24 三轮 full 的文本与长度彼此完全一致；
将 R25 run1 与该固定输出逐请求比较：

| 档位 | length 相同 | text 相同 | R24 -> R25 总输出 token | 1024-token cap |
| --- | ---: | ---: | ---: | ---: |
| 4-8K | `38/50` | `31/50` | `12660 -> 12636` | `7 -> 6` |
| 8-16K | `38/50` | `34/50` | `13746 -> 13746` | `5 -> 5` |
| 16-32K | `34/50` | `32/50` | `10939 -> 10485` | `4 -> 3` |
| 合计 | `110/150` | `97/150` | `37345 -> 36867` | `16 -> 14` |

总输出 token 减少 `478`，即 `-1.27996%`；8-16K 总 token 虽恰好相同，
仍有 16/50 文本变化。因此吞吐、相对提升和 pre-K 分数都必须视为受输出
漂移影响的 provisional raw 数据；只有固定 accuracy、三轮重复性与停止原因
审计完成后才能得出最终正确性/计分结论。

#### Full run2 与两轮重复性

run2 证据目录：
`/public/home/tangyu408/testdata/goal_runs/20260712_h11_5_h10_8_full_run2`。
固定 `run_throughput.sh all` 于 epoch `1783835836–1783838224` 运行，
wrapper 窗口 `2388 s`、status=0、结束后 health/models HTTP 200。三档均
completed=50、failed=0，共 150/150 请求成功。三档 `result.json` SHA256
依次为
`d88cf74185109c553c291cb2fb05c620a6aba6b7a097edfbda98b15f03844724`、
`4ed7e62ebba4662aa6d085f1e17406c9404b40f18ee94e3e22f6bfa85ef213f0`、
`be23ef6e8a693f01b149186cf6f855b4dc00396fcc1a1930016fd66f182f5a6f`。

| 档位 | output tok/s | 相对 R24 三轮均值 | mean TTFT | TTFT P99 | mean TPOT | TPOT P99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4-8K | `19.587005633` | `+6.744316%` | `1547.272 ms` | `1955.345 ms` | `44.997455 ms` | `45.298294 ms` |
| 8-16K | `17.127750783` | `+9.762517%` | `3503.706 ms` | `4317.923 ms` | `45.706613 ms` | `46.208402 ms` |
| 16-32K | `13.003706764` | `+13.720301%` | `6279.346 ms` | `6549.591 ms` | `46.871595 ms` | `47.228318 ms` |

run2 同口径的全局请求 TPOT P99 为 `47.184350059 ms`；先前写入的
`47.817654 ms` 同样是 token-level ITL P99，现已纠正。本轮 TTFT/TPOT
也未触发 SLA 上限。run2 pre-K 公式分为
`88.5784836941864`，相对 R24 加权提升 `+10.3462122102%`，仍未达
`90` 或 `+20%`。

| 统计 | 4-8K tok/s | 8-16K tok/s | 16-32K tok/s | pre-K 分 | 相对 R24 加权 |
| --- | ---: | ---: | ---: | ---: | ---: |
| run1 | `19.589185274` | `17.025544512` | `13.003919637` | `88.490349138` | `+10.021653961%` |
| run2 | `19.587005633` | `17.127750783` | `13.003706764` | `88.578483694` | `+10.346212210%` |
| run3 | `19.584774729` | `17.126009556` | `13.004622852` | `88.576533680` | `+10.340604760%` |
| 三轮均值 | `19.586988545` | `17.093101617` | `13.004083084` | `88.548455504` | `+10.236156977%` |

run1/run2/run3 在三档的 `output_lens` 和 `generated_texts` 都逐请求完全一致。
用 `jq -c '[.output_lens,.generated_texts]' | sha256sum` 生成的三档输出签名
在三轮中均为：

- 4-8K：`c8c611ec25abd660f9eca1adf7b5f919cad7734a5e83e06847a00457855e71e8`
- 8-16K：`91f264f685361c23f6eda35ea95a313be09148dbecc783a3d09338e3a6dbcf42`
- 16-32K：`8b0ffdf407ffcf041f33f7760fe1dd1fe2af2e9c476f8706b3e126916ba8b5de`

这证明 R25 候选在三轮固定请求上可重复，但不消除上述相对 R24 的
length/text 漂移，也不能代替固定 accuracy。三轮均值仍低于两个性能
终止条件。

#### Full run3、三轮统计与 SLA

run3 证据目录：
`/public/home/tangyu408/testdata/goal_runs/20260712_h11_5_h10_8_full_run3`。
固定 `run_throughput.sh all` 于 epoch `1783838265–1783840653` 运行，
wrapper 窗口 `2388 s`、status=0、结束后 health/models HTTP 200。三档均
completed=50、failed=0，共 150/150 请求成功。三档 `result.json` SHA256
依次为
`2873de22595d5320cc9fa23e384992a9b39ea2102b5bb255ed663a644a34e24f`、
`0aecc33425aeab8a3dd48df64fe1f0bebff1fd89a81f220b20e6d59c4f0464c3`、
`f3d439637857291dcbd9504ad2214821a88920b41dab246769a04fe6079c1251`。

| 档位 | output tok/s | 相对 R24 三轮均值 | mean TTFT | TTFT P99 | mean TPOT | TPOT P99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4-8K | `19.584774729` | `+6.732158%` | `1547.329 ms` | `1956.991 ms` | `45.009888 ms` | `45.286272 ms` |
| 8-16K | `17.126009556` | `+9.751359%` | `3503.313 ms` | `4318.681 ms` | `45.706308 ms` | `46.200932 ms` |
| 16-32K | `13.004622852` | `+13.728312%` | `6278.613 ms` | `6549.848 ms` | `46.865152 ms` | `47.225344 ms` |

run3 同口径的全局请求 TPOT P99 为 `47.189229964 ms`。run3 pre-K
公式分为 `88.5765336801012`，相对 R24 加权提升
`+10.3406047597%`，仍未达 `90` 或 `+20%`。

三轮 pre-K 分均值为 `88.5484555040153`，95% t 置信区间为
`[88.4234262842584, 88.6734847237721]`；相对 R24 的三轮加权提升均值为
`+10.2361569769%`，95% t 置信区间为
`[9.7746384263%, 10.6976755275%]`。最终 `K=1.0`，三轮均值及置信区间
上界都没有命中 `90` 分或相对 R24 `+20%` 终止条件。

SLA 采用与 R23/R24 相同的请求口径。三轮全局请求 TPOT P99
依次为 `47.194474909 / 47.184350059 / 47.189229964 ms`；三档
TTFT P99 在三轮中的最大值依次为
`1956.991400 / 6129.860216 / 6549.847646 ms`。

| SLA 项 | 1.5x 上限 | R25 三轮最大值 | 结论 |
| --- | ---: | ---: | --- |
| 4-8K TTFT P99 | `7188.718023 ms` | `1956.991400 ms` | 通过 |
| 8-16K TTFT P99 | `37329.278019 ms` | `6129.860216 ms` | 通过 |
| 16-32K TTFT P99 | `43111.256069 ms` | `6549.847646 ms` | 通过 |
| 全局请求 TPOT P99 | `107.705132 ms` | `47.194474909 ms` | 通过 |

三轮的 `output_lens`、`generated_texts` 和上述三个输出签名完全一致。
相对 R24 的 caveat 也在三轮重现：text 相同 `97/150`、length 相同
`110/150`，总输出 token 每轮均为 `36867`，比 R24 的 `37345` 少
`478`（`-1.27996%`）。跨轮可重复不等于相对 R24 无漂移；固定 accuracy
最终得到 `K=1.0`，使该漂移按 C4 不触发计分惩罚，但风险记录仍保留。

#### Accuracy：完成，`K=1.0`

固定 accuracy 证据目录：
`/public/home/tangyu408/testdata/goal_runs/20260712_h11_5_h10_8_accuracy_all`。
`run_accuracy.sh all` 于 epoch `1783840674–1783841621` 运行，窗口
`947 s`、status=0，结束后 models/health HTTP 200。OpenCompass output directory：
`/public/home/tangyu408/testdata/accuracy_debug/output/local_accuracy_qwen35/20260712_151818`。
固定脚本 `run_accuracy.log` SHA256 为
`0c500759ce663e1196dcf3b573d261b93eb97bd4b5f887979df810f93fe7ecf2`，
OpenCompass 完整运行日志 SHA256 为
`2eebab2d7a96c7ca3a5bdcbb921d989620dc6a2ed85bab0c896ba95b9df6e764`。

| 数据集 | official baseline | R25 Final Results | 相对下降 | `k_i` |
| --- | ---: | ---: | ---: | ---: |
| hotpotqa | `77.959706960` | `77.959706960` | `0.000000%` | `1.00` |
| gov_report | `32.961006236` | `33.054713499` | `-0.284297%` | `1.00` |
| retrieval_multi_point | `100.00` | `100.00` | `0.000000%` | `1.00` |
| aggregation_keyword_aggregation | `100.00` | `100.00` | `0.000000%` | `1.00` |

固定脚本 Final Results 的四项显示值为
`77.96 / 33.05 / 100.00 / 100.00`，所以最终精度系数 `K=1.0`。
OpenCompass 原生 summary 对 `aggregation_keyword_aggregation` 仍显示 `0.00`；
这与 official/R23/R24 的既有口径相同，因为 30 条预测与 gold 的元素相同
但顺序不同，原生 evaluator 做完整字符串顺序敏感比较。固定脚本按 C1
对 `target_list` 与预测列表做 `Counter` 多重集合复算，30/30 通过，权威最终值
为 `100.00`；不得用原生 summary 的 `0.00` 替代固定脚本 Final Results。

`K=1.0` 后，三轮最终综合分与 pre-K 公式分相同，依次为
`88.490349137758 / 88.578483694186 / 88.576533680101`，均值
`88.548455504015`。三轮相对 R24 加权提升均值仍为
`+10.2361569769%`。因此 accuracy 通过不改变性能终止判断：均分未达
`90`，相对 R24 也未达 `+20%`。

#### 服务清理与 5 小时终止闭环

最终服务证据目录：
`/public/home/tangyu408/testdata/goal_runs/20260712_h11_5_h10_8_final_serve`。
accuracy 结束后，`/v1/models` 和 health 在停止前均为 HTTP 200。服务于
epoch `1783841652–1783841659` 停止，停止后 health HTTP 000，表明无残留服务。
停止前/最终 server log SHA256 分别为
`82d0de8b0ac3b9c35b88a97868a9f34677b39177f1d6670b112b08f465b89d30`、
`0017f838c957e4ee673bd6c35a91b85c91a04463fbb6eb3b039e43c81453f18b`。

R25 的 score 90 和相对 R24 +20% 条件均未达到。goal start 为
`1783824849`；epoch `1783842849` 时已持续 `18000 s`，5 小时终止条件
满足。终止复验于 epoch `1783842892` 再次确认累计 `18043 s`。full、
SLA、accuracy、输出审计、服务清理和 final evidence 均已冻结，R25 按
时间条件结束。
