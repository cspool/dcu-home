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
| R26 | 六小时源码优化续轮 | GDN+BF16-GQA 与 GDN-only 均已 reject；GDN-only full×3 共 450/450、SLA/accuracy `K=1.0` 通过，但均分 `88.131406`、相对 current-best `-1.679841%`；源码/runtime 已回退 H11.5+H10.8、服务停止，epoch `1783869431` 已完成 6 小时终止审计 |
| R27 | 八小时源码优化与矩阵单元探索 | H10.18 bundle 因稳定输出漂移 reject；P9 通过 C101/C100 差分把漂移归因到 Hg3，H10-only 两条输出 exact。H10-only 664.215 秒 all3 三档/SLA/输出均通过，但 weighted `+0.958538% < 1%`，不进 full；已回装 pinned baseline wheel，source/runtime/8001/GPU clean；epoch `1783903135` 按 8 小时时间条件结束 |
| R28 | P10 矩阵单元与 LDS/GDN 窄候选 | R28-A/E 性能 reject；R28-B/C/D/G/J 前门或静态 reject；R28-F 强制矩阵化 non-MMOP 边界静态 reject；R28-H C010 证明 S32 单独足以漂移；R28-I 复核旧 validator 九条 `exact=false`。R28-K 最终综合确认 Attention/GDN Prefill 主链与 Prefill Linear 已用 MMAC，向量例外不应为覆盖率强行矩阵化。均不做 production integration |

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

## R26 六小时源码优化续轮

R26 最新目标在 epoch `1783847801` 启动，6 小时终止 epoch 为
`1783869401`；用户将综合分门槛由续轮初始的 `90` 提高为 `95`。
直接基线固定为 R25 已闭环的 H11.5 + H10.8：最终三轮均分
`88.5484555040153`、相对 R24 加权 `+10.2361569769%`、`K=1.0`。
本轮性能增量的“当前最佳”均指该直接基线，不回退到 R24 或 official。

启动状态审计：

- repo 仍为 H11.5 + H10.8 的 12 个 tracked 修改和 1 个 untracked GQA6
  源文件；关键源码 hashes 与 R25 final evidence 一致；
- installed `_rocm_C.abi3.so` SHA256 为
  `51e4839b564355279fcca4bc426ccd1da0a5f03d0e39006210960e99fd124ab1`；
- localhost:8001 health 为 `000`，没有 vLLM、throughput、accuracy 或
  OpenCompass 残留进程；
- 固定脚本、模型、权重、tokenizer、chat template、serve 参数和
  scheduler 边界继续只读。

第一轮候选筛选并行覆盖：

1. H10/GEMV：检查 H10.8 t640 之后仍未测试的 gfx936 BF16 `n=1`
   config/shape，先做数值和交错 microbenchmark；
2. H9：审计 Qwen3.5 当前是否遗漏已有 Q/K norm + RoPE fusion，并验证
   head256/GQA6/full-attention 路径的语义和接入成本；
3. H4/H1.2：复核最终栈上的 GDN/FLA prefill/decode tiling、state update
   和融合候选，排除已失败的 L2 norm、未初始化输出和低收益 cache 补丁。

候选只有完成数值验证、production wheel build/install 后才运行固定
`run_throughput.sh all 3`；除明确报错外，每次小样本外层窗口不少于
`600 s`。终止条件为 score `95`、相对 H11.5 + H10.8 同口径性能再次
提升超过 `20%`，或到达 epoch `1783869401`；当前均未满足。

用 official 三档 baseline `12.2076/8.8108/5.3902 tok/s` 和 R25 三轮
均值重算：若三档近似等比例缩放，达到 score `95` 约需在当前最佳上再增
`33.61%`；当前最佳上再增 `20%` 的公式分约为 `93.02`。所以旧 `90` 桥接线
不再是终止证据，微小配置收益必须继续累积或让位于结构性 fusion。

只读筛选后的第一优先级为 H10.11 `K=5120` wave10 直接归约：用 10 个
wave leader 和约 `160 B` LDS 代替 H10.8 的 `pair_smem[4][320]`，删除
一次 barrier。它先以 standalone r4/r8 variant 覆盖三种真实 decode M、
多 seed/特殊值、code-object metadata 和不少于 `600 s` 的交错微基准，
通过后才允许进入 production。第二优先级为 GDN 长 prefill exact-shape
compiler 配置包：只 gate gfx936/BF16/单序列
`T=4096,Hg=16,H=48,K=V=128,BT=64`，用独立 JIT 副本避免当前 autotune
key 不含 `T` 的跨长度复用；R25 数值失败的 `chunk_delta_h` MFMA 配置仍然
排除。

H10.11 standalone 已在 epoch `1783848620–1783849220` 完成长测，脚本
记录精确窗口 `600.113 s`、`5721` 组，三种 M×10 seeds、repeat、zeros、
low-amplitude 和 NaN/Inf 分类门禁通过。r4 serial/q16 code object 均为
`29 VGPR / 160 B LDS / 0 spill`，r8 为
`43 VGPR / 320 B LDS / 0 spill`。但相对 installed H10.8 的调用频次加权
结果仅为 r4 serial `+0.02845%`、r4 q16 `+0.00204%`，r8 则
`-6.23387%`；三 shape 的 r4 serial 也只有
`+0.0115% / +0.0303% / +0.0295%`。因此 H10.11 明确 reject，不修改
production 源码、不构建 wheel。长测 JSON SHA256 为
`bd4d0c35905f6c538be56e2e623ecaf64cb7f3f868e36ede0496a629753f264b`。

GDN T=4096 四配置包的最终有效 standalone 完成
`600.569 s / 126750 groups`。它先按源码 warmup 的 int64 metadata 执行
T=16/32/64，再以真实 V1 int32 metadata 在 T=4096 独立 autotune；五个
seed 同时覆盖 no-state 和由前块原生 FP32 final_state 构造的 stateful
路径。公开 `o/final_state` 均在 `atol=5e-4,rtol=5e-3` 下 mismatch=0；
所有内部 A/Ai/w/u/h/v_new 的 max abs 不超过一个 BF16 台阶
`0.001953125`，受控比例门禁也通过。

相对真实 int32/T4096 autotune baseline，显式配置的 kernel median 为：
`chunk_fwd_o 0.57136→0.45024 ms (-21.20%)`、
`chunk_scaled_dot_kkt 0.19952→0.19088 ms (-4.33%)`、
`recompute_w_u 0.42176→0.38640 ms (-8.38%)`、
`solve_tril64 0.28240→0.26080 ms (-7.65%)`；按四 kernel R23 profile
权重合计降时 `14.0646%`，全部 code object 为零 spill。有效 JSON SHA256
为 `7e6244be6680251a307bda3918fa0a32b57350718d82f50597d61a840be0245f`。
但该绝对节省按 R11/R23 全 trace 仅约 `0.48%`，不足以单独触发一次
production/service 周期；当前保留为与更高收益候选一起叠加的已验证项。
此前 retry2/3/4 分别因未复现 dtype-separated autotune、BF16 state 构造、
int64 measured-request metadata 被主动终止，均不进入性能结论。

H10.12 persistent-row GEMV 保持 H10.8 每次只让四行 weight/acc live，但让
同一 640-thread WG 顺序处理 G2/G4/G8 个四行组，activation 只加载/转换一次。
code object 为 `34/35/35 VGPR`、`5200 B LDS`、零 spill/private；反汇编
确认每个变体静态只有 `1` 条 activation 加 `4` 条 weight
`global_load_dwordx4` 和动态 row-group loop。三 M×10 seeds 相对 H10.8
逐位一致，repeat 和特殊值门禁通过；正式随机交错完成
`600.079 s / 5468 groups`。

但三个变体全部稳定回归：真实调用频次加权 G2/G4/G8 分别为
`-7.2026% / -8.2729% / -8.6336%`；G2 三 shape 也分别为
`-7.2883% / -7.3108% / -7.1559%`。这说明 x 的 cache 命中已使显式复用
收益不足以抵消长 WG/寄存器 live-range 开销。H10.12 明确 reject，不改
production；长测 JSON SHA256 为
`8818893c84411d63fc553595cc10604dc12205aa2fdb03eeebd7eaa5ee7fd3e0`。

GQA6 decode segmented attention standalone 证据目录为
`/public/home/tangyu408/testdata/goal_runs/20260712_r26_gqa6_decode_probe`。
初始 12 配置中，logical/padded `32/32` 的 3 个 stage code object 为
`251 VGPR`，超过预设 `224 VGPR` 门槛，因此 retry1 在计时前明确
拒绝。retry2 仅保留 logical/padded `16/16`、`56/64`、`64/64`
与 `S=8/16/32` 的 9 个资源合格变体。

retry2 的 5 长度×2 seeds 共 90/90 个 candidate/reference 数值门禁
通过；9 stage + 9 reduce code object 全部零 spill/private，共享内存、
VGPR 和 SGPR 均过门。随机交错计时完成 `600.284 s / 1581 rounds`，
每样本 50 次调用。最佳 `m8_l64_p64_s32` 在 context
`6315/13295/21562` 上的 wall 降时分别为
`20.5551% / 20.5218% / 20.4996%`，GPU 降时分别为
`20.6111% / 20.5744% / 20.5428%`，无单长度回归；加权 wall/GPU 为
`20.5218% / 20.5722%`。不过这低于预注册的双 `40%` production
门槛，所以结果状态为 `reject_performance`，不直接改 production。
retry2 JSON SHA256 为
`49fa5a146fed3a2574b303f3b5d6559307584be29469d22f61d1f0f127b01e07`，
kernel/probe 源码 SHA256 分别为
`e2efe938708d113847ed7217f21d670b5eaef2a941d207851adddabebe26b958`、
`cd13f72c3457e28fa410504a7b995bba3d646652c65299ea85cc849bd91ef4f1`。
该候选是正数但未达独立晋级门槛；保留用于分析 S32 平台和更少
workspace/reduce 开销的后续结构，未宣称端到端或计分收益。
用 16 个 full-attention 层和 R25 full 的真实 output length 折算，其
三档理想吞吐上限约为 `+1.7614% / +1.5243% / +1.1631%`，
20/50/30 加权仅 `+1.4633%`，公式分约 `+0.42`。因此它不单独
进入 production，只可在 H11.7/GDN 另行过门后于同一组合 build 复议。

H11.7 logical64/cache-page-cross standalone 证据目录为
`/public/home/tangyu408/testdata/goal_runs/20260712_r26_h11_7_tile64_cross_probe`。
它保留 H11.5 的 BQ32/BM64、两 query heads/CTA、causal bound 和 compiler
布局，仅将 logical K/V tile `56` 改为 `64`，并在约 6% 跨越
784-token 物理 cache page 的 tile 上显式选择两个 block-table entry。
候选 code object 为 `198 VGPR / 32768 B LDS / 0 spill`，基线 H11.5 为
`216 VGPR / 32768 B LDS / 0 spill`。

前两次均在计时前主动停止。初始 `results.json` 使用相对 H11.5
过严的固定 `max_abs<=1e-3/mean<=5e-6`；retry1 加入冻结 H11.4
同输入对照后，75 行中 33 行只因 H11.5 自身超过固定 sanity 值而
失败，随机非连续 cache page 下 H11.5-vs-H11.4 最大差可达
`0.0078125`。关键的 H11.7 相对同输入 H11.5 新增 max/mean 预算
则 75/75 通过。两次早停 JSON SHA256 分别为
`4fa91b463205c73c7b38f909cfff115cf67761326bc7090244275e916d2eca21`、
`bcc059a54f22b0536c33c85952fc95f1ba5ade78548df13d214978cdb793d833`。

retry2 将 H11.5-vs-H11.4 作为动态基线记录，强制三方 finite、
candidate repeat-exact、零 spill/资源门禁，H11.7-vs-H11.4 归一化
mean `<=1e-3`，且 H11.7 相对同输入 H11.5 的新增 max/mean
`<=0.00390625 / 2.5e-5`。数值与资源 75/75 通过后，正式
随机交错计时完成 `601.005 s / 19448 groups`。

性能结果为明确负数：按真实 all3 每层调用数加权，H11.5/H11.7
分别为 `494.0495 / 531.9669 ms`，候选回归 `-7.6748%`；13 个
真实 shape 无一获益，单 shape 回归范围为 `-6.23%–-22.54%`。
这证明 logical64 虽减少约 12.6% 的 padded dot 列，但跨 784-token page
时的逐 lane block-table/物理地址选择成本更高。H11.7 明确 reject，
不应用其 production patch；retry2 JSON SHA256 为
`ab5e96021997d5d36577f4900992f14d8dcb43d4672b91b7453d11b905e22f89`。

R26 又对 H10.10 K6144 的“standalone 获益、production 回归”做了
只读闭环。compile signature、R23 层数和 exact gate 证明每个 decode
token 确实命中 `48+16=64` 次 K6144 projection；三档 H10.10 TPOT
相对 H10.8 增加 `0.8030 / 0.8318 / 0.8508 ms`，折算每目标调用慢
`12.55 / 13.00 / 13.29 us`。H10.8/H10.10 computation graph 和 6 个
compile artifact 逐文件相同，capture 都成功；4-8K 的 3/3 输出文本/长度
也完全一致，因此排除 ABI、gate、build/install、dispatch、graph 和
数值测量偏差。

根因证据是 cache-residency 协议失配：standalone 对同一份约 60 MiB
weight 先 100 次 warmup，每个 sample 又连续调用 50 次且没有 L2 flush；
production 的 64 份不同 K6144 权重每 token 流过约 `3.75 GiB`。所以
微基准宣称每次节省约 `5.93 us`，production 却发生约
`18.47–19.23 us/call` 的性能反转。H10.10 继续 reject/no-go，没有
可修复的接入 bug。若测 K17408 新 kernel，必须同时记录 hot 和
multi-weight/`256 MiB` flush cold-stream，并只用 cold-stream 作 production 门槛。
审计 README SHA256 为
`547108fc2ea4bc4305743da3c0b8e78fc7e13049a778e198683cff1bd7fff8d3`。

H11.8 reverse causal q-block standalone 证据目录为
`/public/home/tangyu408/testdata/goal_runs/20260712_r26_h11_8_prefill_next_prep`。
候选仅将 axis-0 查询块映射从正序改为
`tl.num_programs(0)-1-tl.program_id(0)`，使最长 causal 工作块先发射；
grid、attention 数学、CTA 内归约、访存和 compiler 配置均不变。
CPU FIFO 模型在 80/160 resident CTA 两个边界上预测加权 raw
`+1.1777% / +3.2506%`，因此预设 `2%` 晋级线。

实测先完成 13 shape×3 seeds 的 39/39 位级一致/repeat 一致门禁；
baseline/candidate code object 均为 `216 VGPR / 32768 B LDS / 0 spill`。
随机交错计时完成 `600.831 s / 605 cycles`，加权 baseline/candidate
分别为 `493.8766 / 493.9310 ms`，即 `-0.0110%`。所有 shape 都只在
约 `-0.053%–+0.022%` 内波动，说明真实 GPU 调度已消除简化 FIFO
顺序优势。H11.8 明确 reject，不改 production；JSON SHA256 为
`5c2038049f06e67b2669e337af5148e57ed32cada470007832cd037b52404d27`。

H10.13 K17408 direct GEMV cold-stream 证据目录为
`/public/home/tangyu408/testdata/goal_runs/20260712_r26_h10_13_k17408_probe`。
候选避开 H10.10 suffix-pair/三 cohort，测试 logical544/physical576 的
4 slots、logical272/physical320 的 8 slots、C1/C2/C4 以及 G2-x。
全部 code object 为 `31–51 VGPR / 80–288 B LDS / 0 spill/private`，
6 seeds 的 rocBLAS/H10.10 同输入数值与 repeat 门禁通过。

测量严格使用 `r26_cold_stream_flush256m_single_call_v2`：每次单调用前
在同 stream 对独立 256 MiB int32 buffer 做 changing-value fill，start event
位于 fill 之后，event 内只有一次目标 kernel。正式窗口完成
`600.015 s / 21219 batches`，每方法 `169752` 次单调用、总计
`1527768` 次 flush。cold H10.10 reference/rocBLAS 中位数为
`153.20 / 154.1595 us`；最佳 `k17408_l544_p576_s4_c1` 为
`161.04 us`，相对 H10.10 回归 `-5.1175%`，G2-x 更回归
`-9.2947%`。7 个候选无一通过 `8%` 门槛，H10.13 reject、不生成
production patch。JSON SHA256 为
`fb9c029fae32ee7445a344c41ff10f3697dab81175026c0114d80075cde2fc10`。

H11.9 cache-divisible half-width prefill 证据目录为
`/public/home/tangyu408/testdata/goal_runs/20260712_r26_h11_9_tile28_probe`。
静态工作模型表明 logical28/pad32 和 logical14/pad16 相对 H11.5
只减少 `0.1452% / 0.2182%` padded dot 列，但 online-softmax 循环
增至 `1.9971x / 3.9913x`；唯一假设是 pad 变小可降低 LDS 并提升
residency。真实编译后，pad32 为 `222 VGPR / 32768 B LDS`，pad16 为
`169 VGPR / 32768 B LDS`，两者均零 spill 但 LDS 完全未降，不可能将
64 KiB LDS-only CTA/CU ceiling 从 2 提到 4/8。因此在计时前
`rejected_before_timing_resource`，这是协议错误/资源否决，不构成
缩短性能窗口。JSON SHA256 为
`af7b7a1cb6a3ad800a82f3fc610e94a06c9622fecc9c639c7fe4889c2a368ba9`。

GQA6 normalized-BF16 workspace standalone 证据目录为
`/public/home/tangyu408/testdata/goal_runs/20260712_r26_gqa6_bf16_workspace_probe`。
FP32 stage 保持在线 softmax，但将每段 `y_j=BF16(a_j/l_j)` 写入 workspace；
max/expsum 保持 FP32，reducer 用 `w_j=l_j*exp(m_j-M)` 重构
`sum(w_j*y_j)/sum(w_j)`，不存在重复除法。vector workspace 因此从
792576 B 降至 399360 B，节省 `393216 B / 49.61%`。

5 lengths×3 seeds 同时对照 AITER 和冻结 FP32 S32，数值/repeat 门禁
通过；BF16/FP32 stage+reduce 共4个 code object 全部过资源门禁。
随机交错计时完成 `600.077 s / 5618 rounds`。在 context
`6315/13295/21562` 上，BF16 相对 FP32 的 wall 降时为
`2.0550% / 2.0058% / 1.9839%`，GPU 降时为
`2.0667% / 2.0130% / 1.9880%`；加权 wall/GPU 为
`2.0090% / 2.0162%`，无长度回归，但低于预注册双 `5%` 门槛。
结果为 `reject_performance`，不独立进 production；只保留为原 FP32
S32+GDN 组合栈的稳定正数 alternate。JSON SHA256 为
`e53c2f0d99bca09a8c5d5809c3e4dfcd9631e462c834ebed2d7b6ea508d13918`。

H10.14 fused-SiLU down-projection 只读审计也已 no-go。固定 Inductor
native 激活路径每层约 `5.218646 us`，64 层为 `333.993 us/token`，
即相对 R23 mean TPOT 的免费消除上限仅 `0.594%`。若把它并入
4-row K17408 GEMV，sigmoid/exp 会按 1280 row groups 重复；在还要
净省 0.25 ms/token 的条件下需约 `1.70e13 activation eval/s`，显然
不可行。因此不 HIP 编译、不 GPU 测试、不改 production；审计
README SHA256 为
`ccc33fd47422e9ca8be6494fa97a5b055f7707f10695fec00e71eff0ec1b1840`。

H10.15 K6144 direct/serial cold-stream standalone 证据目录为
`/public/home/tangyu408/testdata/goal_runs/20260712_r26_h10_15_k6144_direct_cold_probe`。
它覆盖 768x1、384x2、256x3 三种 thread/serial-slot 分解及各自 C2/C4，
六个候选全部通过 resource、disassembly、多 seed、特殊值和 determinism
门禁。正式冷流窗口完整运行 `600.001501 s`，每方法完成 `256944` 次
随机 paired single-call，总计 `2055552` 次独立 256 MiB changing-value
flush；run status `3` 表示完成后的 performance reject，并非提前终止。

更快的同轮基线 H10.10 C2 中位数为 `51.360 us`，rocBLAS 为
`53.760 us`；最佳 H10.15 768x1 C2 为 `51.520 us`。它虽相对 rocBLAS
快约 `4.1667%`，但相对真正的 faster baseline H10.10 为
`-0.311529%`，paired 分布的 p10/p90 也仅
`-1.5674% / +0.6192%`，远低于预注册 `+8%` 门槛。六候选最终
`passing=[]`，因此全部 reject。`results.json` SHA256 为
`1533c461322515ca33027e56eee02904761263b588c4aa31f6e358d0361cd508`。
production `skinny_gemms.cu` 与 `utils.py` 的 before/after hashes 逐字节
相同；没有 production patch/build/wheel/install/service/fixed all3，
冻结 H11.5 + H10.8 production 未改。该 no-go 不结束 R26，goal 仍进行中。

R11 中排名第四的 `FillFunctor<int>` 也完成了 kernel-to-source
attribution。trace 的 `83740` 次/`18320.211 ms` 中，`83476`
次/`18319.614 ms`（`99.9967%`）是 Triton AMD autotune 计时前对
`67108864 x int32 = 256 MiB` 临时 buffer 执行的 L2 cache flush。
其后继 kernel 全部是六类 GDN prefill autotune kernel：
`chunk_fwd_o/kkt/merge/recompute/cumsum/delta`；大型 fill 只出现在服务
ready 前和 initial test/warmup，正式 benchmark 区间为 `0` 次。

固定正式区间真实的 int fill 只有 `144` 次/`0.30096 ms`，来自
`attention/backends/utils.py` 两个 `2048 x int32` causal-conv metadata
buffer。逐层 `core_attn_out=torch.zeros(...)` 则是 `1152` 次
BF16 fill/`22.733 ms`，且 H4.5 已证明改为未初始化 output 会破坏
结果。真实 int fill 理想消除的得分上限约 `+0.000055`；连同
BF16 output clear 全部理想消除也仅约 `+0.00423`。warmup 的
int64 `cu_seqlens` 改为 int32 可避免首次请求重复冷调优，但对
计分为零。因此 Fill/H12 明确 no-go：不禁用 cache flush，不进入
production 或固定 `all 3` 测试。

H9 的现有 Q/K norm + RoPE fusion 在只读审计后不进入实现：Qwen3.5 的
full-attention Q 段按 head 交错保存 Q/output-gate，norm 是 Gemma
`(1+w)` 语义，positions 为 `[3,T]` 的 partial MRoPE64；现有 matcher/kernel
则要求连续普通 QKV、普通 RMSNorm、1-D positions 和整 head256 rotary，直接
启用既不会正确匹配，也不能保持语义。gfx936 上标准现有 kernel 虽可运行，
但 T=4096 约 `223.13 us`，加未融合 gate 后已不优于当前三 kernel 合计约
`261.39 us`；decode 乐观端到端上限仅约 `0.18%–0.35%`。因此本轮不翻全局
开关；若未来重开，必须是严格限定 Qwen3.5 gated-MRoPE 的新专用 op。

### GDN + normalized-BF16 GQA6 production：603 秒小样本 reject

组合 production 证据目录为
`/public/home/tangyu408/testdata/goal_runs/20260712_r26_gdn_gqa6_bf16_production`。
该栈在冻结 H11.5 + H10.8 上同时加入 GDN T=4096 四配置包和
normalized-BF16 GQA6 segmented decode。production runtime validation
总体 `passed=true`：GDN 覆盖 5 seeds 的 no-state/stateful 输出与
final state；GQA 覆盖 3 个随机长度×3 seeds、四类特殊值、AITER 与 dense
FP32 对照、repeat，以及 exact-hit/fallback/H11.5-prefill 路由。构建、安装
和 repo/build/wheel/site-packages identity 均通过；wheel SHA256 为
`dbe583515d9e7e9c00d2ac75c9455521173bd60eecfc98a8778eb09254195d83`。

固定 `run_throughput.sh all 3` 主体正常结束，外层窗口为
`1783858670–1783859273`、`603 s`、`status=0`，9/9 请求成功：

| 档位 | output throughput | mean TTFT | mean TPOT | 相对 R25 throughput | 相对 R25 TPOT |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4-8K | `12.931565111 tok/s` | `1613.659 ms` | `45.217007 ms` | `-0.131210%` | `+0.523718%` |
| 8-16K | `15.754052865 tok/s` | `3351.333 ms` | `45.687476 ms` | `-0.111872%` | `+0.135409%` |
| 16-32K | `9.991860103 tok/s` | `6218.938 ms` | `46.533518 ms` | `+1.037731%` | `-0.473064%` |

20/50/30 加权 throughput 仅 `+0.2291416%`，加权 TPOT 回归
`+0.0305292%`。逐条审计还发现 2/9 相对冻结 R25 输出漂移：8-16K 第 3 条
长度 `498→497`，16-32K 第 2 条长度 `259→263`；其余 7 条文本和长度一致。
这既没有形成可区分于噪声的性能收益，也没有保持小样本逐条输出，因此组合栈
明确 reject；服务已正常停止，BF16 GQA6 decode 不进入后继 production。

### GDN-only transition：小样本晋级、full/accuracy reject 与回退

GDN-only 证据目录为
`/public/home/tangyu408/testdata/goal_runs/20260712_r26_gdn_only_production`。
transition 精确恢复冻结 H11.5 attention backend
`2d038efffe89cec4e3101bd95505f520e1a49c753c77044adb35ec14b92e2c92`，
删除 `rocm_aiter_unified_attention_gqa6_decode.py`，同时逐 hash 保持六个
GDN 文件、H11.5 prefill 和 H10.8 native 栈不变。post static audit 通过，
wheel、build-lib 和 site-packages 中均不存在已拒绝的 decode 文件；新 wheel
SHA256 为
`469f83fc974c79eaf0042b5001c5e2e395bacd6fd3da6e1db89b1fac6ffd5466`。

transition 后最小 production-wrapper 数值门 `passed=true`：gfx936、真实
T=4096/int32 exact gate 两次命中；seed `20260712` 的 no-state/stateful
output 与 final state 相对同一 wrapper 强制 fallback 全部在
`atol=5e-4,rtol=5e-3` 下通过。它与前一组合周期对相同六个 GDN 文件完成的
5-seed 门禁共同确认 GDN 改动；attention backend 因按字节恢复冻结基线，
不再携带 BF16 decode 候选。

固定 `run_throughput.sh all 3` 主体正常结束；外层窗口为
`1783859822–1783860446`、`624 s`、`status=0`，服务和 models 审计均为
HTTP 200，9/9 请求成功：

| 档位 | output throughput | mean TTFT | mean TPOT | 相对 R25 throughput | 相对 R25 TPOT | 输出审计 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4-8K | `12.954590848 tok/s` | `1615.897 ms` | `45.013637 ms` | `+0.046615%` | `+0.071600%` | 3/3 文本、长度一致 |
| 8-16K | `16.058930524 tok/s` | `3355.861 ms` | `45.602727 ms` | `+1.821197%` | `-0.050339%` | 第 3 条长度 `498→537`、文本漂移 |
| 16-32K | `9.893121789 tok/s` | `6232.187 ms` | `46.772369 ms` | `+0.039289%` | `+0.037796%` | 第 2 条长度同为 `259`、文本漂移 |

20/50/30 加权 throughput 为 `+0.9317080%`，加权 TTFT 变化
`-0.1793132%`，加权 TPOT 回归仅 `+0.0004890%`。三档吞吐均为正、
runtime/route/health 门禁通过，且这是 R26 首个 production 端到端正数，
所以按筛选协议晋级 fixed full×3 和 accuracy，以检验跨轮重复性、SLA、
输出长度噪声及最终 `K`。但主要小样本增量来自 8-16K 漂移后的额外 39 个
output tokens，另有一条同长度文本漂移；因此 `+0.9317080%` 只是晋级信号，
不是已接受的可计分收益。small audit 后立即执行了 fixed full×3 和 accuracy。

三次 fixed full `all` 的窗口分别为 `2347 / 2324 / 2324 s`，每轮
`status=0`、每档 50/50 成功，合计 `450/450`、failed=`0`：

| 轮次 | 4-8K throughput | 8-16K throughput | 16-32K throughput | 最终综合分 |
| --- | ---: | ---: | ---: | ---: |
| run1 | `19.574725732385` | `16.556590105919` | `12.548636950442` | `87.836326412592` |
| run2 | `19.562743160014` | `17.049593698172` | `12.547661542843` | `88.278637194248` |
| run3 | `19.563605452953` | `17.048709058448` | `12.549713716311` | `88.279255341242` |
| 均值 | `19.567024781784` | `16.884964287513` | `12.548670736532` | `88.131406316028` |

三档的 output length 和 generated text 在三轮之间逐条一致。三轮 global
request TPOT P99 为 `47.175881430 / 47.164198942 / 47.168228130 ms`，均低于
冻结上限 `107.705131741 ms`；三档最大 TTFT P99 为
`1957.654316 / 15813.356886 / 6540.883634 ms`，分别低于
`7188.718023 / 37329.278019 / 43111.256068 ms`，所以完成率、TTFT 和 TPOT
SLA 全部通过。输出的跨轮稳定性不能改变性能判断：三档 full 均值相对 R25
current-best 的 20/50/30 加权变化为 `-1.679840604%`，小样本正数没有在
full 中复现。

固定 accuracy 于 epoch `1783867502–1783868455` 完成，窗口 `953 s`、
`status=0`。权威全精度/固定脚本等价复算结果为：

| 数据集 | official baseline | GDN-only | `k_i` |
| --- | ---: | ---: | ---: |
| hotpotqa | `77.959706960` | `77.959706960` | `1.00` |
| gov_report | `32.961006236` | `32.875225385` | `1.00` |
| retrieval_multi_point | `100.00` | `100.00` | `1.00` |
| aggregation_keyword_aggregation | `100.00` | `100.00` | `1.00` |

aggregation 的 OpenCompass 原生顺序敏感分仍为 `0.00`，固定脚本等价
`Counter` 复算为 30/30、权威值 `100.00`；retrieval 同样由固定等价复算
确认 30/30。四项均得到 `k_i=1.0`，所以最终 `K=1.0`。三轮最终综合分即
`87.836326412592 / 88.278637194248 / 88.279255341242`，均值
`88.131406316028`。该均值低于冻结 current-best 的 `88.548455504015`，
相对 current-best 加权吞吐又为 `-1.679840604%`，因此 GDN-only 明确
reject；score `95` 和 current-best `+20%` 两个性能终止条件均未达到。

最终 audit JSON/Markdown SHA256 分别为
`b89363fc3391388f13d586e5fc93782cf6af80be3d9f153c983b8529dd58211a`、
`8fe8eda53e91730b1ffa01bc8f38d48eae27134774a5bb02336aedd04ecd6095`，
全部 eligibility checks 为 PASS。reject 后已撤销 GDN overlay；回退后的
tracked repo diff 与 R26 overlay 前及 R25 final tracked diff 经 `cmp` 按字节
一致，SHA256 均为
`b9983b7764c66d76eb976897d1143712b3464a4b086c7dc15f778eb924b87600`。
runtime 已重新安装冻结 H11.5 + H10.8 baseline wheel，wheel SHA256 为
`03568ba87ff64fd0a8aade299026d7ee78cbf40d9c1ed5884fb584250b2031f2`，
installed backend/H11.5/H10.8 native identity 复核通过。服务于 epoch
`1783868534–1783868542` 停止，记录 `health=000, alive=0`。

R26 start/deadline 为 `1783847801 / 1783869401`。只读终止审计已于 epoch
`1783869431` 完成，满足 `1783869431 >= 1783869401`；源码、runtime、服务
和时间条件均已闭环，R26 按 6 小时时间条件结束，不再启动新实验。最终
evidence manifest 共 `153` 项并已 `153/153` 校验通过，manifest SHA256 为
`58f721374ef9cb189eebee0a7c493ebf2334dc8149d4271ac754a3a428fb6723`。

## R27 八小时源码优化与矩阵单元探索

R27 start/deadline 为 `1783874322 / 1783903122`。唯一直接基线仍是冻结
H11.5 + H10.8：full x3 综合分均值 `88.5484555040153`、相对 R24 加权
吞吐 `+10.2361569769%`、accuracy `K=1.0`。终止条件为 score `>=95`、
相对 current-best 同口径性能再次超过 `20%`，或持续 8 小时。production
repo 在本节三个候选后仍 clean、HEAD `3754870`；未 build/install wheel、
未启动服务，也未修改固定脚本。

### Attention/GDN 矩阵单元动态与 ISA 证据

证据目录为
`/public/home/tangyu408/testdata/goal_runs/20260712_attn_gdn_tensor_unit_probe`。
gfx936 的矩阵 datapath 在 profiler 中记作 MMOP，实际主要指令为
`v_mmac_f32_16x16x16_bf16`。动态 `SQ_INSTS_MMOP` 与 production HSACO
反汇编得到一致结论：

| 路径/kernel | `SQ_INSTS_MMOP` | 结论 |
| --- | ---: | --- |
| Attention prefill `kernel_unified_attention_2d_gqa6`，q512/seq12000 | `20,686,848` | QK 与 PV 均使用 MMAC |
| Attention decode `kernel_unified_attention_3d`，q1/seq12000 | `384,000` | stage 使用 MMAC |
| Attention decode `reduce_segments` | `0` | 普通 VALU reduction |
| GDN prefill KKT/solve/recompute/delta-h/output | 每个实际 dispatch 均 `>0` | 主矩阵链全部使用 MMAC |
| GDN decode packed recurrent core | `0`，同时 VALU `258,048` | 当前不使用 MMAC |

GDN prefill 的 cumsum、L2 norm、gating 和 causal-conv production code
objects 均没有 matrix opcode；它们是 scan/elementwise/stencil，不应仅为提高
MMAC 覆盖率而矩阵化。Attention/GDN core 的 Triton `tl.dot` 直接编译为
MMAC，不经过 rocBLAS；外围 QKV/O、qkvz/out 等 `F.linear` 则走 production
默认 hipBLAS/rocBLAS。R27 因此把后续问题拆成两类：对已有 MMAC kernel
减少冗余 dot、padding 和 LDS/HBM wait；对 Linear 记录实际 rocBLAS
kernel/solution/workspace 与 MMOP 利用，而不是再次切换 backend preference。

GDN packed decode 的“改写为 MMAC”还完成了确定性上限筛选，证据目录为
`/public/home/tangyu408/testdata/goal_runs/20260712_r27_gdn_decode_mmac_feasibility`。
当前 `B1/H16/HV48/K128/V128/BV32` 每 token 读写 786432 个 FP32 state
元素，仅 state 流量即 6291456 B；decay、两个 matvec 和 rank-1 update 的
最低核心计算为 5505024 FLOP，算术强度 `0.875 flop/B`。把
`h@[k,q]` 放入 16-column FP32 MMAC 仅有 2/16=12.5% 有效 N lanes，会使
核心 FLOP 变为 5x；rank-1 outer 放入 K8 MMAC 也仅 1/8=12.5% 有效，
两者同时矩阵化为 7x FLOP。该 kernel 在 fixed full trace 只占 `0.472%`，
即使免费删除的 Amdahl 上限也仅 `+0.47424%`，低于 1% production 门。
因此结果为 `reject_before_compile_static_upper_bound`，不为提高 MMOP 覆盖率
启动编译或短测。

### H11.10 H11.5 LDS/MMAC layout：编译/资源/PMC no-go

证据目录为
`/public/home/tangyu408/testdata/goal_runs/20260712_r27_h11_10_pv_transpose_probe`。
旧 generic attention2d 的 `28.134%` bank-conflict 不能直接外推，因此先对
当前 H11.5、BF16、Q24/KV4/D256、page784 的 q512/seq12000 shape 重采 PMC：

| counter | H11.5 |
| --- | ---: |
| `SQ_INSTS_LDS` | `54,989,376` |
| `SQ_WAIT_INST_LDS` | `122,376,261` |
| `SQ_LDS_BANK_CONFLICT` | `93,189,120` |
| `SQ_LDS_ADDR_CONFLICT` | `323,232` |
| `SQ_INSTS_MMOP` | `20,686,848` |

`mmac_layout_force=-1/1` 生成相同 legacy MMAC；`2/3/4` 生成 gfx936 汇编器
不支持的 `lit/lts` 字段而在 HSACO compile gate 失败。`llvm-iglp-8` 的最终
AMDGCN/HSACO 与 baseline 按字节一致；其它 schedule hint 编译失败。
`mmac5-ds6/ds10` 保持 `216 VGPR/32768 B LDS/0 spill`，但 bank conflict
不变，DS6 的 LDS wait 仅降 `0.531%`，DS10 反增约 `3.23%`。

等价 `V^T @ P^T` 候选的 12/12 数值 case finite/repeat-exact，最大差
`<=0.0009765625`。但 2-warp 为 `256 VGPR/167 spill`，4-warp 为
`221 VGPR`，均过不了冻结资源门；8-warp 虽为
`176 VGPR/32768 B/0 spill`，bank conflict 降 `33.35%` 的同时 LDS wait
增至 baseline `2.51x`、MMOP 增至 `1.50x`。所有候选都在预注册的
编译/资源/PMC 门失败，因此不需要制造一个低信息量 600 秒计时；H11.10
no-go，不改 production。

### H4.6 `delta_h + chunk_o` strict fusion：可实现但无可信性能上限

证据目录为
`/public/home/tangyu408/testdata/goal_runs/20260712_r27_h4_6_delta_o_fusion_probe`。
对 exact `B1/T4096/Hg16/H48/K128/V128/BT64`，同一 CTA 内按 chunk 顺序
执行 output 和 recurrent update，可以不改变 state dependency 地消除全局
`h` 与 `v_new`。BV16/BQ16 网格为 384 CTA，在 80 CU 上约 4.8 waves，
理想 tail 96%；可删除 144 MiB allocation、288 MiB global write+read 和
一条相邻 launch。

代价是每个 V tile 重算 QK：相对当前 BV64 output kernel，QK FLOP 变为
`4x`，output 总 FLOP 变为 `2.2x`。BV32/64/128 虽减少重复计算，却只有
192/96/48 CTA，tail 与 live-state 风险更差；现有 delta kernel 本身已达
`194 VGPR/36864 B LDS/0 spill`。独立原型因此只完成 source/syntax 和
字节/FLOP/grid 上限审计，没有 JIT、launch 或短性能测试。结论为 strict
fusion 暂不进 production；后继改为在现有 `chunk_fwd_o` 内让同一 Hg 的
三个 value heads 共享一次原始 QK，以减少重复 MMAC 而不串行 recurrent
time 维度。

### H4.7 `chunk_fwd_o` Hg3 grouped-head QK hoist：600 秒 kernel pass

证据目录为
`/public/home/tangyu408/testdata/goal_runs/20260712_r27_h4_6_hg3_qk_hoist_probe`。
exact `B1/T4096/H48/Hg16/K=V=128/BT64` 候选保持 BV64 与 V-tile grid，
只把 head grid 从 H48 改为 Hg16；同一 CTA 保留 raw FP32 QK，依次处理三个
具有各自 g/h/v 的 value heads。选中配置为 BQ64/BK64/BV64、4 warps、
1 stage、MI16/kpack2，资源 `156 VGPR/8192 B LDS/0 spill`，HSACO 使用
`v_mmac_f32_16x16x16_bf16`。

五 seeds×no-state/stateful 的十组数值门均 mismatch=0，最大绝对差
`6.103515625e-05`，candidate repeat-exact；长测前 Hg3 与同配置 ungrouped
输出位级一致。正式窗口同时随机交错 production current、同配置 ungrouped
control 和 Hg3，运行 `600.002450 s / 182282 groups`：

| mode | production | same-config ungrouped | Hg3 | Hg3 vs production | Hg3 pure structure |
| --- | ---: | ---: | ---: | ---: | ---: |
| no-state median | `0.524639 ms` | `0.488000 ms` | `0.416320 ms` | `+20.646%` | `+14.689%` |
| stateful median | `0.524479 ms` | `0.487999 ms` | `0.416319 ms` | `+20.622%` | `+14.689%` |

两 mode 的 p90 reduction 也均为正，预注册门通过。preflight JSON 中早先
21-repeat latency 字段少于 600 秒，已明确标记 `protocol-invalid`，未用于
config 选择、晋级或本结论；权威性能数据只有上述 600 秒窗口。

静态模型表明 QK FLOP 降 `66.67%`、output 总 FLOP 降 `26.67%`，q+k read
从 192 MiB 降至 64 MiB，major IO 减少 128 MiB；这说明优化是减少冗余 MMAC，
不是增加无效 matrix instructions。R23 的 `chunk_fwd_o` 占 full trace
`2.362%`，把实测 20.634% kernel 收益折算后仅约 `+0.487%`，纯结构约
`+0.347%`；这只是 raw trace 排序启发式。进一步按正式 all3/full 输入重建
4096-token chunks 后，20% exact-kernel 降时只预测 weighted throughput
`+0.15041%/+0.10535%`；即使 exact kernel 免费也仅
`+0.75711%/+0.52944%`，full score 上限 `88.683218`。因此 standalone
候选有效但不够进入 production/build/all3，只作为以后与其它独立大收益候选
组合时的可消融项。正式上限与接入审计见同目录 `production_upper_bound.md`。

### H4.8 其余 GDN grouped-head 去重：static no-go

证据目录为
`/public/home/tangyu408/testdata/goal_runs/20260712_r27_h4_8_remaining_hg3_static_audit`；
全程未 import/compile/launch/timing。KKT 若按 Hg 共享 K，可把 grid
3072→1024、K read 48→16 MiB，但 `beta_h` 在 dot 前按 value head 相乘并
回舍 BF16，移动到 MMAC 后会改变严格语义；数学上可省的 2.147 GFLOP 不能
作为等价候选。recompute_w_u 和 delta_h grouping 分别只省 32/128 MiB K
read，无法消除各 value head 独立的 MMAC；delta 还会从 192 降到 64 CTA，
三 state+三 BF16 v 的 live 下界 60 KiB，而当前已达
`194 VGPR/36 KiB LDS`。solve 没有 Hg/K/Q 共享。

免费 endpoint 上限分别为 KKT `0.853%`、recompute `0.702%`、delta
`1.508%`、solve `0.731%`；KKT+solve 整段免费也仅 `1.597%`。
recompute+delta 免费之和虽为 `2.183%`，但 Hg3 结构不删除其中任何 MMAC，
且 delta 并行度/资源门失败。因此 `next_candidate=null`，不建 JIT。

### D3 local4 decode workspace：600 秒 performance reject

证据目录为
`/public/home/tangyu408/testdata/goal_runs/20260712_r27_d3_decode_fusion_probe`。
候选把四个相邻 S32 fine ranges 合到一个 stage CTA，stage grid 从
`(1,4,32)` 变为 `(1,4,8)`，并用确定性的 8-part reducer。FP32 workspace
从 792576 B 减为 198144 B，即 `-75%`。

compile/numeric gate 覆盖 context `4096/6315/13295/21562/32768`，全部
finite/repeat-exact，相对冻结 FP32 S32 最大绝对差 `0.00048828125`；stage
为 `216 VGPR/32768 B LDS/0 spill`，reducer 为
`29 VGPR/4096 B LDS/0 spill`。

正式性能脚本每 sample 执行 50 次完整 wrapper，逐轮随机打乱 AITER、S32、
local4 与三个目标长度，运行 `600.028949 s / 4995 rounds`。GPU 中位数为：

| context | AITER | S32 | local4 | local4 vs S32 | local4 vs AITER |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6315 | `269.382 us` | `213.133 us` | `220.864 us` | `-3.63%` | `+18.01%` |
| 13295 | `269.117 us` | `213.136 us` | `283.699 us` | `-33.11%` | `-5.42%` |
| 21562 | `270.579 us` | `214.249 us` | `435.104 us` | `-103.08%` | `-60.80%` |

20/50/30 加权后，local4 相对 S32 的 GPU/wall reduction 为
`-48.2899% / -48.3089%`，相对 AITER 为
`-17.4123% / -17.5083%`，`status=reject_performance`。75% workspace
节省不足以抵消 producer CTA 从 128 降至 32 后的并行度损失，且串行四个
fine ranges 的惩罚随 context 增长。该结果排除继续减少 producer CTA 的
workspace-only 变体；不应用 production patch。

### D3 S20 CU-balanced follow-up：数值通过、资源门 reject

隔离目录为
`/public/home/tangyu408/testdata/goal_runs/20260712_r27_d3_decode_fusion_probe/s20_standalone`。
S20 保持 frozen S32 的 logical/padded64、4 warps、页映射、FP32 partial 和
确定性 reducer，只把 producer grid 改为 `(1,4,20)`，即 80 CTA；compact
workspace 为 495360 B，比 S32 少 37.5%。reducer 使用合法的 padded32
register tile，但全局地址和归约都 mask `lane<20 && lane<active_segments`。

纯 stdlib verifier 未 import torch/Triton/HIP，已证明五个 context 的
64-token tiles exact-cover、无 gap/overlap，tile-loop AST 与审计过的 S32
body hash 相同，compact stride20 与 identity lanes 正确。早期 CPU 表曾把
`waves_per_eu=1` 错当成一 CTA/CU 硬上限，并据此给出 S20 对 S32 的
critical-depth 优势；后续 HSACO descriptor/sysfs 审计证明它只是最小 occupancy
hint，32 KiB LDS 在 64 KiB/CU 上允许两 CTA/CU，故该性能模型已撤回。
exact-cover 与地址/identity 静态证明仍成立，但它们不预测 S20 更快。

Hg3 释放设备后按授权执行了只含 compile/resource/numeric 的 GPU preflight，
目录内没有性能 runner，`performance_timing_present=false`。五 contexts×三
seeds、zero/bounded-extreme/NaN 分类和 repeat 门全部通过；普通 case 相对
S32 最大绝对差不超过 `0.00048828125`。reducer 为
`38 VGPR/4096 B LDS/0 spill`，但 stage 为
`217 VGPR/32768 B LDS/0 spill`，超过该候选事前冻结的 `<=216 VGPR`
资源门 1 个。因此 `status=reject`，没有进入 600 秒计时。即使 general D3
协议允许到 224，也不能事后覆盖本候选更严格的资源判据；后继只能先解释
VGPR allocation/occupancy 粒度。离线解码显示 217→220、216→216 allocated
VGPR，二者在 32 KiB LDS 限制下仍是相同 CTA residency；这说明自设门过严，
却不能事后改写结论。且旧 S20 调度优势模型已撤回，因此不再启动 S20r1。

### H10.16 prefill rocBLAS solution-index：600 秒确认正数，单项不进 production

证据目录为
`/public/home/tangyu408/testdata/goal_runs/20260712_r27_prefill_linear_gemm_solution_audit`。
phase-labelled trace 已把六个 n4096 大投影绑定到真实 rocBLAS MMAC kernel；
固定 all3 输入中 4096-token full chunks 占输入 token 的 `86.74997%`。现有
one-dispatch PMC 为 `SQ_INSTS_MMOP=22282240`，kernel name 也明确包含
`MAC_MMAC/ISA936`，因此本项优化的是实际 solution，不是确认是否使用矩阵单元。

DTK 的 beta `rocblas_gemm_ex_get_solutions` 在 32/64/128 MiB workspace 下，
对六个 shape 均返回完全相同的 1319 个正 ID（19537--20982）；auto/default
`solution_index=0` 只是 sentinel，公开 API 不能把它反查成某个正 ID。
生产 32 MiB 口径的 7914 个 solution-shape one-shot correctness checks 全部
status OK、mismatch=0、nonfinite=0。该阶段无 timing/ranking。

正式 discovery runner 交叉 auto control + 1319 IDs + 六 shapes，共 7920
组合，随机交错记录 HIP event 与 wall。它自然完成
`1727.0307 s / 39600 samples`，每组合恰好 5 个样本，终端记录
`complete=true`。raw JSONL SHA256 为
`ab9b36082e469fd894371c4a50c44e85edfed54ed2ffaf4ebf7ec641a878d5ea`。

首次排名只在窗口完成后执行，并用实际降时
`1-candidate/auto`，不用 `auto/candidate-1` speedup 放大百分比。逻辑
shape 各自选 ID 会让 GDN out 与 full-attention out 对同一 TunableOp
签名分配不同结果，因而不可部署。按五个 exact key 重分组后，
shared out key 以 `48+16` 层联合权重选择 ID 20981，整个可部署
discovery portfolio 的 GPU/wall 降时为
`7.394655% / 7.331939%`。它超过 all3 前门 `5.813563%`，但低于
full endpoint `12.361010%`，严格为 discovery-only。

新 seed `2704097` 的独立确认只包含 auto 和入围 ID
`20981/20980/20979/20846`，仍使用 32 MiB workspace 与单个 >=600 秒
随机窗口。它自然完成 `600.000075 s / 286646 samples`，30 个
组合每个有 `9554--9555` 样本，`complete=true`。raw/analysis SHA256
分别为
`496cfde18de1260dc052938211cb7e57fe8b5906df36fed752fe52c3897538b9` /
`67146e284707b90c4caf2c7282d04e8178e296329e94b13938f755924ed33bd0`。

最终 gate/up=20981、down=20979、GDN qkvz=20981 和 attention qkv=20981
通过 GPU>=3%、wall>=2% 和上下半窗口同向门。shared out 的 20981
联合 GPU 降时为 `2.843%`，低于 3% 门，所以 GDN/full-attention out
共同保留 auto。四个可部署 explicit keys 的确认 GPU/wall 降时为
`5.786412% / 5.718477%`，相比 discovery 的 `7.394655% / 7.331939%`
发生了预期的 winner's-curse 缩水。保守值比 all3 `5.813563%` 前门
低约 `0.0951 pp`，也远低于 full `12.361010%` 端点门。因此 H10.16
不单独写 TunableOp CSV、build 或运行 all3；只作为已独立确认的
可消融项进入精确组合上限审计。

与运行中样本隔离的纯 stdlib 上限审计已冻结 16 个输入哈希。六类
M4096 投影按真实 layer 次数的时间权重依次为 MLP gate/up
`45.4621%`、MLP down `25.4187%`、GDN qkvz `16.2808%`、GDN out
`5.8753%`、full-attention qkv `4.9843%`、full-attention out `1.9788%`。
要预测 all3 加权吞吐 `+1%`，portfolio 必须使这些投影时间下降
`5.813563%`；full throughput `+1.5%` 则需 `12.361010%`。严格
weighted mean-E2E `-1.5%` 需 `12.570410%`。单靠本 portfolio 到
score 95 需不可能的 `266.818153%` 降时；即使六类完全免费，
score 也只到 `91.410759`。

TunableOp 路径静态审计还发现，六个逻辑来源只有五个可部署签名：
GDN out 与 full-attention out 共用
`tn_5120_4096_6144_ld_6144_6144_5120`，不能在 CSV 中按来源选不同
solution。未映射签名在 tuning 关闭时可回退 Default，但 CSV 命中了当前
runtime 没有注册的 ID 会硬失败，因此后续必须用当前 validators、
32 MiB workspace 和 graph-capture canary 二次确认，不能只写 CSV。

### H10.17 non-4096 tail-M rocBLAS：静态上限 no-go

纯 stdlib 审计证据目录为
`/public/home/tangyu408/testdata/goal_runs/20260712_r27_h10_17_tail_m_rocblas_static_audit`。
它冻结 17 个输入哈希，从固定 all3/full 请求长度重建实际 tail-M，
未 import torch/HIP/rocBLAS，未读取 H10.16 的进行中确认样本。

all3 有 9 个 tail/2304 次投影 GEMM；full 每轮 150 个 tail/38400 次，
三轮共 115200 次。tail 在六类投影时间中的加权 share 为 all3
`17.2694%`、full `18.1855%`；使用测量上界时分别为
`20.5285% / 20.9829%`。即使所有 tail 投影全部免费，乐观的
all3/full 加权吞吐上限也只有 `+3.6590% / +2.1248%`；要单靠
tail 达到 full `+1.5%`，需整体降时 `71.0618%`，高侧时间估计也需
`58.8905%`。

首轮按权重选出的六个 exact key 是 M=`2231/2004/1674/3478/3452`
的 MLP gate/up 以及 M=2231 的 MLP down。它们即使全免费，只预测
all3 `+1.4060%`、full `+0.0682%`。因此在完整 >=600 秒性能枚举
之前达到静态上限失败条件，H10.17 no-go；不为少数 exact tail keys
占用 GPU，转向审计已有独立 600 秒证据的可消融组合。

### H10.18 H10.16 + D3 S32 + Hg3：可消融组合静态必要门通过

纯 stdlib 组合审计证据目录为
`/public/home/tangyu408/testdata/goal_runs/20260712_r27_h10_18_composable_bundle_static_audit`。
它仅读取已完成并冻结 SHA256 的 H10.16 confirmation analysis、D3 S32
600 秒结果和 Hg3 600 秒结果；未读 raw confirmation JSONL、未初始化 GPU，
也未把组合上限写成实测结论。

模型按每条固定请求的绝对毫秒节省重建：H10.16 对每个 M4096
full chunk 减去 confirmed wall `35.685664 ms`；D3 使用
`(output_len-1)*16` 次 decode 调用与各档 S32-vs-AITER wall delta；Hg3
使用首个 no-state 和后续 stateful exact chunks 的实测 delta，各乘 48 个
GDN 层。三项毫秒相加后才重算每请求 duration、TTFT、TPOT 和吞吐，
没有直接相加独立百分比。

| 口径 | H10.16 | D3 S32 | Hg3 | bundle |
| --- | ---: | ---: | ---: | ---: |
| all3 weighted throughput | `+0.987656%` | `+1.196218%` | `+0.142432%` | `+2.355221%` |
| full x3 weighted throughput | `+0.690220%` | `+1.466368%` | `+0.099763%` | `+2.279951%` |

bundle 在 all3 三档的上限依次为
`+2.094986% / +2.292951% / +2.632493%`，full 三档为
`+2.090351% / +2.237514% / +2.477081%`，因此 all3 `+1%`、full
`+1.5%` 与组合 `>2%` 三个静态必要门均通过。full 预测 weighted
TTFT/TPOT 降幅为 `3.055022% / 1.931013%`，score 仅从
`88.548456` 到 `89.170302`，仍不接近 95。

结论是允许一次 controlled production build/canary，而非已可晋级：三条 route
需要独立 env gate 和 fresh-process 消融；TunableOp 需证明 4 hit + shared
out miss-to-Default 且 graph capture 成功；S32/Hg3 需 exact hit、fallback、数值与
resource 复验。全部前门通过后才可运行一次固定 `all 3`，外窗口
必须自然达到 600 秒；未达加权 `+1%`、任一档为负或输出漂移即回退。

### H10.18 production：build/route pass，661.94 秒 all3 因稳定输出漂移 reject

三项实施为独立 fail-closed env gate。TunableOp 加载 wheel 内四行 CSV，
gate/up、GDN qkvz 和 attention qkv 映射 ID 20981，down 映射 20979；
shared out 不在 CSV 中。S32 固定 logical/padded64、segments32；Hg3 只覆盖
exact `B1/T4096/Hg16/H48/K=V=128/BT64`。S32/Hg3 默认为关，只有
`{1,true,yes,on}` 开启，未知值也 fail closed。

combined wheel SHA256 为
`f877d08fdf2380a87298006c915d14077ca947225e50e5bcf56e028fc9075d80`，
`build_py --force`/wheel/force-reinstall 均成功。repo、build/lib、wheel 和
site-packages 的六个关键 Python/CSV 四方按字节一致。安装态数值/
资源复验中，S32 三 contexts×3 seeds 与 Hg3 三 seeds×no/stateful
全部 mismatch=0/repeat-exact；S32 stage `216 VGPR/32 KiB/0 spill`，Hg3
`156 VGPR/8 KiB/0 spill`，HSACO 分别含 128/76 个静态 MMAC/MFMA 指令。

Tunable 安装态 canary 观测到恰好 4 个 `ResultEntry found`、shared out
`no result, using default`，五个零输入 `F.linear` 输出 finite/形状正确，
online tuning/record 均关闭。fresh combined route service 还证明 init/pre-capture、
graph-finished、四个 H10 hit、shared miss、S32/Hg3 HIT 均命中；三个长
prompt 都返回固定 `ROUTE_OK_ROUTE_OK`。该阶段没有性能字段。

正式小样本使用另一 fresh confirmed/1/1 service，固定
`./run_throughput.sh all 3` 连续完成 3 个完整 active rounds，窗口
`661.937321805 s`、idle padding=0、27/27 requests failed=0，固定脚本前后 SHA
一致。相对冻结 H11.5+H10.8 all3：

| 档位 | baseline tok/s | candidate 3-round mean | 相对变化 |
| --- | ---: | ---: | ---: |
| 4–8K | `12.948555` | `13.084737` | `+1.0517%` |
| 8–16K | `15.771697` | `15.920351` | `+0.9425%` |
| 16–32K | `9.889236` | `10.189651` | `+3.0378%` |

20/50/30 加权改善为 `+1.59294973%`，三轮分别为
`+1.57313% / +1.60257% / +1.60315%`；各档均为正，TTFT 与 pooled TPOT
SLA 都通过。但两条输出在三轮中稳定漂移：8–16K index2
长度 `498→497`且文本 hash 改变；16–32K index1 `259→265`且文本 hash
改变。因此 `paired_input_output_length_and_text_exact=false`，硬门优先于吞吐正数，
结果是 `status=fail`、`advance_to_full=false`、`rollback_required=true`。
没有运行 full 或 accuracy。

候选服务已停止并确认 8001/runtime clean。S32/Hg3 与 TunableOp 源码分别
使用冻结补丁、仅经 `apply_patch` 精确撤销；工作树已恢复 clean HEAD
`3754870`。测量时 candidate wheel/site 身份已单独冻结，不与回退后的
repo 身份混用。安装态随后回装 pinned baseline wheel `03568ba8…`。

### H10.19 targeted output attribution：Hg3 归因 reject，C100 性能未达 1% 门

P9 只对 H10.18 已知漂移的两条请求做 fresh-state 输出归因。C000 的
retry0–5 全部属于 harness/contract-only 调试，没有产生两条满足同一
exact-replay 合同的可比输出，因而不得纳入 state comparison。特别是 retry5
直接发送 JSONL raw prompt，没有经过历史 benchmark 中
`CustomDataset.sample(skip_chat_template=false)` 的客户端预渲染；该单条返回
不是 `custom + openai-chat` 历史请求，已判无效且不作为 baseline。

retry6 按冻结 `REQUEST_CONTRACT.json` 先构造 benchmark rendered prompt，再把它
作为 `user` content 发送给 `/v1/chat/completions`，从而复刻客户端预渲染加
服务端二次 chat template 的实际请求路径。candidate wheel、H10/S32/Hg3
均关闭的 C000 得到 `status=pass`，两条 `frozen_baseline_exact_match=true`：

| 请求 | benchmark client input tokens | request payload SHA256 | server `usage.prompt_tokens` | completion / frozen text SHA256 |
| --- | ---: | --- | ---: | --- |
| 8–16K index2 | `10196` | `990c3aa5fba4fa045932e8c5e0217acca699bfdabcabbacf85bcde480c2cfacc` | `10207` | `498` / `63f45927cfa19bd66bc19b8e7146974de0e0d476d9f0c09afb4b354c981de09a`，exact |
| 16–32K index1 | `22294` | `5d6f3d217c5c9cb04888fc6076511d8301afe135d875e9d7e62cead1413f74cd` | `22305` | `259` / `729dd61de7b35eea3d821e657d89729d9fe436c5ce48d9a6a184c6aa0234fffe`，exact |

`benchmark client input tokens` 来自客户端 `SampleRequest.prompt_len`，服务端
usage 来自最终 engine prompt token ids；二者不是同一计数域，不要求相等。
跨态硬门改为：每个后续 fresh state 的两条 `request_payload_sha256` 必须分别与
C000 相同，且 `usage.prompt_tokens` 必须分别保持 `10207/22305`。任一 payload
或 usage 不一致均先归类为 harness/contract failure，不比较输出；两项都相等后，
才用 completion token 数和完整文本 SHA256 做 feature attribution。

C000 retry6 的完整证据在
`/public/home/tangyu408/testdata/goal_runs/20260712_r27_p9_targeted_output_attribution/02_C000_retry6`，
其中 output canary 与 route-marker audit 均为 PASS，source/runtime guards 前后一致。
C101（H10/S32/Hg3=`1/0/1`）通过同一 payload/usage 合同和 route audit：H10
四个 expected key 均命中，Hg3 HIT，S32 ABLATION/FALLBACK。但两条输出均未
恢复 baseline：

| state / 请求 | completion tokens | 文本 SHA256 | baseline exact | C111 exact |
| --- | ---: | --- | --- | --- |
| C101 / 8–16K index2 | `517` | `5847dbdaf03979d40d54823b98110f278fd47c7c0667e41c36d9ec3f17333484` | false | false |
| C101 / 16–32K index1 | `265` | `549e21adb28dfea6769941404eb19ab000830afae945492f6125b7768ace441e` | false | true |

随后 fresh C100（`1/0/0`）保持两个 request payload SHA、服务端
`usage.prompt_tokens=10207/22305`、wheel 和 H10 profile 全部相同，仅关闭 Hg3。
H10 四个 expected key 仍命中，Hg3/S32 均走 fallback；两条输出恢复为
`498/63f45927…` 和 `259/729dd61d…`，与冻结 baseline 完整文本和长度 exact。
因此 C101 与 C100 的服务级差分把漂移定位到 Hg3。独立只读微核矩阵进一步
观察到 Hg3 BK64 在 `5 configs × 5 seeds × 2 modes` 的 `40/40` case 中相对
production baseline 非 bitwise exact，candidate repeat-exact/allclose，最大绝对差
`6.103515625e-05`；它解释了为何 standalone 数值门能通过，却仍可能改变长文本
greedy decoding basin。

C100 correctness 通过后，使用 candidate wheel、H10 profile on、S32/Hg3 off
的另一 fresh performance service；`PYTORCH_TUNABLEOP_VERBOSE` 和误拼
`VEROBSE` 均 unset，tuning/record 置 0。固定 `./run_throughput.sh all 3`
连续完成三个完整 round，自然活动窗口 `664.215310295 s`、idle padding=0：

| 档位 | frozen baseline tok/s | C100 三轮均值 | 相对变化 |
| --- | ---: | ---: | ---: |
| 4–8K | `12.948554862` | `13.056761833` | `+0.835668%` |
| 8–16K | `15.771696925` | `15.886377602` | `+0.727130%` |
| 16–32K | `9.889236403` | `10.030269961` | `+1.426132%` |

27/27 requests completed、failed=0，全部逐请求 input/output length 和文本 hash
exact；三档均无回归。三轮最大 TTFT P99 为
`1811.759/3761.892/6348.377 ms`，低于 paired 1.5x 上限
`2775.973/5796.599/9774.263 ms`；最大 pooled TPOT P99 为 `47.751407 ms`，
低于 `71.536465 ms`。但 20/50/30 weighted 仅 `+0.958538041%`，小于冻结
all3 screen 的 `+1%` 门，所以 audit `status=fail`、`advance_to_full=false`；
不运行 full/accuracy。该 fail 只来自性能晋级阈值，不来自输出、SLA、完成率或
单档回归。

性能服务停止后，route/profile audit 确认 H10 INIT/PRE_CAPTURE ready、profile
SHA 正确、S32/Hg3 fallback 且无 HIT、`ResultEntry found=0`、
`Finding fastest=0`、无 Traceback。按 `rollback_required` 回装 pinned baseline
wheel SHA256 `03568ba87ff64fd0a8aade299026d7ee78cbf40d9c1ed5884fb584250b2031f2`；
site/direct_url、clean HEAD source、8001/runtime 和 HCU use `0.0%` 最终复验通过。

状态证据根：

- `/public/home/tangyu408/testdata/goal_runs/20260712_r27_p9_targeted_output_attribution`
- `/public/home/tangyu408/testdata/goal_runs/20260712_r27_p9_c100_fixed_all3_600`

### H10.20 M4096-scoped TunableOp feasibility：只读 NO-GO

C100 的 weighted throughput 为 `+0.958538041%`，距离冻结 `+1%` 门为
`0.041461959 pp`。本项只检查是否值得通过“仅在 M4096 prefill model forward
开启 TunableOp、tail/decode 关闭”回收 untuned lookup；没有使用 GPU、修改源码、
安装包或服务状态。

PyTorch/vLLM 调用路径给出四项约束：

1. C100 实测服务设置了 `PYTORCH_TUNABLEOP_ENABLED=1`。PyTorch 对该环境值做
   首次读取缓存，且环境值优先于 Python setter；因此原部署不能按 step 关闭，
   future patch 必须取消该环境固定值，只通过 API 控制。
2. `TunableOp::operator()` 在 enabled 状态下构造 op/params/BLAS signatures，并
   进入 `TuningResultsManager::Lookup`；manager 带 mutex。disabled 状态直接选择
   Default，可以绕过该 lookup。
3. `TuningContext.enable_` 是非 atomic 普通 `bool`，setter 是全局 byte store。
   因此逐 step 切换只可能在当前 TP/PP/DP=1、单 model-execution thread 下配合
   owner-thread/非重入断言使用，不能宣称通用线程安全。
4. 当前 CUDA graph capture 覆盖小尺寸 decode model；graph replay 不会逐层再次
   执行 TunableOp lookup。step gating 对 decode 的主要可回收部分是 graph 外
   `compute_logits`/M1 lm-head，而不是已捕获的全部 model GEMM。

历史 route canary 在 application ready 后共有 1728 hits、1643 misses。1008 个
miss 位于 M4096 forward 内部，整个 M4096 forward 开启后仍保留；两个 tail 的
608 个和 M1 lm-head 的 27 个可以关闭，故实际只删除
`635/1643=38.65%`。按固定 all3 每轮 9 tail、1135 decode output tokens 外推，
约删除 3871 次 lookup/round。结合三档实测 duration 和 miss 分布，达到门槛需
约 `8.7 us/removed lookup`；当前没有这样的调用成本证据。C100 三轮改善为
`0.937260/0.972017/0.966338%`，其波动跨度与 0.041462 pp 缺口相近，直接做
一个长窗无法给出足够的预期裕量。

判定为 **NO-GO**：不 build/install，不启动性能实验。未来最小 patch 仅 defer：

- loader 取消 `PYTORCH_TUNABLEOP_ENABLED` 固定值，profile 加载完成后默认 disabled；
- 在 `gpu_model_runner.execute_model()` 的 `_model_forward` 周围仅当实际
  `num_tokens_padded == 4096` 时 enable，并以 `finally` 恢复 disabled；
- `_dummy_run(4096)` 使用相同 scope 做 solution warmup，小尺寸 graph capture
  保持 disabled；
- 继续限制 TP/PP/DP=1，并增加 owner-thread 和非重入 fail-closed 断言。

只有独立测量证明 untuned miss 成本至少约 `10 us/call`，且预测增益明显超过
三轮噪声，才值得重新申请 `>=600 s` 窗口。

### R28 / P10a：R28-A 正式性能 NO-GO

R28-A 已通过 compile、correctness 和 resource 前门。两个 exact M4096 MLP shape
的四个 config 均确认使用 MMAC，资源均为 `0 spill`；32/32 candidate 数值 case
通过 BF16 allclose、finite、repeat bitwise 和 zero exact。

正式单一自然随机交错窗口完成 `600.0428034258075 s`，共 `8664` 个完整
balanced sweeps、`103968` 个单 GEMM samples，每个 shape/method 为 `8664`
samples。窗内没有性能早停。结果：

| shape | control auto | H10 explicit | best Triton | vs auto | vs H10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| gate/up | 4.373435 ms | 4.102876 ms (`20981`) | 6.320472 ms (`aiter_256x256x64`) | `-44.5196%` | `-54.0498%` |
| down | 2.452157 ms | 2.302397 ms (`20979`) | 3.047676 ms (`aiter_256x256x64`) | `-24.2855%` | `-32.3697%` |

双 shape 按冻结层权重合并的 reduction 为 `-37.2634376%`，未达到
`+15.3439%` 门；driver 在完整窗后以 status `43` 正常给出 performance reject。
因此 P10a/R28-A 最终 NO-GO：不修改 production source，不 build/install，不启服，
不重扫相同 blocked GEMM config。证据根为
`/public/home/tangyu408/testdata/goal_runs/20260712_r28_static_candidate_screening`。
post-run repo 仍 clean HEAD `3754870`，pinned baseline wheel `03568ba8…` 四方验证
PASS，8001/runtime 为空，HCU use `0.0%`、memory use `0%`。

### R28-B / P10b：H11.5 P/V LDS physical XOR swizzle 资源 NO-GO

证据目录为
`/public/home/tangyu408/testdata/goal_runs/20260712_r28_b_lds_swizzle_preflight`。
该项没有修改 production repo、build/install wheel、启动服务或采集 latency。

静态审计先排除了 source token-offset 方案。统一 `t'=t XOR c` 必须同时重排
scores、causal mask 和 V，因而会改变 softmax sum 与 P@V 的浮点 reduction 顺序；
按 feature 维变化的 `t'=t XOR f(d)` 则会在一个 QK score 内混入不同 token，
数学上错误。合法候选只能位于 TTIR 逻辑张量之下，把 P/V 的物理 LDS slot 做
可逆 XOR，而保持逻辑 token 顺序不变。

冻结 H11.5 TTGIR SHA256 为 `6f1fb485…bee61b5`。其 P `[64,64]` 和
V `[64,256]` dot operand 原为 `vec=1/maxPhase=1`。R28-B 使用
`vec=4/perPhase=1/maxPhase=16`，令 `phase=t&15`，BF16-word 物理地址为：

```text
L_P(q,t) = 64*t  + (q XOR (phase << 2))
L_V(t,d) = 256*t + (d XOR (phase << 2))
```

`local_alloc/local_load` 共享同一 memdesc，XOR 自反，因此全局 K/V offset、
coalescing、两个逻辑 `tt.dot`、softmax IR、MMOP 个数和 token 累加顺序均不变。
三份 exact TTGIR 只分别修改 P、V 或 P+V 的 shared encoding；stdlib static gate
确认 diff 仅为 `2/2/4` 行，installed Triton 的 `.ttgir` IRSource 会跳过 TTGIR
passes 后继续编译。

owner 授权的 compile/correctness/resource front gate 不含 event 或性能字段。
三候选均通过 IRSource compile，保持两个 logical dot、128 个静态
`v_mmac_f32_16x16x16_bf16`、32768 B LDS 和零 spill，但首个 same-input launch
暴露的资源为：

| candidate | VGPR | spill | LDS | front decision |
| --- | ---: | ---: | ---: | --- |
| P-only | 222 | 0 | 32768 B | `>216` resource reject |
| V-only | 232 | 0 | 32768 B | `>216` resource reject |
| P+V | 217 | 0 | 32768 B | `>216` resource reject |

结果为 `status=rejected_before_pmc`、`survivors=[]`。correctness matrix 只产生
用于物化 runtime resource metadata 的首行，未宣称候选数值通过；PMC 和性能
均未启动。P+V 的 217/216 近失配也不允许事后放宽冻结门。该 exact
`vec4/maxPhase16` 分支最终 NO-GO，不做 production patch/build/install。
post-run 无 R28-B/rocprof 进程，HCU0 memory use `0%`。

B2 对 217-VGPR P+V near miss 只做三种新的 both-operand physical encoding：

| B2 candidate | static MMAC | VGPR | spill | LDS | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `vec4/perPhase1/maxPhase8` | 128 | 225 | 0 | 32768 B | resource reject |
| `vec2/perPhase1/maxPhase16` | 128 | 233 | 0 | 32768 B | resource reject |
| `vec8/perPhase1/maxPhase8` | 128 | 219 | 0 | 32768 B | resource reject |

三项 exact TTGIR 和 diff/hash 冻结于 `PREPARED_B2.json`。IRSource compile 与
assembly 门均通过，但首次 untimed resource-discovery launch 后全部超过 `<=216`
硬门，故 `front_gates_b2.json` 为 `rejected_before_pmc/survivors=[]`。39-case
correctness 未运行，PMC/latency/performance 均未启动；不以 219/216 近失配放宽门。
post-run 8001 closed、无相关 runtime、HCU use `0.0%`/memory `0%`，repo 仍为
clean HEAD `3754870`。

### R28-C / P10c：exact BK32 Hg3 group2+1 前门 NO-GO

证据目录：
`/public/home/tangyu408/testdata/goal_runs/20260712_r28_c_hg3_bk32_group2plus1_preflight`。
本项只创建 evidence-local kernel/probe，没有修改 production repo、安装 wheel、
启动服务或运行任何性能计时。

候选使用两个 launch。pair kernel 对每个 Hg group 的 value heads 0/1 共享一次
raw QK；singleton kernel 只写 head 2，并保持原 single-head BK32 数学。每组三头
从 9 个 dot work unit 降到 8，只删除一份重复 QK；所有 K reduction 均为 BK32，
两个 code object 均包含 MMAC，静态 MMAC 合计也不低于单 baseline code object。

5 seeds×no-state/stateful 的完整 bitwise/repeat 与资源矩阵如下：

| pair config | exact / repeat | pair resource | singleton resource | matrix | front gate |
| --- | --- | --- | --- | --- | --- |
| `w2/s2` | `10/10 / 10/10` | `227 VGPR, 16 KiB, 0 spill` | `117 VGPR, 12 KiB, 0 spill` | pair 240 + singleton 144 static MMAC | resource FAIL |
| `w4/s1/mi16/kpack2` | `0/10 / 10/10` | `130 VGPR, 8 KiB, 0 spill` | `117 VGPR, 12 KiB, 0 spill` | 104 + 144 static MMAC | bitwise FAIL |
| `w8/s1/mi16/kpack2` | `0/10 / 10/10` | `76 VGPR, 8 KiB, 0 spill` | `117 VGPR, 12 KiB, 0 spill` | 56 + 144 static MMAC | bitwise FAIL |

resource-valid 两项在全部 10 cases 中 deterministic non-bitwise，最大绝对差
`6.103515625e-05`。唯一 bitwise-exact 配置比冻结 `<=216 VGPR` 门高 11。
因此 `passing_configs=[]`、decision=`NO_GO_FRONT_GATE_FAILED`，按要求没有继续
config 搜索，也没有 short/event/600s timing。

组合动机是 H10-only `+0.958538041%` 尚缺 `0.041461959 pp`。旧 Hg3 模型线性
折算要求未来 exact candidate 至少约 6.0% `chunk_fwd_o` reduction；group2+1
纸面筛选约 `+0.0712 pp`，但它不是实测，且本候选已在 correctness/resource
前门失败，不能进入 H10 组合。post-run port 8001 closed、无相关进程，HCU use
`0.0%`、memory use `0%`。

### R28-D / P10d：both-shared perPhase 最终资源前门 NO-GO

证据目录为
`/public/home/tangyu408/testdata/goal_runs/20260712_r28_d_lds_perphase_preflight`。
该项只延续 R28-B P+V `217 VGPR` near miss，预注册
`vec4/pp2/mp8`、`vec4/pp2/mp16`、`vec4/pp4/mp8` 三项 both-shared TTGIR。
按 `phase=(t//perPhase)%maxPhase`，同一 memdesc 使用自反 XOR 物理映射，
不改变 source token offset、逻辑 P/V、dot、softmax 或 token 累加顺序；8/16 个
bank shift 的理想 alias dispersion 为 `87.5%/93.75%`，只表示仍有理论
`>=20%` conflict 降幅空间，不是 PMC 结论。

三项均编译为 128 BF16 MMAC、0 spill、32768 B LDS，但分别为
`223/233/229 VGPR`，全部超过冻结 `<=216` 门。因此 `survivors=[]`，在
bitwise correctness、PMC 和性能前停止；不放宽门，不做 production
patch/build/install。post-run 无 R28-D/rocprof 进程，HCU0 memory use `0%`。

### R28-E / P10e：BQ32/BK32 group2+1 正式性能 NO-GO

证据根为
`/public/home/tangyu408/testdata/goal_runs/20260712_r28_e_hg3_bq32_bk32_group2plus1_preflight`。
R28-E 通过两个串行 BQ32 q tiles 将 R28-C exact pair 从 227 降到 201 VGPR；
unchanged singleton 为 117 VGPR。passing pair/singleton 均 `<=32 KiB LDS/0 spill/MMAC`，
5 seeds×两 mode 为 10/10 bitwise/repeat exact。

预授权静态表曾给出 major IO/FLOPs `-7.7821%/-13.3333%`，故授权恰好一个正式窗。
窗后源码复核发现 major IO 漏计 pair 第二 q tile 的 H/V 重载；修正 H/V 为
`160/80 MiB` 后 candidate total 为 `451.5 MiB`，相对 baseline 385.5 MiB
增加 `17.1206%`。冻结原表保留，独立 CORRECTION 覆盖其 IO 字段。harness 冻结 explicit raw
BK32 production baseline、R28-E pair、unchanged singleton、preflight/static/
identity/idle hashes；claim 只在 compile/warmup/pre-bitwise 和 256 MiB flush
allocation 成功后领取。

随机平衡窗自然完成 `600.000565983 s`，188864 个完整 sweep、每 cell 188864
samples、总计 755456；changing-value 256 MiB flush 次数同为 755456，且位于
event 外：

| mode | baseline median | candidate pair+singleton median | median reduction | baseline/candidate p90 | p90 reduction |
| --- | ---: | ---: | ---: | --- | ---: |
| no-state | `0.434399009 ms` | `0.584478974 ms` | `-34.5489%` | `0.436639994/0.587998986 ms` | `-34.6645%` |
| stateful | `0.434240013 ms` | `0.584478974 ms` | `-34.5981%` | `0.436479986/0.587998986 ms` | `-34.7138%` |

两 mode pre/post bitwise 均 exact，但 median `>=6.01%` 和 p90-not-slower 都失败。
独立审计复算一致，修正流量模型也解释性能回归；最终 `NO_GO_PERFORMANCE`。没有 production patch、build/install
或服务；post-run 8001/runtime/GPU clean。

### R28-F / P10f：剩余 non-MMOP 边界离线静态 NO-GO

证据根为
`/public/home/tangyu408/testdata/goal_runs/20260712_r28_f_static_mmac_boundary`。
该项只读筛选 GDN BT64×H48 triangular-cumsum MMAC、width-4 depthwise
causal-conv block-diagonal MMAC，并排除 Attention reducer 与 GDN L2 norm；
没有 import/compile/GPU/PMC/timing 或 production 修改。

F1 dense triangular MMAC 相对 scalar scan 的 MAC/add 比约 65x，而当前 cumsum
整 kernel 免费端点仅 `+0.030009%`，低于 H10-only 尚缺的 `0.041461959 pp`。
F2 的 block-diagonal causal-conv 只有 6.25% 乘积有效、至少增加 16x 计算；
prefill fwd 整 kernel 免费端点也只有 `+0.370367%`。Attention reducer/L2 norm
免费端点分别为 `+0.078061%/+0.084071%`，且矩阵有效率至多约 1/16。
因此 R28-F 为 offline static NO-GO，不申请 `>=600 s` 窗。

### R28-G / P10g：BQ16/BQ32 Hg3 group3 GPU 前静态 NO-GO

证据根为
`/public/home/tangyu408/testdata/goal_runs/20260712_r28_g_hg3_bq16_bk32_group3_front_gate`。
候选只复用冻结 Hg3 QK-hoist 源码，预注册 BQ16 primary 与 BQ32 companion，均为
`BK32/BV64/w2/s2/GROUP_SIZE3`；没有新增其它 config，也没有改 production。

源码循环审计确认 K/H/V 加载位于 qtile 循环内。相对 explicit raw baseline
`385.5 MiB`，BQ16 四 qtiles 的逻辑 major I/O 为 `791.5 MiB`
（`+105.3178%`），BQ32 两 qtiles 为 `436.5 MiB`（`+13.2296%`）。两者 FLOPs
均为 `11,811,160,064`，相对 baseline `-26.6667%`。从 R28-E 已测回归恢复并达到
`+6.01%` 需最少 `30.1443%` candidate latency reduction；BQ32 相对 R28-E
修正流量/FLOPs 的极乐观相加上界仅 `18.7069%`，仍不足。

因此 R28-G 在 GPU 前判为 `STATIC_NO_GO_BEFORE_GPU_FRONT_GATE`。未执行 probe
草稿已删除；没有 GPU init、compile、launch、bitwise/repeat、resource/MMAC 或
performance timing，没有申请 `>=600 s` 窗。`FROZEN_INPUTS.json` SHA256 为
`7266dc0149df72821d0533f600bb6e1c395e5a69400875fe9f54d1579557fc7a`。

### R28-H / P10h：C110 S32 exact-replay correctness NO-GO

P9 历史矩阵虽然列出 C110，但实际只留下 C000/C101/C100 证据；从未对
`H10=confirmed,S32=1,Hg3=0` 做 exact-replay 或独立性能。R28-H 使用 frozen
candidate wheel `f877d08f...` 启动一次 fresh output-only service，复用 P9 的
两条冻结请求。两条 payload SHA 与 `usage.prompt_tokens=10207/22305` 完全一致。

route marker audit 为 pass：四个 H10 key 全部命中；S32 hit、无 fallback/ablation；
Hg3 fallback、无 hit。尽管路由和请求合同成立，8–16K index2 输出为
`572` tokens、SHA `4655f859...`，16–32K index1 为 `289` tokens、SHA
`6cb59007...`，两条逐请求 `frozen_baseline_exact_match` 都是 false。

随后 C010 关闭 H10，仅保留 S32 on、Hg3 off。其 route 同样无 mismatch：H10
disabled/0 hit、S32 hit/no fallback、Hg3 fallback/no hit；两条请求逐字节复现
C110 的 `572/4655f859...` 与 `289/6cb59007...`，usage 仍为 `10207/22305`。
因此漂移不依赖 H10×S32 交互，S32 单独充分。

`output_canary.json` 顶层 `status=pass` 只表示旧 harness 成功完成请求；其代码只
对 C000 强制 baseline control，不能解释为 C110 correctness pass。逐请求 exact
字段是硬门权威。因此 C110 在 performance 前直接 `NO_GO_CORRECTNESS`，不运行
`>=600 s all 3`、full 或 accuracy。服务停止后 repository source 和 port/runtime
均 clean；最终回装并验证 pinned baseline wheel `03568ba8...`，candidate-only
installed files 均 absent。证据根：
`/public/home/tangyu408/testdata/goal_runs/20260712_r28_h_c110_revival`。

### R28-I / P10i：S32 bitwise-exact salvage 静态 NO-GO

证据根为
`/public/home/tangyu408/testdata/goal_runs/20260712_r28_i_s32_exact_salvage_static_audit`。
本项只读检查旧 validator/result、S32/AITER 源码树、R26 historical matrix、C010
输出和模型 layer contract；没有 GPU、compile、launch、timing 或 production 改动。

旧 validator 使用 `atol=0.015/rtol=0.01/max_abs<=4.8828125e-4`。“zero tolerance
mismatches”是超该非零容差的元素数为零。九条 S32 records 全部
`allclose=true,mismatch_count=0,repeat_exact=true`，但九条
`numeric.exact=false`，最大绝对差为 `2.44140625e-4/4.8828125e-4`。所以旧证据
从未证明 candidate/reference bitwise。

AITER 的 exact runtime 树是 `M16/token tile16/segments16`；selected S32 为
`M8/tile64/segments32`，且更换 MI16/kpack2 codegen，stage 内 QK/P@V 与
online-softmax 边界、最终 reducer 均不同。历史 closest `M8/L16/S16` 虽有
`185 VGPR/8 KiB LDS/0 spill` 和 weighted GPU `+16.2191%`，十条 case 仍全部
non-bitwise，不能作为 exact clone 的收益证据。

唯一可信 exact salvage 是 literal AITER 或保持其全部树与 codegen 的 clone。
literal fallback 收益为 0；specialized clone 只可能删除很小 generic control，且
无静态正上界、codegen 改变本身仍可能破坏 exact。双算 S32+AITER 再返回 AITER
约 `483 us vs 269 us`，更慢。因此 `STATIC_NO_GO`；未来必须先以
`atol=rtol=0,numeric.exact=true` 和 frozen service exact replay 过门。
`EVIDENCE.sha256` SHA256 为
`0bbf67e37fafbb024438033937b363f39c66c0a44956d9a5140eca76e23cbc36`。

### R28-J / P10j：raw-A 两 kernel GPU 前 bandwidth 静态 NO-GO

证据根为
`/public/home/tangyu408/testdata/goal_runs/20260712_r28_j_raw_a_two_kernel_static`。
本项没有 kernel 实现或 GPU 行为：只按 exact
`B1/T4096/H48/Hg16/K=V128/BT64/BV64` 数据流计算 producer、workspace 与
consumer 的 FLOPs、major I/O、CTA、launch 和 allocation。

producer 用 `(64 chunks,16 Hg)=1024` CTAs 计算一次 raw QK，写
`[64,16,64,64]` FP32 workspace，即 16 MiB。consumer 仍按三 value heads×两个
BV64 tiles 执行 QH/A@V，因此其 `(2,64,48)=6144` CTAs 共读取 `A_raw` 六次，
即 96 MiB。精算结果为：

| quantity | explicit raw baseline | R28-J | delta |
| --- | ---: | ---: | ---: |
| QK/QH/A@V FLOPs | `6.442451/6.442451/3.221225 GF` | `1.073742/6.442451/3.221225 GF` | total `-33.3333%` |
| major I/O | `385.5 MiB` | `433.5 MiB` | `+48 MiB / +12.4514%` |
| CTA | `6144` | `1024+6144=7168` | `+16.6667%` |
| launch | 1 | 2 | +1 |
| temporary workspace | 0 | 16 MiB FP32 | +16 MiB |

candidate I/O 分项为
`Q112+K16+H96+V48+O48+G1.5+Awrite16+Aread96=433.5 MiB`。因此 QK 去重的
计算上界不能通过保守带宽门。作为经验交叉检查，R28-E measured ratio 已为
`1.34549x` baseline；恢复并达到 `+6.01%` 需相对 E 至少 `30.1443%` 降时，
而 R28-J 相对修正后 E 的 FLOPs/I/O 纸面改善即使直接相加也只有
`23.0769%+3.9867%=27.0636%`，还未扣除第二 launch 和 allocation。

最终决定为 `STATIC_NO_GO_BANDWIDTH_LAUNCH_WORKSPACE`：没有 GPU init、compile、
launch、correctness/resource/MMAC/PMC/timing，不申请 `>=600 s` 窗，不修改
production。`SUMMARY.md` 与 `STATIC_MODEL.json` SHA256 分别为
`faf9a067c41880352ce9247e5b5c1416f798d67a7dc2e5a01f63dd1de02f0fc3`、
`1a9b98f174cbe12e1d1f124bf2b3c5be4f32e08ed33bd3eddab92e6f75474300`。

### R28-K / P10k：DCU 矩阵单元覆盖最终综合

证据根：
`/public/home/tangyu408/testdata/goal_runs/20260712_r28_k_matrix_unit_coverage_synthesis`。
本项只读汇总 45 个已冻结 source/evidence inputs，没有 GPU init、compile、launch、
PMC、timing、服务或 production 修改。

最终覆盖矩阵如下：Attention Prefill `kernel_unified_attention_2d_gqa6` 动态
`SQ_INSTS_MMOP=20,686,848`，QK/PV reference object 有 128 BF16 MMAC；
Attention Decode `kernel_unified_attention_3d` 为 `384,000` MMOP，而
`reduce_segments` 为 `MMOP=0/VALU=22,752`。GDN Prefill 的 KKT、solve、
recompute、delta-h、output 每个实际 dispatch 都 MMOP-positive，production
objects 分别 28/28、13/13、10/10、24/24、37/37 含 MMAC；cumsum、L2 norm、
gating、causal-conv 则为 matrix opcode 0。GDN packed Decode 动态
`MMOP=0/VALU=258,048`，11/11 unique HSACO 也无 matrix opcode。

多 token Prefill Linear 的运行调用链是 `F.linear → hipblasGemmEx_v2 →
rocblas_gemm_ex → Tensile ISA936 MAC_MMAC`；one-dispatch PMC 有 `22,282,240`
MMOP。`solution_index=0` 是 automatic sentinel；1319 个正 ID 已枚举。静态映射的
auto 等价正 ID 为 `20844/20845/20846/20838`，独立确认 portfolio 则让三个 key
使用 `20981`、MLP down 使用 `20979`，共享 out 留 automatic。其 wall reduction
`5.7185% < 5.8136%`，所以没有 production CSV。H10.8 的 `n=1,K=5120`
exact shapes 走 `LLMM1/LLMM1Strided`，是 FP32 `fmaf`/vector FMA + wave reduction
的 GEMV 例外，不应与 Prefill GEMM 混写。

Decode 路径还校正为实际 import 的
`/usr/local/lib/python3.10/dist-packages/aiter/ops/triton/unified_attention.py`
（SHA `004d569a...`）：AITER 是 `M16/token-block16/S16`；S32 是
`M8/L64/S32`，stage 有 128 MMAC、reducer 为 0。旧容差 validator 的九条
`numeric.exact=false` 与 R28-H service 漂移一致，说明 MMAC 命中不能替代
归约树正确性。

R28 A-J 的统一边界是：已有 MMAC 候选分别在性能、资源、bitwise、流量或
correctness 门失败；scan/reducer/recurrent 等向量算子被强行矩阵化时又有
6.25%/12.5% 有效 tile、5x/7x FLOP 或不足 Amdahl 上界。因此后续只优先：
减少已有 MMAC 的重复 work/LDS/HBM wait；有足够裕量的新 Linear fusion/算法；
同时通过 bitwise、resource、完整 qtile traffic 的 GDN dataflow；以及先过
service exact-replay 的任何 reduction-tree 改动。

### R27/R28 八小时时间终止闭环

goal start/deadline 为 `1783874322 / 1783903122`。系统与目标双时钟在
`2026-07-13T00:38:55Z`（epoch `1783903135`）复核，目标计时为
`28813 s`；两者均证明已经持续至少 `28800 s`。最终可计分栈仍是冻结的
H11.5 + H10.8，full x3 综合分均值 `88.5484555040153`、accuracy `K=1.0`。
综合分没有达到 `95`，也没有出现相对 current-best 同口径性能严格超过
`20%` 的新合格结果，所以唯一满足的终止项是 8 小时时间条件。

本轮所有正式性能窗口均自然达到不少于 600 秒；资源、静态 I/O、bitwise、
correctness 或 profiler 前门失败的分支没有冒充性能测试。R28 没有候选接入
production；结束时 repo 保持 clean HEAD `3754870ff9a3cd4ab087307b8adda46980d78f09`，
pinned baseline wheel SHA256 为
`03568ba87ff64fd0a8aade299026d7ee78cbf40d9c1ed5884fb584250b2031f2`，
没有 commit/push，服务、8001 端口和 GPU 均为空闲。终止证据目录为
`/public/home/tangyu408/testdata/goal_runs/20260712_r28_8h_termination_audit`。
