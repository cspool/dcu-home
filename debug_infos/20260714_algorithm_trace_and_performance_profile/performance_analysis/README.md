# 当前最佳版本性能分析数据说明

本目录只整理当前最佳性能版本 **H11.5 + H10.8（R25）** 的原始性能数据和原始分析报告。除本说明外，共 68 个文件，全部直接放在本目录；没有混入其他候选版本的全量结果，也不包含脚本、源码、服务日志、PID、wheel、HSACO、CO 或 fatbin。

归档过程只通过文件名前缀消除了原目录层级与重名，文件内容没有改写。建议先阅读 [`current_best_final_summary.md`](current_best_final_summary.md)，再按需要查看三轮原始结果和各组件 profiler 数据。

## 当前最佳结果

- 三轮固定 full 测试综合分：`88.490349137758`、`88.578483694186`、`88.576533680101`。
- 三轮均值：`88.5484555040153`；accuracy 系数 `K=1.00`。
- 相对 R24 的 20/50/30 加权吞吐提升：`+10.2361569769%`。
- 三轮合计 `450/450` 请求成功，`failed=0`，固定 SLA 全部通过。
- 这份证据确认 H11.5 + H10.8 是当前已完整验证的最佳版本，但原报告明确记录：它没有达到 90 分或相对 R24 `+20%`。

## 文件及用途

### 最终结论与版本身份（8 个）

| 文件 | 用途 |
| --- | --- |
| `current_best_final_summary.md` | 当前最佳版本的权威总报告：三轮吞吐、综合分、SLA、重复性、accuracy、服务闭环及结论。 |
| `current_best_worst_request_fixtures.md` | 固定测试中最差请求样本及 SLA 复核说明。 |
| `current_best_final_identity.txt` | 最终源码、wheel 和运行时身份一致性检查结果。 |
| `current_best_finalization_utc.txt` | 最终证据冻结时间。 |
| `current_best_final_wheel.sha256` | 最终 wheel 的 SHA256；用于唯一标识被测构建，不包含 wheel 本体。 |
| `current_best_final_wheel_stat.txt` | 最终 wheel 的文件大小、时间等 stat 信息。 |
| `current_best_git_head.txt` | 当前最佳构建对应的源码 Git HEAD。 |
| `current_best_rocm_native_extension_attestation.txt` | 已安装 ROCm native extension 与最终 wheel、目标 ABI marker 的一致性证明。 |

### Accuracy 数据（6 个）

| 文件 | 用途 |
| --- | --- |
| `current_best_accuracy_summary.csv` | OpenCompass 原生 accuracy 汇总表，适合程序读取。 |
| `current_best_accuracy_summary.md` | OpenCompass 原生 accuracy 汇总表的 Markdown 版本。 |
| `current_best_accuracy_summary.txt` | OpenCompass 原生 accuracy 文本输出。 |
| `current_best_accuracy_status.txt` | 固定 accuracy 测试退出状态；`0` 表示正常完成。 |
| `current_best_accuracy_window_start_epoch.txt` | accuracy 测试开始 epoch。 |
| `current_best_accuracy_window_end_epoch.txt` | accuracy 测试结束 epoch。 |

注意：OpenCompass 原生中间 summary 将 aggregation 记为 `0.0`；固定评分脚本按预测列表与 gold 多重集合等价重算为 `100.00`。最终计分应以 `current_best_final_summary.md` 中的四数据集结果和 `K=1.00` 为准。

### 三轮固定 full 原始吞吐数据（18 个）

`<run>` 为 `1`、`2`、`3`，`<band>` 为 `4-8K`、`8-16K`、`16-32K`。

| 文件模式 | 数量 | 用途 |
| --- | ---: | --- |
| `full_run<run>_<band>_result.json` | 9 | 每轮每档的原始 benchmark JSON；包含逐请求 latency、TTFT、ITL/TPOT、输出长度、生成文本、成功/失败数及汇总吞吐。 |
| `full_run<run>_status.txt` | 3 | 每轮固定 `run_throughput.sh all` 的退出状态。 |
| `full_run<run>_window_start_epoch.txt` | 3 | 每轮测试窗口开始 epoch。 |
| `full_run<run>_window_end_epoch.txt` | 3 | 每轮测试窗口结束 epoch。 |

这些 JSON 是重算三轮吞吐分、P99 SLA、请求成功率和输出重复性的主要原始数据。

### H10.8 gfx936 strided LLMM1 验证（14 个）

| 文件 | 用途 |
| --- | --- |
| `h10_8_runtime_validation.json` | H10.8 的正确性、重复性、负例拒绝及 31 组 timing 原始结果；`all_passed=true`。 |
| `h10_8_runtime_validation_status.txt` | runtime validation 退出状态。 |
| `h10_8_runtime_validation_start_epoch.txt` | runtime validation 开始 epoch。 |
| `h10_8_runtime_validation_end_epoch.txt` | runtime validation 结束 epoch。 |
| `h10_8_runtime_validation_script.sha256` | 产生 runtime validation 数据的脚本哈希，仅用于确认测试口径。 |
| `h10_8_compile_artifact_hashes.txt` | 单编译单元产物哈希清单。 |
| `h10_8_compile_gfx936_notes.txt` | 编译产物中的 AMDGPU note/metadata 原始文本。 |
| `h10_8_kernel_metadata.txt` | H10.8 目标 kernel 的 gfx936 资源和 ABI metadata。 |
| `h10_8_compile_status.txt` | 单编译单元构建状态；`0` 表示成功。 |
| `h10_8_compile_object_hash.txt` | 目标 object/code object 的哈希。 |
| `h10_8_compile_source_hashes_before.txt` | 编译前相关源码哈希。 |
| `h10_8_compile_source_hashes_after.txt` | 编译后相关源码哈希，用于检查采集期间是否漂移。 |
| `h10_8_compile_start_epoch.txt` | 单编译单元验证开始 epoch。 |
| `h10_8_compile_end_epoch.txt` | 单编译单元验证结束 epoch。 |

### H11.5、Attention 与 GDN profiler 数据（7 个）

| 文件 | 用途 |
| --- | --- |
| `h11_5_attention_lds.csv` | 当前 H11.5 attention kernel 的 rocprof LDS counter 原始数据。 |
| `attention_gdn_tensor_unit_report.md` | Attention/GDN 是否使用 gfx936 MMAC/MMOP 矩阵单元的完整分析报告。 |
| `attention_prefill_mmop.csv` | H11.5 full-attention prefill core 的动态 MMOP/VALU/wave counter。 |
| `attention_decode_mmop.csv` | attention decode stage 与 reduction 的动态 counter，可区分矩阵阶段和非矩阵 reduction。 |
| `gdn_prefill_mmop.csv` | GDN prefill 主矩阵链各 kernel 的动态 MMOP counter。 |
| `gdn_prefill_aux_mmop.csv` | GDN prefill 辅助采集的原始 CSV；此次为 `0 contexts`，不能作为“不使用矩阵单元”的动态证据。 |
| `gdn_decode_mmop.csv` | GDN packed decode core 的有效动态负证据：WAVES/VALU 非零、MMOP 为零。 |

这些 counter 用于回答是否发出矩阵指令，不能直接解释成整模型的矩阵单元利用率百分比。GDN prefill 数据包含 profiler 新进程触发的 Triton autotune 行，报告已单独说明其边界。

### Prefill GEMM profiler 数据（7 个）

| 文件 | 用途 |
| --- | --- |
| `prefill_tensor_unit_report.md` | BF16 prefill projection 经 hipBLAS/rocBLAS 并使用 gfx936 MMAC 的分析报告。 |
| `prefill_gemm_mmop.csv` | 代表性 `512x5120x34816` BF16 GEMM 的动态 MMOP/VALU/wave counter。 |
| `prefill_gemm_hipprof_trace.db` | 该代表性 GEMM 的 hipprof 原始 trace 数据库；这是数据文件，不是可执行文件。 |
| `prefill_gemm_hipprof_trace.json` | hipprof trace 的 JSON 导出。 |
| `prefill_gemm_hipkernel.csv` | hipprof kernel 汇总，含 kernel 名称、调用次数和耗时。 |
| `prefill_gemm_hiptrace.csv` | HIP API/dispatch trace 表。 |
| `prefill_linear_existing_evidence.json` | 当前生产 prefill linear/GEMM 路径的既有 trace 证据索引与摘要。 |

该微基准证明代表性大投影会使用矩阵单元，但不证明所有小型或 ragged prefill GEMM 都使用同一路径，也不提供整段 prefill 的利用率。

### H11.5 L4096 参考 profile（8 个）

| 文件 | 用途 |
| --- | --- |
| `h11_5_l4096_profile_result.json` | H11.5 在 `seq_len=4096`、24 Q heads、4 KV heads、head size 256 下的 20 次 event timing 原始结果；median 为约 `4.694 ms`。 |
| `h11_5_l4096_kernel_trace.csv` | 上述 micro-profile 的 rocprof kernel trace。 |
| `h11_5_l4096_device_before.txt` | profile 前设备状态。 |
| `h11_5_l4096_device_after.txt` | profile 后设备状态。 |
| `h11_5_kernel_code_object_elf_header.txt` | 对应 H11.5 code object 的 ELF header 文本。 |
| `h11_5_kernel_code_object_notes.txt` | code object 的 AMDGPU notes/metadata 文本。 |
| `h11_5_kernel_code_object.sha256` | 被分析 code object 的哈希，不包含 HSACO 本体。 |
| `h11_5_kernel_metadata.txt` | 精确 target/kernel metadata。 |

该 L4096 结果是独立 micro-profile（`model_loaded=false`、未使用官方请求），用于解释 H11.5 kernel 本身，不替代三轮 fixed full 端到端数据。

## 证据边界

- 当前版本的端到端结论以 `current_best_final_summary.md` 和 9 个 `full_run*_result.json` 为准；组件级 micro-profile 只解释具体 kernel/path。
- 本目录没有一份覆盖最终 wheel、全模型 prefill/decode 全阶段且带 phase 标签的统一 counter profile，因此不能据此给出各模块占总时延的精确百分比。
- 原始报告中的路径保留采集时的位置。归档已把本 README 所列的相关原始数据扁平复制到当前目录，但没有复制脚本、日志或二进制产物。
