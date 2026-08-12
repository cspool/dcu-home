# Qwen3.5-27B 精简优化版 DP=2 全量验证报告

## 1. 结论

提交 `5355ceab41a0801c611d5267ef6e36eefdcd4c77` 的隔离 wheel 已完成 DP=2 全量精度与吞吐验证，结果通过：

- 精度实际覆盖 110 条唯一样本，四个数据集全部完成；HotpotQA 为 77.96，GovReport 为 32.95，两个 RULER 数据集均为 100.00。
- 吞吐覆盖 3 个输入长度桶、并发 2/4/8，共 9 组；每组 50 个主请求、2 个 warmup，合计 450 个主请求，全部成功。
- 服务日志共有 580 个 `/v1/chat/completions` HTTP 200，非 200 为 0，traceback、assertion、OOM、worker 退出均为 0。
- 两个 worker 均选择 `ROCM_AITER_UNIFIED_ATTN`，均使用动态编译范围 `(1, 4096)`，不是之前无效记录中的 `TRITON_ATTN`。
- GQA6、page784、GDN Norm+SiLU 的正式服务 Triton 缓存均有运行时编译证据；K5120/GateUp+SwiGLU 通过隔离 wheel 的直接算子矩阵，原生扩展包含三个目标 kernel 变体。
- 测试完成后服务已停止，HCU0/HCU1 显存占用均恢复为 0%。

## 2. 被测对象与服务配置

| 项目 | 值 |
|---|---|
| 仓库分支 | `repro-gqa-page784-k5120` |
| 提交 | `5355ceab41a0801c611d5267ef6e36eefdcd4c77` |
| vLLM wheel | `vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl` |
| wheel SHA256 | `e23e033ed580dded4e9261f96a37ab62357f84e3e4f8d9617005e111cbe6391f` |
| `_rocm_C` SHA256 | `edf0ec0b42e1689adc021ccc2849b2b37da184b2b16ef3455d4300d69a1628d0` |
| 模型 | `/root/Qwen3.5-27B`，BF16 |
| 设备 | HCU0/HCU1，`gfx936:sramecc+:xnack-` |
| 并行 | DP=2，TP=1，`data_parallel_backend=mp` |
| Attention | `ROCM_AITER_UNIFIED_ATTN` |
| KV block | 784 tokens |
| 调度 | `max_num_batched_tokens=4096`，`max_num_seqs=128` |
| 编译 | 非 eager，动态范围 `(1, 4096)`，CUDA graph 开启 |
| KV cache | 每卡 25,872 tokens |
| GPU memory utilization | 0.95 |

构建来自提交的独立 detached worktree，wheel 安装到独立 `site` 后从源码树外导入。五个关键 Python 文件与安装产物逐字节一致。构建过程只在临时 worktree 中按发行流程改写了 `vllm/version.py` 的版本戳；主仓库工作树保持干净，且与 GitHub 跟踪分支为 `0 ahead / 0 behind`。

冷启动从首条 CLI 日志到两个 API server 就绪约 463 秒；两个引擎的 profile、KV cache 创建和 warmup 分别为 339.91 秒和 338.00 秒。

## 3. 精度结果

| 数据集 | 实际样本数 | 指标 | 得分 |
|---|---:|---|---:|
| HotpotQA | 20 | LongBench score | 77.96 |
| GovReport | 30 | LongBench score | 32.95 |
| Retrieval Multi Point | 30 | accuracy | 100.00 |
| Aggregation Keyword Aggregation | 30 | accuracy | 100.00 |
| 合计 | 110 | 请求成功/失败 | 110 / 0 |

官方脚本用 `wc -l` 显示 GovReport 为 29，是因为 JSONL 最后一行没有换行符；JSON 解析、OpenCompass 请求和预测文件均确认实际为 30 条。因此本次不是 109 条，而是完整覆盖 110 条唯一样本。GovReport 的奇数条数问题也不存在。

两个 RULER 分数采用官方脚本中针对预测列表的重算逻辑；OpenCompass 原始 Aggregation 汇总列不适用于该自定义列表格式。

## 4. 吞吐结果

共同条件：每组 50 个主请求，2 个不计时 warmup，`request_rate=inf`，`custom_output_len=1024`，`temperature=0`，`ignore_eos=False`，固定数据顺序。

| 输入桶 | 并发 | 成功/失败 | 时长(s) | 输出 tok/s | 总 tok/s | TTFT P50/P95(ms) | TPOT P50/P95(ms) | E2E P50/P95(ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4-8K | 2 | 50/0 | 360.55 | 34.95 | 910.74 | 1759.86 / 3222.52 | 46.15 / 76.53 | 6472.82 / 52452.30 |
| 8-16K | 2 | 50/0 | 406.05 | 33.17 | 1670.31 | 3321.00 / 4299.37 | 43.24 / 46.16 | 6544.67 / 47380.72 |
| 16-32K | 2 | 50/0 | 362.20 | 28.41 | 3004.95 | 5276.58 / 5459.08 | 44.49 / 44.70 | 12635.51 / 37806.24 |
| 4-8K | 4 | 50/0 | 228.44 | 54.90 | 1437.19 | 2670.78 / 3505.11 | 49.47 / 103.97 | 7451.14 / 67240.78 |
| 8-16K | 4 | 50/0 | 496.13 | 27.60 | 1367.49 | 7669.59 / 10904.21 | 52.67 / 412.19 | 16168.86 / 119867.41 |
| 16-32K | 4 | 50/0 | 430.57 | 23.71 | 2527.60 | 13577.32 / 15376.94 | 51.95 / 275.93 | 18368.87 / 86317.42 |
| 4-8K | 8 | 50/0 | 164.58 | 75.31 | 1993.91 | 3124.36 / 6903.18 | 73.20 / 197.20 | 11021.15 / 74753.10 |
| 8-16K | 8 | 50/0 | 298.48 | 45.64 | 2272.76 | 8248.26 / 18303.86 | 111.60 / 515.86 | 18807.28 / 126063.07 |
| 16-32K | 8 | 50/0 | 493.24 | 23.93 | 2209.68 | 15353.69 / 47604.37 | 229.58 / 1441.48 | 58833.16 / 159717.04 |

实际 token 范围为：4-8K 桶 4,226–8,024，8-16K 桶 8,297–16,017，16-32K 桶 20,518–22,391。服务端聊天模板增加约 10 tokens，因此 result JSON 中范围分别为 4,236–8,034、8,307–16,027、20,528–22,401。

## 5. 性能解读

- 4-8K：并发 8 最优，75.31 output tok/s；比并发 2 提升 115.5%，比并发 4 提升 37.2%。
- 8-16K：并发 8 最优，45.64 output tok/s；比并发 2 提升 37.6%。并发 4 的 27.60 tok/s 是 KV 驻留和长输出组合下的局部低点。
- 16-32K：并发 2 最优，28.41 output tok/s。并发 4/8 分别下降 16.5%/15.8%；并发 8 虽与并发 4 的输出吞吐接近，但 TTFT P95 增至 47.6 秒、TPOT P95 增至 1.44 秒，不应作为该长度的默认并发。
- 原因：长桶单请求约 21–22K token，而每卡 KV cache 只有 25,872 tokens；每卡只能稳定驻留一个该长度请求。继续提高客户端并发主要增加排队和抢占。
- `ignore_eos=False` 导致每组实际生成 token 数略有差异。因此选择并发时应同时看 output tok/s、total tok/s 和尾延迟，不能只看单一吞吐值。

与此前那个实际选择 `TRITON_ATTN`、只完成 c2 的无效记录相比，本次 c2 output tok/s 在三个桶分别变化 +0.7%、+49.1%、+176.4%。该比较只用于说明后端和实现差异，不能替代同后端消融。

## 6. 完成性与正确性审计

- 9 个 result JSON 全部存在；每个都包含 50 个 input length、output length、生成文本、TTFT 和 start time 明细。
- 每个 JSON 的 input/output token 数组求和均与汇总字段完全一致；`errors` 的 50 个槽位全部为空。
- 吞吐主请求共 450 成功、0 失败；warmup 共 18 个。
- 精度 OpenCompass 共发出并成功接收 110 个请求；四个预测文件条数为 20/30/30/30。
- 加上 2 个冒烟请求，服务日志 POST 计数为 `110 + 450 + 18 + 2 = 580`，全部 HTTP 200。
- 服务日志中非 200 为 0；`ERROR`、`Traceback`、`AssertionError`、`OutOfMemory`、engine/worker died 为 0。
- 两个 AITER worker、两个 `(1, 4096)` 动态编译记录、两个引擎完成记录、两个 API startup complete 均可见。
- 停机前 HCU0/HCU1 显存占用均为 95%；停机后均为 0%。

隔离 wheel 的算子矩阵状态为 `pass`：

- GQA6 对官方 FlashAttention 最大绝对误差 0.015625。
- page784 对官方 FlashAttention 最大绝对误差 0.00390625，并验证四种拒绝形状不修改输出。
- GDN Norm+SiLU 在 16/32/64/128/4096 token 上与参考结果完全一致，非目标 stride 正确回退。
- K5120 覆盖 M=96/14336/16384/34816/248320；GateUp+SwiGLU 输出形状为 17408，最大绝对误差 0.0625；不支持形状均正确回退。

正式服务 Triton 缓存已固化 2 个 `_gqa6_prefill` 变体、1 个 `_pack_page784`、2 个 `_merge_page784` 变体和 1 个 `_gdn_rmsnorm_silu_gate`。K5120 是 `_rocm_C` 原生 kernel，不进入 Triton 缓存；其三个模板变体和 `LLMM1` ABI 均在扩展符号表中，且直接算子矩阵已实际执行。当前实现没有 K5120 每调用计数器，因此端到端服务中的逐次命中由固定 Qwen3.5 decode shape 和接入条件推断，而不是日志计数。

## 7. 结果文件

- `summary.json`：机器可读摘要。
- `server/cold-start.log`：完整冷启动、服务和 HTTP 日志。
- `accuracy/full-accuracy.log`：最终四项精度结果。
- `accuracy/work/accuracy_debug/output/local_accuracy_qwen35/20260811_004810/`：OpenCompass 配置、110 条预测与评分。
- `throughput/results/*/result.json`：9 组详细请求级结果。
- `throughput/run_full_throughput.sh`：完整吞吐复现命令。
- `provenance/`：提交、artifact 哈希、GPU 状态、服务进程、后端选择、原生符号和运行时 kernel 证据。

