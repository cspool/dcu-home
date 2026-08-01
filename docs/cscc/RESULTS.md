# 重构后最终评测结果

## 结论

`repro-minimal` 使用空 build tree 从源码构建 wheel，并在 fresh 单卡服务上运行
固定的全量性能与精度脚本。最终结果为：

- 性能分：`91.67227258084725`；
- 精度：`110/110` 请求完成，四项精度系数均为 1，最终 `K=1.00`；
- 综合分：`91.67227258084725`。

重构前权威稳态锚点为 `91.66283719585402`。本轮高
`0.00943538499322`（约 `0.0103%`），只能判定为性能等价，不能当作新增优化
收益。重构保留了当前最高性能，同时删除了不可达或已否决实验。

## 评测身份与契约

| 项目 | 值 |
| --- | --- |
| 直接代码基线 | OpenDAS `vllm_cscc` `fa718036bdb9dfd80a872b86c8ac16c9d02bfd31` |
| 目标 | gfx936，Qwen3.5-27B，BF16，单卡，TP/PP/DP=1 |
| 量化 / 投机解码 / prefix cache | none / disabled / disabled |
| 并发与请求率 | `MAX_CONCURRENCY=1`，`REQUEST_RATE=1` |
| 每档性能请求 | 2 条 warmup + 50 条计分请求 |
| 最大输出 | 1024 token |
| 性能脚本 SHA-256 | `91b72cfca61994c38f302d3735b3ab7f85481939364876bc1ce2b6496820749e` |
| 精度脚本 SHA-256 | `2e641672a45ac96318c2118df8df4dae2babf87c16afd49cbe4b037ff9beed4e` |
| 启动脚本 SHA-256 | `6f794f30e5c031a4d62ff8cc2ecaaa35162e79c23de897a023180ae70cc17f97` |

服务日志确认 `speculative_config=None`、`quantization=None`、
`enable_prefix_caching=False`，且 `tensor_parallel_size=1`、
`pipeline_parallel_size=1`、`data_parallel_size=1`。

## 全量性能

| 输入档 | 成功 | 输出 token | 时长（s） | tok/s | 计分分量 | TTFT P99（ms） | 请求 TPOT P99（ms） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4–8K | 50/50 | 12434 | 575.555717079 | 21.603468841 | 17.058632287 | 1800.471265 | 40.749717 |
| 8–16K | 50/50 | 13667 | 714.090929799 | 19.139019178 | 45.642771846 | 4024.934331 | 41.670475 |
| 16–32K | 50/50 | 11624 | 746.361036092 | 15.574232091 | 28.970868447 | 5242.875767 | 42.671636 |

三档 `150/150` 成功，`failed=0`。计分分量之和为
`91.67227258084725`；全局逐请求 TPOT P99 为 `42.662897898 ms`。

| SLA | 限制（ms） | 本轮（ms） | 结果 |
| --- | ---: | ---: | --- |
| 4–8K TTFT P99 | 7188.718023 | 1800.471265 | pass |
| 8–16K TTFT P99 | 37329.278019 | 4024.934331 | pass |
| 16–32K TTFT P99 | 43111.256069 | 5242.875767 | pass |
| Global TPOT P99 | 107.705132 | 42.662898 | pass |

8–16K 最大 TTFT 为 `4387.544623 ms`，没有出现历史偶发的约 33 秒冷编译
长尾。

三档结果 JSON 的 SHA-256：

| 输入档 | SHA-256 |
| --- | --- |
| 4–8K | `0e733be396faa8bbb4cead86b1a1246b4279140053460f034cdc81c4a0815413` |
| 8–16K | `70713083ede14492e8f87aabb9c7efef137e48a9965b725bf798763c8af162bb` |
| 16–32K | `2517de495e3bb122938d9ad9f3050445552b385b8eee5e0d1870b7ad1f1021fc` |

## 全量精度

| 数据集 | 请求 | 官方基线 | 本轮 | k |
| --- | ---: | ---: | ---: | ---: |
| HotpotQA | 20/20 | 77.959706960 | 77.959706960 | 1.00 |
| GovReport | 30/30 | 32.961006236 | 33.223915908 | 1.00 |
| Retrieval Multi Point | 30/30 | 100.00 | 100.00 | 1.00 |
| Aggregation Keyword | 30/30 | 100.00 | 100.00 | 1.00 |

总计 `110/110`，最终 `K=1.00`。GovReport 文件末尾没有换行，因此
`wc -l` 显示 29；JSONL 实际解析为 30 条，prediction 也为 30 条。

Retrieval 和 Aggregation 均按固定脚本使用 `Counter` 比较 prediction 与
gold 多重集合，独立复核为 `30/30`。Aggregation 的 OpenCompass 原生中间
`accuracy=0` 不支持列表多重集合口径，不是最终固定脚本结果。

精度 summary CSV SHA-256 为
`3191b178dcea4a44bbd73ac3bca83cbd49f3409530dbab383e8effe42663e081`。

## 重构非回归证据

重构前后使用冻结的三档最差样本执行了 22 次请求：

- `22/22` 成功；
- 输入长度、输出长度和生成文本 SHA-256 逐条一致；
- 相对重构前，加权请求速率 `-0.072%`、TPOT `-0.146%`，属于运行波动。

最终 full 的性能分相对重构前为 `+0.0103%`，再次确认性能等价。只有上述
worst-3 门禁声明输出 exact；本轮 full 以固定 accuracy 结果作为精度证据，
不把性能请求文本宣称为逐条 exact。

## 干净构建与冷启动

空 build tree 产物：

| Artifact | SHA-256 |
| --- | --- |
| wheel | `0d3e061765b1899771d2719acf190ed81b27f651645e1b0774eb5f1dd79158dd` |
| wheel 中 `_rocm_C` | `d7cc10b1aa7383e7a31f163e782b7a8af5a9ac3d708ebed4bca6d539fb71d0df` |
| TunableOp profile | `169c7b11a0340d9e22405327b5e5667b2aa9e9e8d899bd59e10ca4fb7fb52030` |

wheel 共 1989 个条目，不含 `__pycache__`、`.pyc`、`perf_trace` 或已删除
实验模块。预编译 wheel 不提交到仓库；评测机按
[闭卷复现手册](CLOSED_BOOK_REPRODUCTION.md) 从源码构建。

fresh cache 的 `torch.compile` 用时 `495.49 s`，初始 profiling/warmup
用时 `161.23 s`，graph capture 用时约 `22 s`。这些均发生在 health ready
之前，没有计入 TTFT。日志同时确认 5 条 profile 结果和 5 条 validator
加载成功，INIT 与 PRE_CAPTURE 均为 `status=ready`。

## 证据边界

性能、精度、服务日志和模型权重不提交到源码仓库；上文保留固定脚本、结果
JSON、summary 和构建产物的哈希用于核对。源码边界由
`evidence/manifests/repro_minimal_runtime.sha256` 固定，并可通过
`scripts/verify_cscc_repro.sh` 验证。
