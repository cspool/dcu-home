# 等价压缩后最终评测结果

## 结论

`repro-minimal` 使用空 build tree 从源码构建 wheel，并在 fresh 单卡服务上运行
固定的全量性能与精度脚本。最终结果为：

- 性能分：`91.080201612`；
- 精度：`110/110` 请求完成，四项精度系数均为 1，最终 `K=1.00`；
- 综合分：`91.080201612`。

压缩前现有 Triton 最优版的同脚本分数为 `91.676381851`；本轮低 `0.6503%`，
满足小于 `1%` 的性能等价门禁。三档请求 TPOT P99 均略低于压缩前；分数差异
主要来自本轮 16–32K 请求更早生成 EOS、计入的输出 token 数减少，而不是每
token 热路径变慢。本节不把运行波动声明为新增收益。

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
| 4–8K | 50/50 | 12525 | 577.941048691 | 21.671760517 | 17.112557080 | 1793.997339 | 40.648105 |
| 8–16K | 50/50 | 13386 | 700.912988034 | 19.097948288 | 45.544825905 | 4021.503712 | 41.580532 |
| 16–32K | 50/50 | 10933 | 715.528729605 | 15.279610095 | 28.422818627 | 5238.142234 | 42.585493 |

三档 `150/150` 成功，`failed=0`。计分分量之和为
`91.080201612`；全局逐请求 TPOT P99 为 `42.568376392 ms`。

| SLA | 限制（ms） | 本轮（ms） | 结果 |
| --- | ---: | ---: | --- |
| 4–8K TTFT P99 | 7188.718023 | 1793.997339 | pass |
| 8–16K TTFT P99 | 37329.278019 | 4021.503712 | pass |
| 16–32K TTFT P99 | 43111.256069 | 5238.142234 | pass |
| Global TPOT P99 | 107.705132 | 42.568376 | pass |

8–16K 最大 TTFT 为 `4383.284670 ms`，没有出现历史偶发的约 33 秒冷编译
长尾。

三档结果 JSON 的 SHA-256：

| 输入档 | SHA-256 |
| --- | --- |
| 4–8K | `b55509962bb1f509b52c76114190b119df554b77ae0f8651ecdafd3a1beceb36` |
| 8–16K | `4eaeae175c4917004522f35540c48b18f29aaf437ad71bae3fbab3e57e4f2452` |
| 16–32K | `a6c1d1647c4ee34e72641e99fc3046499fd8f49da54cd891e95b9ceb96d26d0d` |

## 全量精度

| 数据集 | 请求 | 官方基线 | 本轮 | k |
| --- | ---: | ---: | ---: | ---: |
| HotpotQA | 20/20 | 77.959706960 | 77.96 | 1.00 |
| GovReport | 30/30 | 32.961006236 | 33.15 | 1.00 |
| Retrieval Multi Point | 30/30 | 100.00 | 100.00 | 1.00 |
| Aggregation Keyword | 30/30 | 100.00 | 100.00 | 1.00 |

总计 `110/110`，最终 `K=1.00`。GovReport 文件末尾没有换行，因此
`wc -l` 显示 29；JSONL 实际解析为 30 条，prediction 也为 30 条。

Retrieval 和 Aggregation 均按固定脚本使用 `Counter` 比较 prediction 与
gold 多重集合，独立复核为 `30/30`。Aggregation 的 OpenCompass 原生中间
`accuracy=0` 不支持列表多重集合口径，不是最终固定脚本结果。

精度 summary CSV SHA-256 为
`3a28b4d1eac037cbf7fc7360a38456808c65fbc9da7860539005d3e0cffebe3e`。

## 重构非回归证据

重构前后使用冻结的三档最差样本执行了 22 次请求：

- `22/22` 成功；
- 输入长度、输出长度和生成文本 SHA-256 逐条一致；
- 相对重构前，加权请求速率 `-0.072%`、TPOT `-0.146%`，属于运行波动。

早期重构 full 的性能分相对重构前为 `+0.0103%`，再次确认性能等价。只有上述
worst-3 门禁声明输出 exact；本轮 full 以固定 accuracy 结果作为精度证据，
不把性能请求文本宣称为逐条 exact。

## Triton DSL 简化复测

在不改变 shape gate、外层 custom-op 边界和 CUDA Graph 拆分的前提下，
`(1,17408) @ (5120,17408).T` 的手写 HIP output-projection kernel 已替换为
18 行独立 Triton 模块。K=5120 pair-reduction 的源码随后做了语法等价压缩，
但 640-thread 分块、load/FMA/reduction 顺序和 launch 参数均未改变。最终运行时
补丁相对直接代码基线为新增 469 行、删除 30 行，保守改动量共 499 行；压缩前
为 2656 行，因此减少 2157 行（81.2%）。该口径不依赖删除换行或空白符。

同一 gfx936、同一 seed 和同一 BF16 tensor 上各执行 9 轮、每轮 200 次：

| 实现 | 中位延迟（ms） | 最小–最大（ms） | 对 `torch.linear` 最大绝对误差 |
| --- | ---: | ---: | ---: |
| 原 HIP | 0.131415 | 0.131291–0.132086 | 1.49e-8 |
| Triton | 0.130084 | 0.129424–0.130170 | 1.49e-8 |

Triton 中位延迟低 `1.013%`，三 seed dispatch 单测均通过。clean wheel 的官方
单请求三档复测如下；旧值取自上面的权威 full 结果：

| 输入档 | 原 tok/s | Triton tok/s | 变化 | 原 TPOT P99（ms） | Triton TPOT P99（ms） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4–8K | 21.603469 | 21.672151 | +0.318% | 40.749717 | 40.726129 |
| 8–16K | 19.139019 | 19.128435 | -0.055% | 41.670475 | 41.654575 |
| 16–32K | 15.574232 | 15.560856 | -0.086% | 42.671636 | 42.651121 |

三档共 `150/150` 成功，TPOT P99 均改善，吞吐差异不超过 `0.318%`，因此只
判定为性能等价。相同 wheel 的全量精度仍为 `110/110`：HotpotQA `77.96`、
GovReport `33.03`、Retrieval 与 Aggregation 均为 `100.00`，四项 `k=1.00`。
本节不把细小时间差声明为新优化收益。

上述全量性能与精度使用的 clean wheel SHA-256 为
`4673386de52e3d396812f6242e2d67d790d8cf624e978b6b8108d0c0bf79698d`。
仅将局部 dispatch 布尔变量改为更明确的名称并补齐文档后，又从空 build tree
构建最终 wheel：
`0c8bafdfd97f4301234b298961a498c4dbe82f3d88e92d5749b214feffb621e9`。
两者 Triton kernel 完全相同；最终构建再次匹配计时为原 HIP `0.131950 ms`、
Triton `0.130163 ms`（低 `1.354%`），最大绝对误差均为 `1.49e-8`。最终构建
还通过了实际分发计数的三 seed GPU 检查和完整复现脚本。

## 干净构建与冷启动

空 build tree 产物：

| Artifact | SHA-256 |
| --- | --- |
| wheel | `50f21c3a6a952be49d9cf5db19b0ec030796d310f8c99e48ad5bfe3b8ecb1d8d` |
| wheel 中 `_rocm_C` | `f498997455ff8c75c74ca2ba791380b5a311ae17f8a1398bd3219e306adea0f3` |
| TunableOp profile | `169c7b11a0340d9e22405327b5e5667b2aa9e9e8d899bd59e10ca4fb7fb52030` |

wheel 共 1992 个条目，不含 `__pycache__`、`.pyc`、`perf_trace` 或已删除
实验模块。预编译 wheel 不提交到仓库；评测机按
[闭卷复现手册](CLOSED_BOOK_REPRODUCTION.md) 从源码构建。

fresh cache 的 `torch.compile` 用时 `463.78 s`，初始 profiling/warmup
用时 `157.55 s`，graph capture 用时约 `23 s`。这些均发生在 health ready
之前，没有计入 TTFT。日志同时确认 5 条 profile 结果和 5 条 validator
加载成功，INIT 与 PRE_CAPTURE 均为 `status=ready`。

## 证据边界

性能、精度、服务日志和模型权重不提交到源码仓库；上文保留固定脚本、结果
JSON、summary 和构建产物的哈希用于核对。源码边界由
`evidence/manifests/repro_minimal_runtime.sha256` 固定，并可通过
`scripts/verify_cscc_repro.sh` 验证。

## 可选双卡多请求结果

上述 91.08 分与精度结论仍对应官方单请求 DP1 契约。分支另提供同机
`TP=1, DP=2, backend=mp` 服务与固定多请求 benchmark；它复用相同单卡
kernel，不使用投机解码。最终 499 行版本的 concurrency 2/4/8 × 三输入档
全矩阵为 `72/72` 成功且无 OOM；相对 2656 行最优实现，concurrency 8 三档
吞吐依次为 `99.695%`、`100.360%`、`98.257%`，几何平均 `99.433%`。
同一热服务的全量精度为 110/110、四项系数均为 1。双 rank 冷编译、稳态
重启条件、完整数据与同负载 DP1 对照见
[DP2_MULTI_REQUEST.md](DP2_MULTI_REQUEST.md)。多请求 `output_tok_s` 不参与
本页官方分数计算。
