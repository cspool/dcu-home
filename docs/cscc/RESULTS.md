# 最终测试结果

## 测试契约

最终候选 H11.5 + H10.8 使用未修改的固定脚本：

- `run_throughput.sh all`，不传第二参数；
- `MAX_CONCURRENCY=1`；
- `REQUEST_RATE=1`；
- `CUSTOM_OUTPUT_LEN=1024`；
- `NUM_WARMUPS=2`；
- 固定 `run_accuracy.sh all`。

三次 full 每档各 50 请求，每档 benchmark 主体均超过 600 秒。

## 三次 full

| Run | 4-8K tok/s | 8-16K tok/s | 16-32K tok/s | 综合分，K=1 | 相对 R24 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 19.589185273966 | 17.025544511643 | 13.003919636950 | 88.490349137758 | +10.021653961% |
| 2 | 19.587005633034 | 17.127750783436 | 13.003706763767 | 88.578483694186 | +10.346212210% |
| 3 | 19.584774728543 | 17.126009556001 | 13.004622851957 | 88.576533680101 | +10.340604760% |
| 均值 | 19.586988545181 | 17.093101617027 | 13.004083084225 | **88.5484555040153** | **+10.2361569769%** |

共 `450/450` 请求成功，`failed=0`。三轮逐请求 `output_lens` 和
`generated_texts` 完全一致。

## 相对 R24 分档提升

| Band | R24 均值 | 当前均值 | 提升 |
| --- | ---: | ---: | ---: |
| 4-8K | 18.349460025353 | 19.586988545181 | +6.744223% |
| 8-16K | 15.604371326753 | 17.093101617027 | +9.540469% |
| 16-32K | 11.434815640267 | 13.004083084225 | +13.723592% |

20/50/30 加权提升为 `+10.2361569769%`。

## SLA

| Metric | 官方 1.5x 限制 | 当前最大值 | 结果 |
| --- | ---: | ---: | --- |
| 4-8K TTFT P99 | 7188.718023 ms | 1956.991400 ms | pass |
| 8-16K TTFT P99 | 37329.278019 ms | 6129.860216 ms | pass |
| 16-32K TTFT P99 | 43111.256069 ms | 6549.847646 ms | pass |
| Global TPOT P99 | 107.705132 ms | 47.194475 ms | pass |

Global TPOT P99 按每请求 `sum(itls)/(output_len-1)` 重建 150 请求后计算。

## Accuracy

| 数据集 | 官方 baseline | 当前结果 | k |
| --- | ---: | ---: | ---: |
| hotpotqa | 77.959706960 | 77.959706960（77.96） | 1.00 |
| gov_report | 32.961006236 | 33.054713499（33.05） | 1.00 |
| retrieval_multi_point | 100.00 | 100.00 | 1.00 |
| aggregation_keyword_aggregation | 100.00 | 100.00 | 1.00 |

最终 `K=1.00`。aggregation 的权威结果按固定脚本 Counter 对 prediction
列表和 gold 多重集合重算为 100.00；OpenCompass 原生中间 summary 的 0.00
不是固定脚本最终口径。

## H10-only 提交增量

提交版从源码独立重建 wheel，在 fresh 服务上连续运行三个完整
`run_throughput.sh all 3` round，自然窗口 `661 s`，idle padding=0。
27/27 请求成功、failed=0；三轮三档的 output length 和完整文本 hash
均与冻结 H11.5+H10.8 baseline exact。

| Band | Baseline all3 | H10-only 三轮均值 | 相对变化 |
| --- | ---: | ---: | ---: |
| 4-8K | 12.948554862 | 13.054359229 | +0.817113% |
| 8-16K | 15.771696925 | 15.883771439 | +0.710605% |
| 16-32K | 9.889236403 | 10.027930935 | +1.402480% |

20/50/30 加权提升为 `+0.939469%`。pooled request TPOT P99 为
`47.135118 ms`，最大 TTFT P99 为 `6349.108759 ms`，SLA 通过。此前使用
同一四行 profile 的独立 C100 窗口为 `664.215 s`、加权 `+0.958538%`，
同样 27/27、三档为正且输出 exact，因此该增量不是单次噪声。

提交版 fixed `run_accuracy.sh all` 自然运行 `943 s`，结果为：

| 数据集 | 官方 baseline | 提交版显示值 | k |
| --- | ---: | ---: | ---: |
| hotpotqa | 77.959706960 | 77.96 | 1.00 |
| gov_report | 32.961006236 | 32.95 | 1.00 |
| retrieval_multi_point | 100.00 | 100.00 | 1.00 |
| aggregation_keyword_aggregation | 100.00 | 100.00 | 1.00 |

最终 `K=1.00`。本增量尚未重跑 full×3，不能把 `+0.939469%` 直接叠加为
新的综合分；上文 `88.5484555040153` 仍是权威 full 计分锚点。

## 构建与运行时锚点

| Artifact | SHA256 |
| --- | --- |
| final wheel | `03568ba87ff64fd0a8aade299026d7ee78cbf40d9c1ed5884fb584250b2031f2` |
| H10-only submission wheel | `fe8ceeec1634db072b179ba88f364e489640ea246eef5aab8a0487253511307a` |
| installed `_rocm_C.abi3.so` | `51e4839b564355279fcca4bc426ccd1da0a5f03d0e39006210960e99fd124ab1` |
| `run_throughput.sh` | `adf0cf91266745b37df916926c7d495ec79f00a11be653c219d1d5df4d93c681` |
| `run_accuracy.sh` | `2e641672a45ac96318c2118df8df4dae2babf87c16afd49cbe4b037ff9beed4e` |
| `start_vllm.sh` | `7c3e8c5ecdf02109e02af8c3b5ba05050b26339c7f50869b5288eea359364fad` |

最终 wheel 未提交；评测机从完整源码重新编译。本文件已自包含最终计分、
SLA、accuracy 与产物哈希；仓库内可直接验证的源码/运行时锚点位于
`evidence/manifests/`。固定脚本哈希另存于
`evidence/manifests/fixed_scripts_reference.txt`；它是评测平台外部脚本
的引用记录，不能在本仓库内执行 `sha256sum -c`。
