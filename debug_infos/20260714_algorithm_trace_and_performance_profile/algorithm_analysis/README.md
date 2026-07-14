# 算法分析数据说明

本目录整理的是 2026-07-07 参考 VisiPrune 思路采集的 patch trace 与 selected-layer FX trace。除本说明外，共保留 94 个原始数据或原始报告文件；所有文件均直接放在本目录，不再保留多层子目录。

归档过程只调整了文件名以消除原目录层级和重名，文件内容没有改写。脚本、源码、服务日志、PID、序列化模型（`.pt`）和性能计时数据不在本目录中。

## 建议阅读顺序

1. [`algorithm_visualization_report.md`](algorithm_visualization_report.md)：原始算法可视化报告，按 12 个逻辑过程展示 FX 节点流，是本目录的主入口。
2. [`selected_layer_fx_process_summary.md`](selected_layer_fx_process_summary.md)：9 个采样事件的过程重建总表；对应机器可读版本为 `selected_layer_fx_process_summary.json`。
3. [`selected_layer_fx_trace_summary.md`](selected_layer_fx_trace_summary.md)：FX trace 的采集范围和完整性摘要；对应机器可读版本为 `selected_layer_fx_trace_summary.json`。
4. [`patch_trace_summary.md`](patch_trace_summary.md)：较早期 patch trace 的采集摘要；对应机器可读版本为 `patch_trace_summary.json`。

## 采样覆盖

每个事件前缀代表一个上下文档位、一次输入/forward 和一个模型层。9 个前缀如下：

| 上下文 | 事件前缀 | layer | forward | q_len |
| --- | --- | ---: | ---: | ---: |
| 4-8K | `4-8K_input1_layer3` | 3 | 1 | 4096 |
| 4-8K | `4-8K_input1_layer31` | 31 | 1 | 4096 |
| 4-8K | `4-8K_input1_layer59` | 59 | 1 | 4096 |
| 8-16K | `8-16K_input1_layer3` | 3 | 1 | 4096 |
| 8-16K | `8-16K_input3_layer59` | 59 | 3 | 4096 |
| 8-16K | `8-16K_input4_layer31` | 31 | 4 | 1685 |
| 16-32K | `16-32K_input1_layer3` | 3 | 1 | 4096 |
| 16-32K | `16-32K_input2_layer31` | 31 | 2 | 4096 |
| 16-32K | `16-32K_input4_layer59` | 59 | 4 | 4096 |

每个事件均包含 155 个 FX 节点，重建为 12 个逻辑过程；总表记录的节点分配完整、无重复。

## 文件及用途

### 总报告与汇总（7 个）

| 文件 | 用途 |
| --- | --- |
| `algorithm_visualization_report.md` | 原始可视化报告；查看 12 个过程的顺序、节点组成、张量形状和跨过程数据流。 |
| `selected_layer_fx_process_summary.md` | 人工阅读的 9 个事件过程重建汇总。 |
| `selected_layer_fx_process_summary.json` | 上述过程汇总的机器可读数据。 |
| `selected_layer_fx_trace_summary.md` | 人工阅读的 selected-layer FX trace 总结。 |
| `selected_layer_fx_trace_summary.json` | 上述 trace 总结的机器可读数据。 |
| `patch_trace_summary.md` | 人工阅读的早期 patch trace 总结。 |
| `patch_trace_summary.json` | 上述 patch trace 总结的机器可读数据。 |

### 上下文级索引与原始事件（每个上下文 11 个，共 33 个）

下表中的 `<context>` 分别为 `4-8K`、`8-16K`、`16-32K`。

| 文件模式 | 每个上下文数量 | 用途 |
| --- | ---: | --- |
| `<context>_fx_layer_events.csv` | 1 | 从原始事件中筛出的 layer/forward 事件索引。 |
| `<context>_fx_layer_trace_manifest.csv` | 1 | 采样事件与各 FX 产物之间的映射及完整性清单。 |
| `<context>_fx_run_metadata.json` | 1 | 本上下文的运行参数、目标层和 trace 元数据。 |
| `<context>_fx_events.<pid>.jsonl` | 4 | selected-layer FX 采集产生的逐进程原始事件流。 |
| `<context>_patch_events.<pid>.jsonl` | 4 | 较早期 patch trace 产生的逐进程原始事件流，用于与 FX 重建交叉核对。 |

原始 JSONL 的进程编号如下：

| 上下文 | FX JSONL PID | patch JSONL PID |
| --- | --- | --- |
| 4-8K | `48423`, `48499`, `48500`, `48657` | `31698`, `31774`, `31775`, `31932` |
| 8-16K | `50121`, `50197`, `50198`, `50358` | `33266`, `33342`, `33343`, `33500` |
| 16-32K | `51661`, `51737`, `51738`, `51895` | `34834`, `34910`, `34911`, `35068` |

### 事件级 FX 数据与报告（每个事件 6 个，共 54 个）

对“采样覆盖”表中的每个事件前缀 `<event>`，均有以下文件：

| 文件模式 | 用途 |
| --- | --- |
| `<event>_fx_graph.txt` | `make_fx` 导出的可读 FX 图文本，用于查看算子顺序和张量流。 |
| `<event>_fx_nodes.json` | 完整 FX 节点、参数和 shape 等机器可读信息。 |
| `<event>_fx_process_nodes.csv` | 每个 FX 节点到 12 个逻辑过程的归属表，便于筛选、统计和二次可视化。 |
| `<event>_fx_process_reconstruction.json` | 过程重建的机器可读结果。 |
| `<event>_fx_process_reconstruction.md` | 单事件的过程重建报告，适合逐层人工核对。 |
| `<event>_fx_trace_metadata.json` | 该事件的层号、forward 次序、输入 shape 和 trace 来源信息。 |

## 证据边界

- 这是固定采样输入上的 FX/patch 执行结构证据，用于解释“算法做了什么”和节点如何组成过程，不是全模型覆盖率证明。
- 12 个过程名称来自对 FX 节点的重建与归类；原始节点和归属表均保留，可用于复核。
- 自定义 attention 等边界内的实现不会被 `make_fx` 无限展开，因此报告描述的是此次实际捕获到的图边界。
- 本目录不包含耗时归因。当前最佳版本的性能结果和 profiler 数据位于相邻的 `performance_analysis/`。
