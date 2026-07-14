# 算法分析：原始 FX Process 可视化

## 主入口

直接阅读原始报告：

[`original/profile_runs/selected_layer_fx_20260707_codex/selected_layer_fx_process_visualization.md`](original/profile_runs/selected_layer_fx_20260707_codex/selected_layer_fx_process_visualization.md)

该报告使用内联 Markdown/ASCII 图展示 Qwen3.5 decoder layer 内的 runtime inputs、residual/RMSNorm、融合 Q/Gate/K/V projection、Q/K head norm、MROPE、Q/K RoPE、KV cache update、attention custom op、attention gate/output projection、MLP/GDN 边界等过程，不依赖外部 PNG、SVG、HTML 或在线资源。

## 证据范围

- 上下文：`4-8K`、`8-16K`、`16-32K`。
- selected-layer event：共 9 个；每个 event 有 155 个 FX nodes 和 12 个重建 process。
- 除 `8-16K/input4_layer31` 的 `S=1685` 外，其余 event 的 sampled prefill chunk 均为 `S=4096`。
- 9 套 `fx_process_reconstruction.{md,json}`、`fx_process_nodes.csv`、`fx_nodes.json`、`fx_trace_metadata.json`、FX graph/module 及 manifest 均随整目录保留，因此原报告中的相对链接可以直接审计。
- 配套 `patch_trace_20260707_codex/` 记录三个上下文的执行调度和 join key：分别为 554、1158、1294 个事件，patch error 均为 0。

## 正确解释方式

这条链借鉴 VisiPrune 的 workload-analysis 思路，但数据来自本项目的 Qwen3.5/vLLM 运行：先在真实请求中采样目标 layer 输入，再对固定输入执行 `make_fx`，最后按规则把低层 ATen DAG 分组为可读 process。

因此：

- process 名称是重建规则标签，不是 PyTorch FX 官方语义，也不是运行时模块归属证明。
- FX DAG 是固定 sampled input 的低层图，不等于完整动态请求图。
- `unified_attention_with_output` 是 custom-op 边界；QK、mask、softmax、weighted-V 等 kernel 内部步骤没有在该 FX 图中展开。
- 这些 trace 用于理解算法结构和数据流，不应拿来替代 hipprof/torch profiler 的性能结论。

`original/profile_runs/VISIPRUNE_METHOD_REFERENCE.md` 是当时保留的 VisiPrune 方法说明原文，其中 `/workspace/VisiPrune/...` 路径和 VisiPrune event 数属于参考工作流，不是本归档 Qwen3.5 产物的实际路径。它作为方法 provenance 保留，不是本目录的入口说明。

## 原始内容

- `original/profile_runs/selected_layer_fx_20260707_codex/`：原始可视化报告、9 套 FX/reconstruction、summary、manifest、bench 和日志。
- `original/profile_runs/patch_trace_20260707_codex/`：执行调度 trace 和 summary。
- `original/source/{trace_patch,fx_trace_patch}/`：注入和采集源码。
- `original/scripts/`：patch trace、selected-layer FX 的采集、汇总和重建脚本。

原始目录中物理混放的 `process_latency/` 以及相应 patch/脚本属于历史性能 instrumentation，不是可视化报告的依赖；按“性能分析只保留当前最佳版本”的边界，本归档没有复制它们。
