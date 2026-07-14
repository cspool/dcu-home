# 算法 Trace 与当前最佳版本性能资料

本目录整理 `/public/home/tangyu408/testdata` 中两类资料：

- **算法分析**：保留早期借鉴 VisiPrune 方法生成的原始 selected-layer FX 可视化报告及其审计资产。
- **性能分析**：只保留明确针对当前最高综合分版本 **H11.5 + H10.8** 的端到端结果、运行时验证和定向硬件 profile；R4/R11/R23 历史性能 profile 及后续候选对比不在本归档中。

## 入口

1. 算法可视化原报告：[`selected_layer_fx_process_visualization.md`](algorithm_analysis/original/profile_runs/selected_layer_fx_20260707_codex/selected_layer_fx_process_visualization.md)
2. 算法报告边界与阅读说明：[`algorithm_analysis/README.md`](algorithm_analysis/README.md)
3. 当前最佳性能分析：[`performance_analysis/README.md`](performance_analysis/README.md)
4. 源目录级盘点与纳入策略：[`SOURCE_INVENTORY.tsv`](SOURCE_INVENTORY.tsv)
5. 281 个原始文件的逐文件来源与 SHA256：[`SOURCE_FILES.tsv`](SOURCE_FILES.tsv)

## 当前最佳身份

- 版本：H11.5 wide-causal GQA6 prefill + H10.8 gfx936 strided LLMM1。
- 源码提交：`89990f44855932fcead4746673abbf847d7717ce`。
- wheel SHA256：`03568ba87ff64fd0a8aade299026d7ee78cbf40d9c1ed5884fb584250b2031f2`。
- full×3 综合分：`88.4903491377583`、`88.5784836941864`、`88.5765336801012`；均值 `88.5484555040153`。
- Accuracy：`K=1.0`；`450/450` 请求完成，`failed=0`。

完整 full×3、accuracy、服务、源码与构建证据已在相邻归档 [`20260713_current_best_h11_5_h10_8_plus_h10_only`](../20260713_current_best_h11_5_h10_8_plus_h10_only/README.md) 中提交。本目录只补充算法可视化和当前最佳路径的硬件分析原件，不重复分发 wheel 或整套 benchmark。

## 目录内容

- `algorithm_analysis/original/`：原始 patch trace、selected-layer FX/reconstruction、可视化报告、注入源码和生成脚本。与性能无关的历史 process-latency 子实验未纳入。
- `performance_analysis/original/current_best/`：H10.8 runtime/compile 证据、H11.5 attention/GDN MMOP 与 LDS counter、当前 prefill GEMM/MMAC 证据、H11.5 L4096 reference trace，以及最终版本身份摘要。
- `shared_tooling/original/`：原始服务、吞吐与环境辅助脚本。
- `FILES.txt`、`SHA256SUMS`：逐文件字节表和 SHA256 校验。

`original/` 下内容从远程容器原文件逐字节复制，未改写。归档新增的 README、inventory 和 checksum 仅负责组织与解释。历史绝对路径、设备编号和 localhost 端口属于采集环境信息；`.pt` 为 PyTorch 序列化文件，只应在可信环境中加载。

## 完整性校验

在本目录执行：

```bash
sha256sum -c SHA256SUMS
```

`FILES.txt` 列出全部归档文件及字节数。
