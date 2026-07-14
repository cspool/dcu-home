# 当前最佳 H11.5 + H10.8 与 H10-only 提交增量原始调试资料

本目录归档截至 2026-07-14 仍然有效的最高综合分版本 **H11.5 + H10.8**，以及当前源码提交在其上增加的 **H10-only ROCm TunableOp profile** 的原始测试数据、固定测试脚本、说明文档与版本身份材料。

## 结论

- 版本：H11.5 wide-causal GQA6 prefill + H10.8 gfx936 strided LLMM1
- 三轮完整 `run_throughput.sh all` 综合分：`88.4903491377583`、`88.5784836941864`、`88.5765336801012`
- 三轮均分：`88.5484555040153`
- Accuracy 系数：`K=1.0`
- 完成情况：`450/450` 请求成功，`failed=0`，三档 SLA 全部通过
- wheel SHA256：`03568ba87ff64fd0a8aade299026d7ee78cbf40d9c1ed5884fb584250b2031f2`
- H11.5 + H10.8 对应仓库提交：`89990f4`（`Submit H11.5 + H10.8 highest-scoring vLLM source`）

三轮原始吞吐如下：

| Run | 4-8K tok/s | 8-16K tok/s | 16-32K tok/s | Score (K=1) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 19.589185273966 | 17.025544511643 | 13.003919636950 | 88.490349137758 |
| 2 | 19.587005633034 | 17.127750783436 | 13.003706763767 | 88.578483694186 |
| 3 | 19.584774728543 | 17.126009556001 | 13.004622851957 | 88.576533680101 |

当前仓库随后增加了 H10-only profile：三个完整 `run_throughput.sh all 3` round 的 20/50/30 加权提升为 `+0.939469%`，共 `27/27` 请求成功，输出长度和全文 hash exact；固定 accuracy 为 `K=1.0`。其 wheel SHA256 为 `fe8ceeec1634db072b179ba88f364e489640ea246eef5aab8a0487253511307a`。该增量尚未运行 full×3，因此不能把小样本提升直接叠加到综合分；`88.5484555040153` 仍是权威 full 计分锚点。

在该版本之后完成最终计分闭环、且最接近当前最佳的三个候选均未超过它：

| 后续候选 | 最终分 | 决策 |
| --- | ---: | --- |
| LR-L1/H10.16 | 88.29734873154621 | `REJECTED-SCORE` |
| LR-A2/H11.8 | 88.50282851763967 | `REJECTED-SCORE` |
| LR-IC-B1 | 88.52564230952228 | `REJECTED-SCORE` |

对应原始评分报告位于 `selection_evidence/post_best_candidates/`。其余后续探索若未完成完整 full/accuracy 计分闭环，不具备替代当前最佳的证据资格。

## 目录内容

- `original/scripts/`：原始固定评测脚本，以及当前源码提交的构建/环境脚本。
- `original/candidate/full_runs/`：当前最佳三轮完整吞吐原始 JSON、日志、状态与时间窗口。
- `original/candidate/final_accuracy/`：完整 OpenCompass 配置、预测、结果、summary 与运行日志。
- `original/candidate/serve_final/`：服务启动、健康检查、最终日志和停服证据。
- `original/current_submission_h10_only/`：H10-only 的构建、三轮 all3、独立 C100 复验、服务、canary、accuracy 与 OpenCompass 原始产物；源端没有保留下单独的完整 `opencompass_run.log`，本归档没有重建或伪造缺失文件。
- `original/baselines/official/`：官方吞吐/accuracy baseline 原始结果。
- `original/baselines/r24/`：用于计算相对 R24 提升的原始对照结果。
- `original/docs/`：冻结证据说明与仓库当前说明文档的原始副本。
- `original/provenance/`：源码 diff、运行时哈希、固定文件哈希、构建命令/日志及 wheel 元数据。
- `selection_evidence/`：后续最接近候选未超过当前最佳的原始评分报告。

`original/` 下文件均从远程容器中的原始产物逐字节复制，未改写内容。官方固定输入 JSONL 属于评测平台外部资产，未在本仓库重复分发；其冻结 SHA256 仍保存在 `original/provenance/fixed/fixed_datasets.sha256`。`README.md`、`FILES.txt` 和 `SHA256SUMS` 是本次归档新增的索引与校验文件。

## 二进制说明

H11.5 + H10.8 wheel 大小为 `57,086,560` 字节，H10-only wheel 大小为 `57,091,016` 字节。为避免在 Git 仓库中重复保存可由已提交源码构建的二进制，本目录没有复制 wheel 本体；其 SHA256、文件状态、完整构建命令和构建/安装日志均已保留。H11.5 + H10.8 源码身份可由提交 `89990f4` 以及 `original/provenance/source/` 中的冻结 diff 和哈希复核，H10-only 源码身份可由仓库 manifest 复核。

## 与源码提交说明的关系

仓库原有 `README.md` 与 `docs/cscc/COMPLIANCE.md` 描述的是正式源码提交主体，按该口径不携带固定评测脚本或原始测试输出。本 `debug_infos/` 目录是按本次明确要求新增的复现实验/调试附件，不是预编译产物替代源码，也不改变上述测试口径。复制进 `original/docs/repository_current/` 的说明文件保持原样，用于记录归档时的源码提交语境。

## 完整性校验

在本目录执行：

```bash
sha256sum -c SHA256SUMS
```

`FILES.txt` 列出归档中的全部文件及字节数。由于上游仓库 `.gitignore` 默认忽略 `*.log` 和 `*.csv`，这些原始产物在本次提交中被显式纳入跟踪。
