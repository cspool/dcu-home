# 优化方案与技术路线

## 目标与最终栈

目标是在统一 Qwen3.5-27B BF16 权重、统一 vLLM 0.18.1、固定服务参数、
固定 tokenizer/chat template 和单卡单请求约束下，满足 TTFT/TPOT SLA 并
提高三档长上下文输出吞吐。

本仓库本次实测候选中最高且已完成 full/accuracy 闭环的版本为：

```text
R25 = R24 累计优化栈 + H11.5 wide-causal GQA6 prefill
                       + H10.8 gfx936 strided LLMM1
```

直接代码基线是 OpenDAS `vllm_cscc` fork
（<http://developer.sourcefind.cn/codes/OpenDAS/vllm_cscc.git>）的提交前
commit `fa718036bdb9dfd80a872b86c8ac16c9d02bfd31`。GitHub vLLM 是该 fork
所基于的原始第三方项目，不应把此 commit 误写成 GitHub upstream commit。

## H11.5：wide-causal GQA6 prefill

主要代码：

- `vllm/v1/attention/ops/rocm_aiter_unified_attention_gqa6.py`
- `vllm/v1/attention/backends/rocm_aiter_unified_attn.py`

精确适用条件：

- gfx936
- BF16
- head size 256
- GQA6
- 单序列长 causal prefill
- cache block size 784

实现要点：

- 将每个 KV head 对应的 6 个 query heads 分为三个两头组；
- 使用逻辑 56-token K/V tile，并为 MFMA padding 到 64；
- `BLOCK_Q=32`；
- 根据 causal query 上界减少无效 K/V 访问；
- 其他 shape、短 query、多序列和 decode 路径回退到 H11.4/AITER 原路径。

## H10.8：gfx936 strided LLMM1

主要代码：

- `csrc/rocm/skinny_gemms.cu`
- `csrc/rocm/ops.h`
- `csrc/rocm/torch_bindings.cpp`
- `vllm/_custom_ops.py`
- `vllm/model_executor/layers/utils.py`

精确 gate：

- gfx936
- BF16
- `n=1`
- `k=5120`
- 无 bias
- 连续输入/权重
- `m in {14336, 16384, 34816}`

三个大投影使用 `LLMM1Strided(4,640)` 的 wave-pair reduction；`m=96`
继续使用原 LLMM1。非目标 shape 走原 GEMM 路径。

H10.9 的 backend 强制切换经审计为 runtime no-op；H10.10 K6144
pair-reduce 虽 standalone microbenchmark 为正，但端到端三档 TPOT 全部回归。
两者均已从最终源码移除。

## 累计语义安全快路径

最高分版本还包含 R24 及更早已闭环的累计改动，主要涉及：

| 类别 | 关键路径 | 约束 |
| --- | --- | --- |
| GDN recurrent | `fused_recurrent.py`、`gdn_attn.py` | 保持状态更新和输出语义 |
| Qwen3.5 执行 | `qwen3_next.py` | 不改变模型结构、权重或 scheduler |
| ROCm 平台/ops | `platforms/rocm.py`、`_custom_ops.py` | 仅目标设备和 shape 命中 |
| Attention wrapper | `rocm_aiter_unified_attn.py` | 非目标路径回退 |
| 构建绑定 | `setup.py`、ROCm bindings | 从源码编入最终 wheel |

完整 13 个最终源码文件及哈希见
`evidence/manifests/repo_source.sha256`。

## 各项性能贡献

| 阶段 | 对照 | 观测贡献 | 用途 |
| --- | --- | --- | --- |
| H11.5 | R24 小样本 | 三档 TTFT 改善 15.89% / 20.47% / 24.65% | prefill 晋级信号 |
| H10.8 | H11.5 小样本 | 三档 TPOT 改善 5.233% / 5.033% / 4.906% | decode 晋级信号 |
| H11.5+H10.8 full | R24 full 均值 | 三档吞吐 +6.744% / +9.540% / +13.724% | 最终计分 |
| H11.5+H10.8 score | R24 score | 85.707490 -> 88.548456 | 本次实测最高 |

小样本只用于候选筛选；最终成绩以三次 full、固定 accuracy 和 SLA 闭环为准。

## 合规边界

本版本没有：

- 修改固定 `start_vllm.sh`、`run_throughput.sh` 或
  `run_accuracy.sh`；
- 修改模型权重、tokenizer、chat template、测试数据或请求流；
- 修改 scheduler/batch 边界；
- 使用 prefix cache、跨样本持久化结果或预生成量化权重；
- 通过环境变量强制开启未验证路径；
- 保留 H10.10 K6144 或 H10.9 backend 强制切换。

本文件与 `RESULTS.md`、`COMPLIANCE.md` 构成自包含提交说明；最终源码
可由 `evidence/manifests/repo_source.sha256` 在仓库根目录直接校验。
