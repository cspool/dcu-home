# PRA2026-BH408：Qwen3.5-27B ROCm 压缩优化实现

本仓库基于 OpenDAS `vllm_cscc` 官方原版提交
`fa718036bdb9dfd80a872b86c8ac16c9d02bfd31`，在 vLLM 0.18.1 内直接修改
ROCm、GDN、Attention 和模型执行路径。第三方来源、版本与许可证见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，上游项目说明保存在
[README_UPSTREAM.md](README_UPSTREAM.md)。

当前实现不依赖历史 `pra2026-bh408` 源码运行，也不再使用独立
`qwen35_rocm_opt` 包；该归档只作为高性能实现与历史结果的只读参照。相对官方
原版，计入 `csrc/ + setup.py + vllm/` 的运行时代码为 18 个文件、增加 567 行、
删除 33 行，churn 合计 **600 行**。

## 当前结论

- 主评测配置：物理 DCU 0、Qwen3.5-27B BF16、TP/PP/DP=1、并发 1。
- 禁用项：权重量化、权重重排或压缩缓存、投机解码、prefix cache，以及采样、
  输出长度和 scheduler 语义修改。
- 相对 3k 最优归档，4--8K、8--16K、16--32K 官方原始输出吞吐分别为
  `-0.471% / -0.328% / -0.511%`，三档都在 99% 性能门内。
- 全量性能请求 `150/150` 成功；全量精度请求 `110/110` 成功，四项精度无扣分。
- `pra2026-bh408` 归档未被本实现修改。

完整吞吐、精度、约束审计和结果哈希见
[600 行复现报告](docs/cscc/MODULAR_3K_PARITY.md)。

## 从官方原版实现

推荐从官方 `fa718036` 逐 hunk 修改，而不是从 3k 目录复制文件：

1. 在官方已有 `LLMM1`、FLA wrapper、ROCm backend、Qwen3.5 model runner 中
   插入精确 shape gate；条件不满足时保留官方 fallback。
2. 只新增三个必要文件：共享 gfx936 helper、GQA6/page784 op 和 5 行
   TunableOp profile。
3. 用 `git diff fa718036 -- csrc setup.py vllm` 审核全部运行时变化。

逐文件官方锚点、实施顺序、性能优先级、修改难度和验证门禁见
[官方原版优化实施指南](docs/cscc/OFFICIAL_BASE_OPTIMIZATION_GUIDE.md)。

## 构建与单卡启动

工具链和干净构建步骤见 [BUILD.md](BUILD.md)，环境变量逐项说明见
[ENVIRONMENT.md](ENVIRONMENT.md)。最小启动方式为：

```bash
source scripts/cscc_gfx936_env.sh
HIP_VISIBLE_DEVICES=0 vllm serve /path/to/Qwen3.5-27B \
  --served-model-name Qwen3.5-27B \
  --port 8001 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.95 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

不要加入量化、`--speculative-config`、draft/MTP 或 prefix-caching 参数。

## 可选 DP=2 配置

仓库保留同机双卡 `TP=1, DP=2, backend=mp` 配置，用于决赛多卡方案准备和多请求
压力测试；两个 DP rank 各自运行完整单卡副本，不改变单卡 kernel 数学语义。
初赛官方评分仍是单卡、并发 1，DP=2 数据不得混入单卡得分。

构建、单卡/DP=2 启动和冷编译后热重启命令见
[构建与启动简明流程](docs/cscc/BUILD_SERVE_CACHE_QUICKSTART.md)。

## 文档入口

- [CSCC 文档索引](docs/cscc/README.md)
- [官方原版优化实施指南](docs/cscc/OFFICIAL_BASE_OPTIMIZATION_GUIDE.md)
- [源码构建、启动与冷热缓存简明流程](docs/cscc/BUILD_SERVE_CACHE_QUICKSTART.md)
- [600 行全量性能与精度报告](docs/cscc/MODULAR_3K_PARITY.md)
- [环境变量](ENVIRONMENT.md)
- [干净构建](BUILD.md)
- [第三方代码与许可证](THIRD_PARTY_NOTICES.md)

仓库不包含模型权重、测试数据、预编译 wheel、服务日志或结果 JSON。
