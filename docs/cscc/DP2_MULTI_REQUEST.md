# 可选 DP=2 配置与验收边界

## 定位

本页只保留同机双卡配置，不把历史 DP=2 实验混入当前单卡 R24 结论。初赛技术
方案明确采用单卡、并发 1；决赛采用同型号多卡但卡数与评分规则另行公布。因此：

- 当前官方性能与精度结论仍以物理 DCU 0、TP/PP/DP=1 为准；
- DP=2 用于决赛准备与多请求压力测试；
- 决赛规则公布后必须重新核对卡数、并发、评分和允许参数，不能直接沿用初赛分数。

DP=2 是两个完整的 TP=1 模型副本，由 vLLM MP data-parallel frontend 分发请求；
它不把现有单卡 kernel 改成 TP kernel，也不改变每个副本的模型数学语义。

## 启动

```bash
MODEL_DIR=/path/to/Qwen3.5-27B \
HIP_VISIBLE_DEVICES=0,1 \
PORT=8001 \
bash scripts/serve_cscc_dp2.sh
```

脚本要求恰好两个不同设备，并固定：

```text
dtype=bfloat16
tensor_parallel_size=1
data_parallel_size=2
data_parallel_backend=mp
max_num_seqs=128
max_num_batched_tokens=4096
compilation_config.compile_sizes=[4096]
gpu_memory_utilization=0.95
```

DP=2 脚本显式传 `compile_sizes=[4096]`，因为源码中的单卡自动门只接管
world/DP=1；显式配置使两个独立 worker 使用相同静态 shape。脚本同时加载冻结
TunableOp profile，且没有量化、投机解码或 prefix cache 参数。

## 冷启动后必须热重启

第一次启动会分别为两个 rank 生成 Triton、TorchInductor 和 AOT cache。历史测试
中，冷编译次序曾使两个 rank 的 KV cache 容量不一致，进而在长上下文并发请求中
产生抢占和吞吐回退。正确流程是：

1. 等两个 rank 都完成编译且 `/health` 成功；
2. 正常停止服务；
3. 使用完全相同的 wheel、源码缓存、模型、环境变量和参数重新启动；
4. 确认两个 rank 都直接命中各自 cache，且报告相同的 KV cache token 容量；
5. 再执行请求级 warmup 和正式 benchmark。

历史 499 行版本的稳定值是两个 rank 均为 28,224 tokens，但验收条件应是“相同
构建下两侧一致且能覆盖目标请求”，而不是把 28,224 写成跨版本常量。

## 多请求 benchmark

```bash
MODEL_DIR=/path/to/Qwen3.5-27B \
DATA_DIR=/path/to/testdata \
RESULT_ROOT=/path/to/results \
RUN_LABEL=dp2-r24-check \
DATASETS='4-8K 8-16K 16-32K' \
CONCURRENCIES='2 4 8' \
NUM_PROMPTS=8 \
OUTPUT_LEN=1024 \
NUM_WARMUPS=2 \
IGNORE_EOS=1 \
bash scripts/bench_cscc_multi_request.sh
```

wrapper 固定 `request-rate=inf`、temperature 0、输入不打乱，并拒绝覆盖已有
`result.json`。每个 case 至少核对 completed/failed、TTFT P99、TPOT P99、两个
rank 是否都实际处理请求，以及服务日志中是否有 OOM、engine death 或 fallback。

## 历史证据与当前状态

499 行压缩版本曾完成 concurrency 2/4/8 × 三输入档共 72/72 请求、无 OOM；
concurrency 8 三档相对当时 2656 行实现的几何平均为 99.433%，同一双卡热服务
完成 110/110 精度请求。该数据只证明上述拓扑和热重启方法曾可用。

当前 600 行 R24 完成的是单卡全量吞吐与精度，尚未用相同 R24 wheel 重跑 DP=2
全矩阵。因此 DP=2 配置保留为可执行入口，但不得把 499 行历史数据写成 R24
实测，也不得据此宣称决赛多卡收益。

## 规则与精度边界

- 两个副本仍加载官方 BF16 权重，不创建量化或重排权重文件；
- 不修改 sampling、chat template、输出长度或 batch scheduler 代码；
- 不使用 prefix cache、投机解码或跨请求答案/中间结果缓存；
- 决赛规则公布后重新进行端到端吞吐、SLA、110 条精度与多 rank 审计。
