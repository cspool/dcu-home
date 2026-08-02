# 本机 DP=2 多请求复现与结果

## 结论与适用范围

本分支在保留原单卡路径的同时，增加了一个严格限定的双卡服务拓扑：同一台
机器、两个不同 gfx936 设备、`TP=1`、`PP=1`、`DP=2`、内部 `mp`
backend。每个数据并行副本仍运行完整模型；请求由 vLLM 的 DP frontend 分发，
因此不需要把已有单卡 kernel 改写为 TP kernel。

该路径继续使用 `max_num_batched_tokens=4096`、`max_num_seqs=128`、BF16、
continuous batching 和现有 page784/TunableOp/GEMV 优化。量化、prefix cache、
投机解码以及 draft/MTP 模型均未启用。Ray、外部 LB、hybrid DP、跨节点和
`TP>1` 不在本次支持范围内；专用 ROCm 路径对这些拓扑 fail closed。

实现与评测源提交为
`4c4aa45b2987e521a226a49d5978d592c440667e`。原
`pra2026-bh408` 目录未被修改。

## 构建产物与静态验收

本次 clean build wheel：

```text
vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl
SHA-256: 1603b2ce5a77e04d6fdabce1aa6af9894ffc81ff8ad2d28ffd837afb8cb13465
```

wheel 从空 build tree 构建，不含 `.pyc`；`vllm._rocm_C` 可以加载，且包含
`qwen35_bf16_gemv`。以下校验均通过：

```bash
bash scripts/verify_cscc_repro.sh /path/to/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl
python3 -m pytest -q --confcutdir=tests/rocm \
  tests/rocm/test_rocm_tunableop_scope.py
```

拓扑测试共 20 项，覆盖允许的 DP1、DP2 rank0/rank1 以及被拒绝的错误
parent/local/rank 组合。

## 启动 DP=2 服务

安装 wheel 后，在仓库根目录执行：

```bash
export MODEL_DIR=/path/to/Qwen3.5-27B
export HIP_VISIBLE_DEVICES=0,1
export PORT=8001
bash scripts/serve_cscc_dp2.sh 2>&1 | tee service-dp2.log
```

脚本固定加入：

```text
--tensor-parallel-size 1
--data-parallel-size 2
--data-parallel-backend mp
--max-num-seqs 128
--max-num-batched-tokens 4096
--gpu-memory-utilization 0.95
```

`HIP_VISIBLE_DEVICES` 必须恰好包含两个不同设备。脚本不接受投机解码参数。

### 冷编译后的必要重启

首次启动会分别为两个 DP rank 生成 Triton/TorchInductor/AOT cache。首次实测
发现 rank1 在 rank0 之后编译时只得到 24,304-token KV cache，而 rank0 为
28,224 token；这种不对称状态会使 16–32K 并发 8 出现严重抢占，不能作为
稳态数据。

首次 `/health` ready 后应停止服务，使用完全相同的 wheel、源码缓存、模型和
参数重启。只有日志同时满足以下条件才开始计时：

```text
直接加载已有 compiled graph cache
AOT 日志分别命中 rank_0_0 和 rank_0_1 cache
两个 rank 均为 GPU KV cache size: 28,224 tokens
两个 rank 均为 VLLM_ROCM_TUNABLEOP_INIT status=ready
两个 rank 均为 VLLM_ROCM_TUNABLEOP_PRE_CAPTURE status=ready
speculative_config=None
compile_sizes=[4096]
```

这不是请求级 warmup 的替代；benchmark 仍单独执行两条 warmup 请求。冷启动
异常产物保留在验证目录中，没有覆盖或删除。

## 多请求 benchmark

另一终端执行固定 wrapper：

```bash
export MODEL_DIR=/path/to/Qwen3.5-27B
export DATA_DIR=/path/to/testdata
export RESULT_ROOT=/path/to/results
export RUN_LABEL=dp2-c8-full-warm-o1024-n8
export DATASETS='4-8K 8-16K 16-32K'
export CONCURRENCIES=8
export NUM_PROMPTS=8
export OUTPUT_LEN=1024
export NUM_WARMUPS=2
export IGNORE_EOS=1
bash scripts/bench_cscc_multi_request.sh
```

wrapper 固定 `request-rate=inf`、`temperature=0`、输入顺序不打乱、每档 8 条
正式请求、强制生成 1024 token，并拒绝覆盖已有 `result.json`。每个结果目录
同时保存详细 JSON、日志、运行元数据和汇总表。

## 稳态 DP=2 实测

测试日期为 2026-08-02。三档均为 concurrency 8，每档 8 条请求且总输出均为
8192 token：

| 输入档 | 成功 | duration（s） | 输出 tok/s | TTFT P99（ms） | TPOT P99（ms） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4–8K | 8/8 | 56.401538 | 145.244267 | 6367.700248 | 52.909948 |
| 8–16K | 8/8 | 68.651796 | 119.326812 | 15310.877853 | 62.713793 |
| 16–32K | 8/8 | 83.533743 | 98.068154 | 27816.719926 | 74.922719 |

8–16K 的并发扩展结果：

| 并发 | 成功 | duration（s） | 输出 tok/s | TTFT P99（ms） | TPOT P99（ms） |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 8/8 | 183.586538 | 44.622008 | 3818.860724 | 41.944916 |
| 4 | 8/8 | 116.570601 | 70.275009 | 6901.854550 | 56.438353 |
| 8 | 8/8 | 68.651796 | 119.326812 | 15310.877853 | 62.713793 |

并发提高使总吞吐上升，同时会提高 TTFT 和 TPOT；因此应根据吞吐目标与延迟
SLO 选择并发，不能只看 tok/s。

启动功能门禁还以 4 条同时发送的 chat 请求检查两个 DP rank，结果为 4/4
HTTP 200。该 smoke test 只证明路由与生成可用，不替代官方 110 条精度评测。

## DP=1 同负载对照

DP1 与 DP2 对照使用同一源码提交、clean wheel、模型、输入顺序、输出长度、
缓存状态和 benchmark 参数。最终结果及加速比记录在下表。

| 输入档 | DP1 tok/s | DP2 tok/s | DP2 / DP1 | 吞吐提升 | duration 降低 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4–8K | 120.093799 | 145.244267 | 1.209424× | 20.942% | 17.316% |
| 8–16K | 14.336466 | 119.326812 | 8.323307× | 732.331% | 87.986% |
| 16–32K | 9.435458 | 98.068154 | 10.393577× | 939.358% | 90.379% |

三档依次执行、每档输出 token 相同。按 `24576 / 三档 duration 之和` 计算，
DP1 聚合吞吐为 16.298836 tok/s，DP2 为 117.821297 tok/s，即 7.228817×；
总计时从 1507.837731 秒降到 208.587077 秒，降低 86.166%。两边均为
24/24 成功、0 失败。

单卡延迟同时表现出明显的容量拐点：

| 输入档 | DP1 TTFT P99（ms） | DP2 TTFT P99（ms） | DP1 TPOT P99（ms） | DP2 TPOT P99（ms） |
| --- | ---: | ---: | ---: | ---: |
| 4–8K | 12154.719841 | 6367.700248 | 63.826259 | 52.909948 |
| 8–16K | 489616.473013 | 15310.877853 | 525.536651 | 62.713793 |
| 16–32K | 812042.049525 | 27816.719926 | 758.836434 | 74.922719 |

8–16K 与 16–32K 的超线性加速不是 kernel 本身获得 8–10 倍算力。DP1 的
28,224-token KV cache 无法让 8 条长请求同时常驻，运行中长期处于约
87–100% cache、请求排队/换入状态；DP2 把请求分到两套独立 cache，跨过了
容量拐点。4–8K 没有同等严重的容量压力，因此其 1.21× 更接近该负载下的
常规双副本增益。不能把 8–10× 外推到低并发或短请求。

结果汇总 SHA-256：

```text
DP1 summary.md: d65941e079aab9e23aff001a858b558eb10654ec2e21ddaa10849cb754b2b12f
DP2 summary.md: 8a375b54d76fbc6eb465105c831af815c9e24d1dd0154b377b30d8119815118c
```

## 结果边界和外部证据

这是多请求吞吐研究，不是组委会当前单请求脚本的官方性能分；不能将
`output_tok_s` 直接换算为现有 91.67 分。单请求官方性能与精度结论仍以
[RESULTS.md](RESULTS.md) 为准。本轮没有把 DP2 smoke test 宣称为新的全量精度
结果。

本次运行产物不提交到源码仓库，验证机上的位置为：

```text
/public/home/tangyu408/Qwen_DCU_Worker_0/repro_minimal_validation/dp2/
  build-final.log
  dist-final/
  runtime/dp2/service-warm.log
  runtime/dp1/service-ab-warm.log
  results/dp2-c8-full-warm-o1024-n8-20260802/
  results/dp2-8-16k-scaling-warm-o1024-n8-20260802/
  results/dp1-c8-full-warm-o1024-n8-20260802/
```

提交边界只包含源码、脚本、测试和文档，不包含模型、testdata、wheel、cache、
服务日志或 benchmark JSON。
