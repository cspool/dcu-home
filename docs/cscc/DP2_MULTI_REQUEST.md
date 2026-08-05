# 本机 DP=2 多请求复现与结果

## 结论与适用范围

本分支在保留原单卡路径的同时，增加了一个严格限定的双卡服务拓扑：同一台
机器、两个不同 gfx936 设备、`TP=1`、`PP=1`、`DP=2`、内部 `mp`
backend。每个数据并行副本仍运行完整模型；请求由 vLLM 的 DP frontend 分发，
因此不需要把已有单卡 kernel 改写为 TP kernel。

该路径继续使用 `max_num_batched_tokens=4096`、`max_num_seqs=128`、BF16、
continuous batching 和现有 GQA6/TunableOp/GEMV 优化。量化、prefix cache、
投机解码以及 draft/MTP 模型均未启用。Ray、外部 LB、hybrid DP、跨节点和
`TP>1` 不在本次支持范围内；专用 ROCm 路径对这些拓扑 fail closed。

初始 DP=2 实现提交为 `4c4aa45b2987e521a226a49d5978d592c440667e`；当前
等价压缩版由 runtime source manifest 和下述 clean wheel 哈希固定。原
`pra2026-bh408` 目录未被修改。

## 构建产物与静态验收

本次 clean build wheel：

```text
vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl
SHA-256: 1603b2ce5a77e04d6fdabce1aa6af9894ffc81ff8ad2d28ffd837afb8cb13465
990 行等价压缩版: 50f21c3a6a952be49d9cf5db19b0ec030796d310f8c99e48ad5bfe3b8ecb1d8d
499 行批量修复版: cb4db9cba095b0eaea14dc5b7dc2688a212c2e761163ef2ee40ba9080be76d7b
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
发现 rank1 在 rank0 之后编译时只能得到 23,520–24,304-token KV cache，而
rank0 为
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

### Triton DSL 简化回归

在保持上述 DP=2 拓扑、服务参数和 benchmark wrapper 不变的条件下，将
`(1,17408) @ (5120,17408).T` 从手写 HIP load/FMA/shuffle/LDS kernel 改为
Triton reduction 后，重新执行三档 concurrency 8：

| 输入档 | 成功 | Triton tok/s | 相对上表原 HIP | TTFT P99（ms） | TPOT P99（ms） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4–8K | 8/8 | 144.491555 | -0.518% | 6504.603082 | 53.393800 |
| 8–16K | 8/8 | 120.232380 | +0.759% | 15120.586033 | 63.283105 |
| 16–32K | 8/8 | 97.147597 | -0.939% | 28187.143597 | 75.780866 |

三档聚合吞吐为 `117.501163 tok/s`，原 HIP 为 `117.821297 tok/s`，差异
`-0.272%`。为避免把不同时段负载误判为 kernel 回退，又在同一时段、相同
wheel 构建参数、两条 warmup、两侧 28,224-token KV cache 的条件下复测最慢
的 16–32K 档：原 HIP 为 `97.600619 tok/s`，Triton 两次分别为
`97.147597` 和 `96.683946 tok/s`，差异为 `-0.464%` 与 `-0.939%`。

端到端波动均小于 `1%`；被替换 GEMV 的独立同卡微基准中，Triton 中位延迟
`0.130084 ms`，原 HIP 为 `0.131415 ms`，反而低 `1.013%`。结合单请求三档
TPOT P99 均未回退，本次只判定为性能等价，不把噪声声明为优化收益。

三份结果汇总 SHA-256：

```text
Triton 三档: ae1768ab0710d0045858b5402dc6ea2ae217a6539f0db646c91311128e4b6683
Triton 16–32K repeat: 54df183782235539ab361b8c016d9e9c2142405b56d454a52045eb9a321d9486
原 HIP 16–32K matched A/B: 5e02af3cf1af995bea35136c596d29c526499b81ee4a73b0ecf2c3ca02301647
```

DP=2 与单请求全量复测使用的 Triton clean wheel SHA-256 为
`4673386de52e3d396812f6242e2d67d790d8cf624e978b6b8108d0c0bf79698d`；补齐
非语义命名与文档后的最终空树构建为
`0c8bafdfd97f4301234b298961a498c4dbe82f3d88e92d5749b214feffb621e9`。
最终构建已通过 wheel 内容校验、三 seed 实际 Triton 分发检查和 20 项 DP=2
拓扑测试。

当前 499 行版本保持相同算子、调度和启动参数。2026-08-03 的针对性复验中，
两个 DP 副本均完成 mixed/decode 图捕获，4 条并发请求全部由两个 API 进程和
两个 GPU worker 成功处理；复测 4/4 HTTP 200，耗时 9.23 秒。

### 等价代码压缩最终回归

2026-08-03 的下表使用当时 990 行运行时补丁和 clean wheel
`50f21c3a6a952be49d9cf5db19b0ec030796d310f8c99e48ad5bfe3b8ecb1d8d`
重跑相同 concurrency 8 契约。冷启动时 rank0/rank1 分别为 28,224/23,520
tokens；使用同一缓存热重启后，两侧均直接加载 `rank_0_0`/`rank_0_1` AOT，
KV cache 均为 28,224 tokens，INIT/PRE_CAPTURE 各两次 `status=ready`。

| 输入档 | 成功 | duration（s） | tok/s | 相对压缩前 Triton | TTFT P99（ms） | TPOT P99（ms） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4–8K | 8/8 | 56.479923 | 145.042691 | +0.381% | 6363.756126 | 53.152582 |
| 8–16K | 8/8 | 70.127485 | 116.815824 | -2.842% | 16761.283472 | 63.610006 |
| 16–32K | 8/8 | 83.768009 | 97.793897 | +0.665% | 27829.366308 | 75.462878 |

三档 `24/24` 成功、失败 0。按固定总输出 `24576 / 三档 duration 之和`，聚合
吞吐为 `116.819733 tok/s`，相对压缩前 `117.501164 tok/s` 为 `-0.580%`。
8–16K 首轮存在单个请求完成长尾；同一热服务立即重复为 `119.589957 tok/s`，
相对压缩前 `-0.534%`，TPOT P99 从 `63.283105` 降至 `62.554961 ms`。用该
匹配重复替换长尾样本后，三档聚合为 `117.730093 tok/s`，相对压缩前
`+0.195%`。因此三档均有同条件小于 1% 的匹配证据，不改变性能等价结论。

```text
当前三档 summary: a33c7bf222d950c072af451d74b4398151936164dfeb912bb0c1697b49ab828b
8–16K repeat result: b1a29042cb4327af332c851812a7bc26e2dd6a892c4fa9779776aa1b1044bc7e
```

### 499 行最终双卡全矩阵回归

2026-08-05 使用空 build tree 构建上述 499 行批量修复 wheel，并在两个 rank
均为 28,224-token KV cache、`gpu-memory-utilization=0.95` 的热服务上执行
concurrency 2/4/8 与三档输入的完整笛卡尔积。九个 case 均为 8 条正式请求、
每条固定输出 1024 token：

| 并发 | 4–8K tok/s | 8–16K tok/s | 16–32K tok/s | 成功 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 48.040019 | 44.777397 | 41.517538 | 24/24 |
| 4 | 81.989201 | 68.676498 | 63.618960 | 24/24 |
| 8 | 144.800996 | 115.695862 | 96.358738 | 24/24 |

合计 `72/72` 成功、失败 0、无 OOM。concurrency 8 的 8–16K 首轮存在完成
长尾；同一热服务立即重复为 `119.756462 tok/s`，8/8 成功。相对 2656 行
最优实现的匹配参考，concurrency 8 三档依次为 `99.695%`、`100.360%`、
`98.257%`，几何平均 `99.433%`；8–16K 的 concurrency 2/4 分别为
`100.348%` 和 `97.725%`。所有匹配点均高于 0.95 性能门禁。

全量精度前的 8 并发定向测试曾发现压缩版 packed GDN decode 的第二维 grid
仍固定为 48，只覆盖 batch 中第一个序列。最终实现改为
`batch_size * 48`；B=1/2/4/8 的 GPU 对照中输出最大绝对误差不超过
`3.7e-9`、状态误差不超过 `2.38e-7`，随后 3 轮 8 并发 Retrieval 请求
`24/24` exact。该修复没有改变算法、KV cache 或调度参数。

完成上述九 case 压力测试后，在同一双卡服务上运行固定全量精度脚本：

| 数据集 | 请求 | 499 行版 | 结果 |
| --- | ---: | ---: | --- |
| HotpotQA | 20/20 | 77.96 | pass |
| GovReport | 30/30 | 32.76 | pass |
| Retrieval Multi Point | 30/30 | 100.00 | exact |
| Aggregation Keyword | 30/30 | 100.00 | exact |

总计 `110/110`、API 错误 0，未发现异常重复字符。HotpotQA prediction 与先前
正确最优版逐字节一致；GovReport 为官方基线的 `99.390%`；两个 RULER 项均
按多重集合独立复核为 30/30，最终四项精度系数均为 1。

```text
九 case summary: 52ffd9554e17c827c1dfbaa18dfcf7ebb3a1329919aa8d5a6807c0ea88f4bf44
8–16K hot repeat: e01aa91ee2e92a6d1628fa50ddde9680a0811c6ac4c5bbb39ebe7a8cc4bf6c21
精度 summary CSV: 7ec8395ad26e33366be077773ef3292b022c1763b462985ce744ea3c194dd755
```

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
`output_tok_s` 直接换算为现有 91.08 分。单请求官方性能与精度结论仍以
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
/public/home/tangyu408/Qwen_DCU_Worker_0/repro_minimal_validation/dsl_simplification/
  dist/
  results/dp2-c8-full-warm-o1024-n8-20260802/
  results/dp2-dsl-repeat2-16-32K-c8-20260802/
  results/dp2-hip-ab-16-32K-c8-20260802/
/public/home/tangyu408/Qwen_DCU_Worker_0/repro_minimal_validation/dsl_simplification-final/
  build-current.log
  dist-current/
```

提交边界只包含源码、脚本、测试和文档，不包含模型、testdata、wheel、cache、
服务日志或 benchmark JSON。
