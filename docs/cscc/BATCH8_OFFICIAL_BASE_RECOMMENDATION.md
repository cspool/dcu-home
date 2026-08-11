# 基于官方原版的 DP2 Batch=8 优化建议与实现状态

> 更新日期：2026-08-11
> 官方基线：`fa718036bdb9dfd80a872b86c8ac16c9d02bfd31`
> 工作树起点：`ca28376909970b447fd6af15c7cdb9a64ff6a6ee`
> 目标：gfx936、Qwen3.5-27B BF16、TP1/DP2、全局并发 8，稳态单卡约 4 请求

本文中的收益和代码量全部相对官方基线计算。算子微基准不能相加为端到端收益；组合端到端结果也不能反向归因给单个 kernel。

启动脚本和测试脚本没有修改。最终交付需始终满足：

```bash
git diff --exit-code HEAD -- scripts tests
```

## 最终建议

推荐在官方原版上保留两类改动：

1. **高置信、窄 gate 的 kernel/数据搬运优化**：page784 GQA、GDN RMSNorm、GDN state/output 复用、B1-B3 packed decode、M4096 Tunable profile。
2. **比继续迁移自定义 kernel 更重要的 prefill 保护**：目标配置下缩小 CUDA Graph 捕获档位；只要仍在 prefill，就按已知输入长度收缩单步 token 预算。

不建议迁移 B4 GEMV，也不建议继续扩展 GQA/GDN 的输入长度 gate。B4 已让官方 GEMM/packed GDN 获得足够并行度；长请求的主要问题是 MLP 临时张量和显存峰值，应由调度预算处理。

## 当前实现状态

| 项目 | 官方基线上的证据或目的 | 当前状态 | 最终行为 |
|---|---|---|---|
| page784 GQA BM32 | local B3/B4/B5 的 q_len 网格 10/10 为正，约快 10.8%-42.8% | **已实现** | page784 且 local batch 为 3-5 时使用 BM32；不按输入长度开关 |
| GDN RMSNorm | B4 算子约快 62%-69% | **已实现** | 仅 gfx936、Qwen3.5 固定 shape、BF16 命中，否则官方 fallback |
| GDN state/output 复用 | 避免约 63-75 us state 搬运和 65-78 us 输出 D2D | **已实现** | uniform state 直传，prefill 直接写调用方 output |
| GDN `chunk_o` 调度 | T4096 固定配置局部约快 20.9%，但最终 prefill 预算最高 2048，目标服务不再命中 | **已删除固定配置** | 恢复官方不含 `T` 的 autotune cache key，同一 H/K/V/BT 只调优一次；保留 output 复用 |
| packed GDN decode | B1/B2/B3 为正；B4 及以上回退 | **已实现** | local B<=3 用 4-wave/1-stage，B>=4 用官方实现 |
| B4 K5120/K17408 GEMV | 强制自定义约为官方 `F.linear` 的 3.1-3.3 倍 | **已删除迁移** | 所有 batch 使用官方 `skinny_gemms.cu`/框架路径 |
| M4096 Tunable profile | 五个真实 GEMM shape 约快 1.9%-8.4% | **已实现** | 保留官方 TunableOp 机制和窄 profile |
| M-RoPE staging | 搬运模型为正，但没有独立服务 A/B | **已实现但未单独归因** | 保留；不把流量模型记为端到端收益 |
| CUDA Graph 显存保护 | ROCm 在 KV 分配前未计入实际 Graph memory | **已实现** | 精确目标配置、且用户未显式指定档位时，最大捕获 size 为 16 |
| 长度分档 prefill 预算 | 防止长 prefill 物化过大的官方 MLP 中间张量 | **已实现；并发吞吐及准确率全量通过** | 输入 >16K 用 512，>8K 用 1024，其余 prefill 用 2048；纯 decode 保持官方预算 |
| DP rank 长度 tie-break | 混合长度时可能改善两卡尾差 | **未实现** | 多 API 前端下没有简单的全局 token 计数；在禁止修改启动脚本的约束下保持官方 DP 分发 |

小算子消融原始数据位于：

- `/public/home/tangyu408/Qwen_DCU_Worker_0/batch8_implementation_validation_20260811/small`
- `/public/home/tangyu408/Qwen_DCU_Worker_0/batch8_official_ablation_20260811/SUMMARY.md`

## 动态调度的最小策略

这里的“动态调度”是 EngineCore 内每一步的 **prefill token 预算**，不是请求跨卡迁移，也不是修改测试参数。

官方外部上限仍为：

```text
max_num_batched_tokens = 4096
```

只有下列条件同时成立时才收缩本步预算：

- 精确 Qwen3.5-27B text shape：hidden 5120、intermediate 17408；
- gfx936、BF16、TP1/DP2、`max_num_seqs=128`、官方 4096 token 上限；
- 至少一条当前请求 `num_computed_tokens < num_prompt_tokens`，即仍处于 prefill。

策略只有三档：

```python
if max_prompt_tokens > 16384:
    prefill_budget = 512
elif max_prompt_tokens > 8192:
    prefill_budget = 1024
else:
    prefill_budget = 2048
```

纯 decode 每条请求通常只调度一个 token，不需要缩预算。所有 prefill 都限到 2048/1024/512，避免 DP2 分流后每卡只有一条请求而绕过保护。此策略不等待凑 batch、不迁移运行中请求、不读取实时 DCU 指标，也不新增跨进程共享状态。

### 官方输入长度是否提前可知

未来尚未到达的请求当然不可知，但当前请求在进入 EngineCore 前已经完成 tokenization。此时 `Request.num_prompt_tokens`（等价于当前请求 prompt token 数）已经存在，因此调度器可以使用当前在途请求的真实 token 长度，不需要测试脚本传 `input_len`。

这与 kernel gate 分工明确：

- 输入长度只用于控制并发 prefill 的资源峰值；
- GQA gate 只看 page、local batch、dtype/shape；
- GDN decode gate 只看 local batch 和固定 shape。

### 为什么没有实现 DP rank 长度 tie-break

现有启动脚本会自动创建两个 API frontend。每个 frontend 只能看到自己的请求，简单的本地 `inflight_prompt_tokens[rank]` 不是全局状态。要做精确的 DP rank 长度 tie-break，需要单 frontend、跨进程共享计数或协调器协议修改。

前两者会修改启动方式，后者明显扩大代码和风险。当前约束禁止修改启动脚本和测试脚本，因此保留官方 DP rank 选择；长度信息只在每个 EngineCore 内用于安全预算。

## OOM 消融与大样本结果

测试协议来自未修改的 `scripts/bench_cscc_multi_request.sh`：全局并发 8、50 prompts、每条固定输出 1024、ignore EOS、2 次 warmup、无限请求速率。

### 失败定位

前两类 OOM 不是自定义 GQA/GDN 数值错误，而是官方 dense MLP：

- 一次 4094-token mixed prefill/decode 需要 gate/up 输出约 `(4094, 34816)` BF16，即约 272 MiB 连续空间；
- 失败时设备 free 为 0，PyTorch 仍有约 500 MiB reserved-but-unallocated，表现为碎片和峰值共同问题；
- ROCm 不支持本环境尝试的 `expandable_segments`，不能依赖 allocator 环境变量修复。

准确率流程还暴露出两个单请求边界：

- DP2 把两条请求分到两卡后，每个 EngineCore 可能只看到一条请求；仅以“多请求”作为 gate 会让约 15.9K 输入仍一次调度 3582 token，申请 238 MiB 后 OOM；
- 即使输入只有 6760 token，单请求 4096-token prefill 仍会为官方 MLP 申请约 272 MiB，因此最终去掉请求数条件，所有 prefill 都使用长度三档。

第三类 OOM 来自在线调优而非模型计算：旧迁移为优化 T4096，把 `T` 加入 `chunk_o` 的 Triton autotune key。Aggregation 最后一批遇到新 T 时，autotuner 为 benchmark 清空 L2 临时申请 256 MiB 并 OOM。最终恢复官方 `key=[H,K,V,BT]`，删除已无法命中的 T4096 pruner；不同输入长度复用同一官方调优结果。

### 逐步消融

| 版本 | 长度桶 | 结果 | 结论 |
|---|---|---:|---|
| Graph cap=16，无 prefill 预算 | 4-8K | 27 成功 / 23 失败 | Graph 从约 0.27 GiB 降至 0.13 GiB，但单独不足 |
| Graph cap=16，并发 prefill 固定 2048 | 4-8K | 50/50，111.67 output tok/s | 4-8K 足够 |
| Graph cap=16，并发 prefill 固定 2048 | 8-16K | 39 成功 / 11 失败 | 长 KV 下仍不够 |
| Graph cap=16，长度三档 | 4-8K | 50/50，111.67 output tok/s | 使用 2048 档，无失败 |
| Graph cap=16，长度三档 | 8-16K | 50/50，98.06 output tok/s | 使用 1024 档，无失败 |
| Graph cap=16，长度三档 | 16-32K | 50/50，58.87 output tok/s | 实际输入约 20.5-22.4K，使用 512 档，无失败 |

结果目录：

```text
/public/home/tangyu408/Qwen_DCU_Worker_0/
  batch8_implementation_validation_20260811/throughput_after_graph_cap/
```

### 最终准确率

使用未修改的准确率脚本、原数据全量、`MODEL_DIR=/root/Qwen3.5-27B`，逐数据集运行。四项结果均来自最终 Python/kernel 源码（官方 `chunk_o` cache key + 全 prefill 长度三档）：

| 数据集 | 样本数 | 指标 | 结果 | 输出目录时间戳 |
|---|---:|---|---:|---|
| HotpotQA | 20 | score | 77.96 | `20260811_213526` |
| GovReport | 29 | score | 32.69 | `20260811_213941` |
| retrieval_multi_point | 30 | accuracy | 100.00 | `20260811_213659` |
| aggregation_keyword_aggregation | 30 | accuracy | 100.00 | `20260811_213210` |

输出根目录：

```text
/public/home/tangyu408/Qwen_DCU_Worker_0/
  batch8_implementation_validation_20260811/accuracy/accuracy_debug/
    output/local_accuracy_qwen35/
```

原生扩展审计发现，工作区当时已有的 `_rocm_C.abi3.so` 早于恢复后的官方
`skinny_gemms.cu`。因此又从当前源码调用未修改的 `scripts/build_cscc_wheel.sh`
重建 wheel，并从仓库外启动服务，避免 Python 优先导入工作区旧 `.so`：

```text
wheel:
  batch8_implementation_validation_20260811/final_wheel/
  vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl
sha256:
  a0ad2c6204d26d8b699078f0b499ca915d7d6aa07f068bdb42e3ac7e7a733b4b
loaded _rocm_C:
  /usr/local/lib/python3.10/dist-packages/vllm/_rocm_C.abi3.so
```

最终 wheel 再跑完整 HotpotQA 20 条，仍为 **77.96**，输出时间戳
`20260811_221024`。全局 batch=2 在 DP2 下覆盖了每 rank batch=1 的官方
skinny GEMM 路径；服务端记录 20 个 HTTP 200。最终日志
`server_final_wheel_smoke_20260811.log` 中无 OOM、500、Traceback、ERROR 或
worker fatal，停止后两卡显存均回落到 2 MiB。

## 哪些单 batch 优化仍对 batch8 有效

保留：

- page784 GQA：local B3/B4/B5 仍显著为正；
- GDN RMSNorm：独立逐 token/逐 head，B4 仍有收益；
- GDN state/output 复用：B4 prefill 仍直接减少搬运和峰值；
- `chunk_o` output 复用，以及官方不按 T 重复在线调优的 cache key；
- M4096 profile：代码很少，单请求或可形成大 M 时仍可命中；
- packed decode 的 B1-B3 尾部，用于 5+3、请求收尾等瞬态。

被 B4 官方路径覆盖：

- K5120/K17408 M1 GEMV：B4 官方 GEMM 复用权重，自定义拆分反而重复读权重；
- packed decode 4-wave：B4 已有足够 workgroups，更多 wave 争抢资源；
- 只清 decode padding tail：B4 仍需清几乎完整的 4096 行；
- KKT/recompute 固定 schedule：实测回退，恢复官方 autotune。

## 代码量与取舍

当前官方基线到工作树的 runtime diff（交付所需 `setup.py + vllm`；`csrc` 最终与官方一致）为：

```text
+387 / -27 = 414 行 churn
```

不计 docs、scripts、tests；预算上限为 500 行。当前仍有 86 行余量，但不建议为了用满预算继续增加 kernel。

调度/OOM 修复本身只涉及两个已有框架文件：

- `vllm/platforms/rocm.py`：目标配置判定和 Graph cap；
- `vllm/v1/core/sched/scheduler.py`：基于已 tokenized 长度的本步预算。

没有新依赖、没有跨卡通信、没有测试协议耦合。相对继续迁移自定义 kernel，这一方向代码更少，且直接把端到端结果从服务崩溃变为完整可测，因此优先级更高。

## 验收原则

1. `scripts/` 和 `tests/` 相对工作树起点必须零 diff。
2. 精确 gate 外必须保持官方 fallback；用户显式指定 Graph capture sizes 时不得覆盖。
3. 小算子数值检查必须通过；端到端 accuracy 必须使用官方提供数据和原测试流程。
4. 三个长度桶均需完成 c8/50、0 failed 后，才能宣称 OOM 修复完成。
5. 吞吐比较必须标明是否 ignore EOS、prompt 数、长度桶和并发；不能把失败运行的部分吞吐当成正式收益。
