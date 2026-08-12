# DP2 全局 Batch=8 / 单 rank Batch=4 验证摘要

日期：2026-08-11

状态：通过。新目录只新增 DP2 全局并发 8 的目标优化；均衡调度时每个 rank 为 4 条序列。正式吞吐测试 3 个长度桶共 150/150 请求成功，预热 8/8 成功，无 HTTP 非 200、OOM、Traceback 或 kernel error。服务测试后已停止，两卡显存占用均回到 0%。

## 口径与来源

- 源码目录：`/public/home/tangyu408/Qwen_DCU_Worker_0/pra2026-bh408-gqa-page784-k5120-batch8`
- 分支：`repro-gqa-page784-k5120-batch8`
- 最终 HEAD：`ca28376909970b447fd6af15c7cdb9a64ff6a6ee`
- kernel 提交：`79a20b0494abff0611419f85d2b15d852cfe3f25`
- 历史全量基线：`gqa_page784_k5120_final_dp2_5355cea_20260811`，源码提交 `5355ceab41a0801c611d5267ef6e36eefdcd4c77`
- 服务参数：DP=2、TP=1、`block_size=784`、`max_num_batched_tokens=4096`、全局 `max_concurrency=8`
- wheel 从隔离的 `site/` 加载，未从源码工作区导入。

DP=2 表示两个数据并行副本，不表示 batch=2。全局同时 8 请求进入时，监控实测 `engine0=4`、`engine1=4`；不同长度请求先后完成后会瞬时变成 5+3、3+4 等，但全局上限仍为 8。

## 实现与 gate

目标 gate 为 `cache_size == 784 && num_sequences == 4`。命中后使用：

- `BLOCK_M=32`
- `num_warps=2`
- `num_stages=1`
- `waves_per_eu=1`
- `matrix_instr_nonkdim=16, kpack=2`

运行时编译证据显示该变体使用 16 KiB LDS/工作组；旧 BM16 变体为 8 KiB，batch=1 的 BM64 变体为 32 KiB。BM32 相比旧 BM16 将 program 数减半，同时只使用 batch=1 路径一半的 LDS，保留并发余量。其他 rank 内序列数和 page size 继续走原有配置。

没有再加输入长度阈值：rank 内 B4 的 BM32 在已测 q_len 8、16、64、256、1024 和混合长度上均优于旧 tile。长度变化需要动态处理的是 KV 驻留与调度，不应伪装成 GQA kernel gate。

## 隔离 microbenchmark

真实 rank 内 B4、page784、Hq=24、Hkv=4、D=256；混合 `q_lens=[16,32,64,128]`、context `[4096,8192,16384,24576]`：

| 实现 | 延迟 |
|---|---:|
| 新 BM32 | 5072.315 us |
| 旧 BM16 | 6903.355 us |
| AITER 通用实现 | 13565.589 us |

新 tile 相对旧 tile 为 1.361x，延迟降低 26.5%；相对 AITER 为 2.674x。输出对旧 tile 和 AITER 均 `allclose(rtol=0.02, atol=0.02)`，最大绝对误差 `4.8828125e-4`。跨 q_len 8–1024 的已测点相对旧 tile 均为正收益，约 6.3%–42.1%。

K5120 的 rank 内 B4 实验没有保留：最佳自定义 Triton 为 312.44 us，仍慢于 `F.linear` 的 304.05 us（慢 2.76%）；低 LDS HIP 方案慢 17%–56%。GDN 当前 packed decode 已天然覆盖 B4，参数扫描没有稳定收益。因此只提交有实测正收益的 GQA 改动。

## DP2 c8 端到端结果

所有正式 case 使用与历史全量结果相同的数据、50 prompts、`request_rate=inf`、`max_concurrency=8`、输出上限 1024、`ignore_eos=false`、2 个 warmup。

| 长度桶 | 成功 | 输出吞吐 old→new | 变化 | 时长 old→new | TTFT p50 old→new | TPOT p50 old→new |
|---|---:|---:|---:|---:|---:|---:|
| 4–8K | 50/50 | 75.31→88.08 tok/s | +16.97% | 164.58→141.33 s | 3124→2013 ms | 73.20→66.84 ms |
| 8–16K | 50/50 | 45.64→62.12 tok/s | +36.11% | 298.48→219.51 s | 8248→4448 ms | 111.60→85.56 ms |
| 16–32K | 50/50 | 23.93→24.51 tok/s | +2.44% | 493.24→541.58 s | 15354→8247 ms | 229.58→118.04 ms |

4–8K 和 8–16K 的总 token 吞吐分别提升 16.47% 和 35.98%。16–32K 的总 token 吞吐下降 8.80%，TTFT p95 从 47.60 s 增至 69.57 s；该桶仍由单卡 KV 容量和抢占尾延迟主导。本次长桶生成 13,276 tokens，历史结果为 11,803，故总时长不能作为等 token 数的纯 kernel A/B；中位 TTFT/TPOT 改善，但尾延迟没有解决。

历史全量结果与当前分支不是只差 BM32 的单变量提交，因此端到端表只能作为整条分支的回归对比；BM32 的独立收益以上述 microbenchmark 为准。

## 完整性与哈希

- `python3 setup.py build_ext --inplace`：通过
- K5120 operator gate/numerics：通过；rank 内 B4 保持 `F.linear`
- `scripts/verify_cscc_repro.sh <wheel>`：PASS
- 运行时源码 churn：501/600（471 additions + 30 deletions）
- wheel SHA256：`5a3464936a7e2864aa9adea4099c6abb027ee7a5a776250faa061cc8bba46d9a`
- `_rocm_C.abi3.so` SHA256：`81bc1c5a9a7ba409330c50344dad9f3c7cfe5131c18e89797dd828cb7f9ab502`
- GQA 源 SHA256：`d7d3b3ed9d4df38caee643551c2f6dff4942c6b2a16e57d35691cbb218dd143f`
- 4–8K 结果 SHA256：`04742878fdd713cafdbcc6735e98486dea74cdd2d07fcccf8d8102047a1b4db3`
- 8–16K 结果 SHA256：`7f8baa61a175c5e69839b76082cbf00d763acfb18ba9d7763f11ed4371653a5d`
- 16–32K 结果 SHA256：`19724faf526971bd22b228e4b72bbbc13e172743a7bfb06bd2037e6eb7aa9d4a`

原历史全量测试另包含 110/110 accuracy 请求和 450/450 throughput 请求，均成功；当前 batch8 适配验证没有重跑整套 accuracy，只做了 kernel 数值一致性和 c8 三桶吞吐回归。
