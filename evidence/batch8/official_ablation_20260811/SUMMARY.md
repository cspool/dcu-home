# DP2 全局 Batch=8（每 rank 目标 B=4）官方相对消融

日期：2026-08-11

## 口径

- 官方源码基线：`fa718036bdb9dfd80a872b86c8ac16c9d02bfd31`。
- 迁移分支：`ca28376909970b447fd6af15c7cdb9a64ff6a6ee`。
- 服务拓扑：TP=1、DP=2、全局并发 8；均衡时每个 rank 为 B=4。请求完成时间不同会短时出现 5+3、3+4 等分布。
- 算子消融均在 gfx936 BF16 上执行，候选与对照路径交替计时；除明确标注的当前 BM fallback 和 M-RoPE traffic model 外，对照均为官方路径。所有记录的数值检查均通过。
- 代码量按官方源码到当前分支的 runtime churn 计算：`+471/-30 = 501` 行（含 `setup.py`；不含文档、脚本和测试）。

历史同协议的组合端到端边界为：4–8K `125.11→144.83 tok/s`（+15.76%）、8–16K `79.33→119.74`（+50.94%）、16–32K `48.05→97.58`（+103.08%）。这是旧完整优化栈对官方的组合结果，不能用于给单个优化归因。

## 消融结论

| 优化点 | 对照路径 | 迁移候选 | 对照相对结果 | B8 结论 |
|---|---:|---:|---:|---|
| page784 GQA，local B4，BM16→BM32 | 6791.14 µs | 4760.32 µs | -29.90% | 保留 |
| page784 GQA，local B4，AITER 通用实现→BM32（已有匹配实验） | 13565.59 µs | 5072.32 µs | -62.61%，2.674x | 保留 |
| GDN `chunk_o`，总 T=4096 | 446.78 µs | 353.38 µs | -20.90% | 保留 |
| GDN KKT 固定配置 | 101.95 µs | 111.39 µs | +9.26% 变慢 | 回官方配置 |
| GDN solve 固定配置 | 201.92 µs | 199.52 µs | -1.19% | 可保留，低优先级 |
| GDN recompute 固定配置 | 285.79 µs | 288.95 µs | +1.10% 变慢 | 回官方配置 |
| GDN strided-z RMSNorm+SiLU，local B4 | 212.16 µs | 64.90 µs | -69.41% | 保留 |
| GDN packed decode 4w1s，local B4 | 官方 1w3s 30.72 µs | 38.07 µs | +23.92% 变慢 | B4 禁用 |
| K17408 B4 GEMV | `F.linear` 156.48 µs | 自定义 519.82 µs | +232.19% 变慢 | 被 B4 GEMM 覆盖 |
| K5120 B4 GEMV，5 个真实输出维度 | `F.linear` | 4 个 M1 kernel | 慢 206.69%–232.16% | 被 B4 GEMM 覆盖 |
| M=4096 TunableOp，5 个真实 GEMM | 默认 rocBLAS | 固定 profile | -2.78%–8.44% | 保留 |
| 新 prefill state，local B4 | gather+mask 64.37 µs | `initial_state=None` 1.07 µs 事件下限 | 避免约 63.3 µs、12 MiB 搬运 | 保留 |
| T=4096 GDN 输出复用 | 48 MiB D2D 78.00 µs | 直接写调用方 buffer | 避免约 77.0 µs | 保留 |
| B4 decode padding 清零 | 全 4096 行 43.32 µs | 尾部 4092 行 43.90 µs | 无收益 | 被固定图 padding 覆盖 |
| 连续 M-RoPE staging 数据搬运模型 | 83.20/96.64 µs | 12.25/13.89 µs | -85.3%/-85.6% | 保留，但非端到端归因 |

GDN 四个 T=4096 固定配置全开时，算子和为 `1036.44→953.24 µs`（-8.03%）。只保留正收益的 `chunk_o` 与 solve、KKT/recompute 回官方，预计为 `940.64 µs`（相对官方 -9.24%，相对当前全开再省 1.32%）。

## local batch gate

### GDN packed decode

| local batch | 官方 1w3s | 迁移 4w1s | 迁移变化 |
|---:|---:|---:|---:|
| 1 | 12.63 µs | 11.15 µs | -11.70% |
| 2 | 21.36 µs | 16.29 µs | -23.74% |
| 3 | 26.58 µs | 21.55 µs | -18.92% |
| 4 | 30.72 µs | 38.07 µs | +23.92% |
| 5 | 29.55 µs | 58.37 µs | +97.53% |
| 6 | 32.67 µs | 54.84 µs | +67.83% |
| 7 | 37.16 µs | 60.58 µs | +63.03% |
| 8 | 42.27 µs | 65.06 µs | +53.89% |

正确 gate 是当前 rank 的 active batch：`B<=3` 使用 4w1s，`B>=4` 使用官方 1w3s。它不是输入长度 gate。

### page784 GQA BM32

同一组 page784 混合长度输入下，BM32 相对当前 fallback：B3 `-36.50%`、B4 `-29.90%`、B5 `-25.78%`。B3/B5 的 q_len 8/16/64/256/约 1024 补充网格共 10/10 点为正：B3 降低 `10.77%–42.26%`，B5 降低 `11.24%–42.63%`。额外 B1–B8 屏幕点也全部为正，但 B1/2/6/7/8 尚未补齐同样网格。

对只支持 DP2 全局 batch8 的版本，安全的第一步是把当前 `num_sequences==4` 扩为 local B=3/4/5，以覆盖稳态 4+4 与常见瞬态 5+3；现有 B4 q_len 8–1024 数据均为正，因此不需要再叠加输入长度阈值。

## 哪些收益被 B4 覆盖

1. 单 token K5120/K17408 GEMV：M 从 1 变成 4 后，官方 GEMM 可复用权重并产生足够并行度；强制拆回四个 M1 kernel 会重复读取权重。
2. packed GDN decode 的 4w1s：B4 已把 grid 扩为 192 个 head workgroups，继续增加每个 workgroup 的 warp 数反而争抢资源。
3. decode 的“只清 padded tail”：B4 仍需清 4092/4096 行，和官方全清相同。
4. 完整 warmup：它减少冷编译尾延迟，不提升热态稳态吞吐，不能计入热路径收益。

## 官方相对代码量与取舍

下表是按 diff hunk 归类的近似 runtime churn；共享的 gfx936 gate/import 只在 501 行总数中计算一次，因此各项不能机械相加。

| 归类 | 官方相对 churn | 建议 |
|---|---:|---|
| GQA kernel + backend 分发 | 约 143 行核心；含自动 backend 路由约 148 行 | 保留，最大 prefill kernel 收益 |
| GDN prefill 配置与 warmup | 约 64 行 | 只保留 `chunk_o`/solve 正收益配置 |
| GDN state/output 数据搬运 | 约 45 行 | 保留，B4 prefill 仍直接省搬运 |
| GDN RMSNorm+SiLU | 约 32 行 | 保留，代码少且 B4 算子收益大 |
| packed GDN decode 专用调度 | 约 23 行 | 改为 B<=3 gate；若严格只跑稳态 B4，可删除 |
| K5120/K17408 decode GEMV | 约 123 行 | B8-only 版本删除，收益已被 B4 覆盖 |
| M=4096 profile + loader | 约 15 行 | 保留，代码/收益比最好 |
| M-RoPE staging | 28 行 | 暂保留；需要单变量服务 A/B 后再决定 |

推荐的 B8-only 最小路径：GQA BM32（local B3/4/5）、GDN RMSNorm、GDN 新 prefill state/output 复用、`chunk_o` 正收益配置、M=4096 profile；删除 B4 GEMV 迁移，修正 packed decode 和 GDN KKT/recompute gate。长上下文 16–32K 的剩余主要问题是 DP/KV 容量、抢占和长度混排，应该由长度感知调度处理，不应把输入长度塞进已经稳定为正的 GQA kernel gate。

## 原始结果

- `raw/gqa_local_batch_bm32_b1_b8.json`
- `raw/gqa_b3_b5_qlen_sweep.json`
- `raw/gdn_decode_b1_b8.json`
- `raw/gdn_rmsnorm_b4.json`
- `raw/gdn_schedule_t4096_official_rerun.json`
- `raw/decode_gemv_b4.json`
- `raw/m4096_tunable_b4.json`
- `raw/gdn_data_movement_b4.json`
- `raw/mrope_staging_b4.json`

所有 GPU 测试结束后，两张卡 HCU 利用率和显存占用均回到 0%。
