# 600 行优化闭卷高收益速记

本文用于背诵时间不足时，从官方 vLLM 0.18.1 基线
`fa718036bdb9dfd80a872b86c8ac16c9d02bfd31` 重建主要性能优化。目标设备与模型固定为
单卡 gfx936、Qwen3.5-27B、BF16、TP/PP/DP=1。

本文不是逐行抄写清单。闭卷时只背四件事：**官方入口、目标 shape、fast path、官方
fallback**。600 行是 `csrc/ + setup.py + vllm/` 相对官方基线新增 567 行、删除 33 行的
churn，不是要求净写 600 行。

## 1. 时间不足时的取舍

按主要性能收益和实现价值排序：

| 优先级 | 优化 | 主要收益 | 闭卷策略 |
| ---: | --- | --- | --- |
| 1 | GQA6 prefill + page784 | 长上下文 TTFT，历史约降 16%--25% | 必须会入口、stride 和三态 LSE merge |
| 2 | K5120 GEMV + GateUp/SwiGLU | decode TPOT，局部约快 8% | 必须会 shape、640 threads 和融合顺序 |
| 3 | K17408 GEMV | decode TPOT | 背 1024 threads、16-wave 两级规约 |
| 4 | TunableOp + 静态 4096 | prefill/图内 GEMM | 代码短，优先保证能加载和 fail closed |
| 5 | GDN 固定 launch + fused norm | 小幅 TTFT/TPOT | 背配置表，不背官方数学 kernel |
| 6 | M-RoPE 连续 staging | 图稳定性，小幅收益 | 时间不够可最后实现 |

若只能完成两组，选 Attention 和 GEMV；不要先花时间重写 GDN 数学 kernel 或做
M-RoPE 微优化。

## 2. 通用改法：在官方函数中插窄分支

所有优化统一写成下面的结构：

```python
def official_function(...):
    # 保留官方的 dtype/rank/metadata 检查。
    if exact_target_gate(...):
        result = try_fast_path(...)
        if result is not None:
            return result

    # 保留官方主体，作为功能和非目标设备的 fallback。
    return official_path(...)
```

公共设备门只缓存 `gcnArchName.startswith("gfx936:")`。每个 fast path 仍须检查
model、BF16、精确 shape、stride/连续性、batch/prefill 条件和功能开关。helper 返回
`None`/`False` 表示不接管，而不是报错。

参考官方代码时使用函数锚点，不背易漂移的行号：

```bash
OFFICIAL=fa718036bdb9dfd80a872b86c8ac16c9d02bfd31
git show "$OFFICIAL":csrc/rocm/skinny_gemms.cu | less
git show "$OFFICIAL":vllm/model_executor/models/qwen3_next.py | less
git show "$OFFICIAL":vllm/v1/attention/backends/rocm_aiter_unified_attn.py | less
```

若 `git show` 报 `bad object`，先从 OpenDAS `vllm_cscc` 获取该提交。没有官方对象时
不要凭当前 HEAD 猜官方代码，也无法审计 600 行。

## 3. 第一收益：GQA6 prefill 与 page784

### 必须背的位置

- 官方宿主：
  `vllm/v1/attention/backends/rocm_aiter_unified_attn.py::RocmAiterUnifiedAttentionImpl.__init__/forward`
- 新窄算子：
  `vllm/v1/attention/ops/rocm_aiter_unified_attention_gqa6.py`
- 当前实现关键锚点：`page784_prefill`、`_gqa6`、`prefill`

### 必须背的 gate

```text
num_heads=24, num_kv_heads=4, head_size=256
BF16，单序列 prefill，max_query_len > 1
cache page shape=(784,4,256)
query/output contiguous
K/V cache shape、dtype、stride 相同
无 alibi/sliding-window/sinks/soft-cap
```

调用顺序：

```text
page784 later-prefill 成功 -> return output
否则目标 prefill          -> GQA6 Triton
任何 gate 不满足          -> 官方 AITER unified_attention
decode                    -> 官方 AITER
```

### GQA6 最短实现思路

一个 CTA 同时处理共享同一 KV head 的两个 Q heads：

```text
4 KV heads × 3 head groups = grid 第二维 12
每组 2 Q heads，共覆盖 24 Q heads
短 query: BLOCK_M=16, 4 warps, 2 stages
长 query: BLOCK_M=64, 4 warps, 1 stage
长块内部顺序处理两个 32-token subtile
```

循环中执行标准 online softmax：更新 `maximum`、用 correction 缩放旧 accumulator、
更新 denominator，再累加 `probabilities @ V`。不要背完整 Triton 语句，先依据官方
paged-attention 的 block table、causal mask 和 softmax 数据流重建。

最高风险点是 cache stride。地址必须使用运行时的：

```python
*key_cache.stride()
```

不能假设 cache contiguous，也不能把首维写成 `784*4*256`；服务中的 K/V 可能交错
存放。R23 的主要性能丢失就是 stride gate 写错，导致所有目标 prefill 静默 fallback。

### page784 只背“拆三份再合并”

```text
每页 784 = 768 main + 16 tail
历史 main      -> FlashAttention，返回 output A 和 LSE A
tail+boundary  -> 打包成 page64，返回 output B 和 LSE B
current K/V    -> causal attention，返回 output C 和 LSE C
最终           -> FP32 LSE 权重稳定合并 A/B/C
```

边界必须包括：`query_len >= 128`、`context >= 784`、`query_len <= 4096`、residual
pages 不超过 96、单请求且实际 token 数相等；否则返回 `False`。

## 4. 第二、三收益：K5120 与 K17408 GEMV

### 必须背的位置

- 官方 native ABI：`csrc/rocm/skinny_gemms.cu::LLMM1`
- 官方 Python GEMM 分派：
  `vllm/model_executor/layers/utils.py::rocm_unquantized_gemm_impl`
- MLP GateUp：`vllm/model_executor/models/qwen3_next.py::Qwen3NextMLP.forward`
- 公共分派：`vllm/model_executor/layers/fla/ops/gfx936.py::qwen35_gemv`

不要新增第二套 extension 或 binding。把 kernel 放在官方 `LLMM1` 前，在官方检查后
early return，保留原 `AT_DISPATCH_REDUCED_FLOATING_TYPES` 作为末尾 fallback。

### 必须背的 shape/launch 表

| Weight `(M,K)` | launch | 行组织 |
| --- | --- | --- |
| `(96,5120)` | 640 threads | 4 rows/CTA |
| `(14336/16384/34816/248320,5120)` | 640 threads | 2 rows/CTA |
| `(5120,17408)` | 1024 threads | 1 row/CTA |

共同条件是 `N=1`、权重和输入 BF16、连续输入。Python 通用线性层只有 `bias is None`
才尝试 GEMV。

### K5120 的实现记忆

```text
5120 / 8 = 640 个 float4 chunk
640 threads = 10 waves
后 320 lanes 写 LDS，前 320 lanes 相加
剩余 5 waves 各自 shuffle 规约
5 个 wave leader 再合并
```

`M=34816` 的 GateUp 用相同 ABI 的特殊 `rows_per_block=-2`：先算 17408 行 gate，
再算 17408 行 up，在第二次写回前执行 BF16-staged `SiLU(gate) * up`，避免单独激活
kernel 和 34816 维中间输出。只有 `expert_gate is None` 才尝试；失败调用
`super().forward(x)`。

### K17408 的实现记忆

```text
17408 / 8 = 2176 个 float4 chunk
1024 threads = 16 waves
每个输出行一个 CTA
wave 内 shuffle 规约
16 个 leader 写 16-float LDS
thread 0 做最终合并
```

两条路径都读取官方 BF16 权重、FP32 累加、BF16 输出；不生成量化或重排权重。

## 5. 短代码收益：静态 4096、TunableOp 与 GDN

### 静态 4096 与 profile

代码位置：

```text
vllm/platforms/rocm.py::set_device/apply_config_platform_defaults
vllm/v1/worker/gpu_worker.py::Worker.init_device
setup.py
vllm/platforms/tunable_profiles/gfx936_qwen3_5_27b_bf16_tn_m4096.csv
```

只有 gfx936、Qwen3.5、BF16、`max_num_batched_tokens=4096`、world/DP=1、无
speculative、用户未指定 compile sizes 时自动设置 `[4096]`。profile 必须在
distributed 选定设备后加载，关闭 online tuning/record，并 fail closed 验证恰好五行。

### GDN 只背 launch，不背数学 kernel

公共入口：`gfx936.py::gdn_kernel`。命中时解包 autotune wrapper 的 raw kernel 并注入
配置；不命中返回官方 wrapper 和空 options。

| kernel/长度 | 配置 |
| --- | --- |
| chunk-o T16 | BK32/BV32，2 warps，2 stages |
| chunk-o T32 | BK32/BV32，2 warps，3 stages |
| chunk-o T64 | BK32/BV64，4 warps，2 stages |
| chunk-o T4096 | BK128/BV128，4 warps，1 stage |
| scaled KKT T4096 | BK128，4 warps，1 stage |
| solve-tril/recompute T4096 | 2 warps，1 stage |

单 stage 再加 `waves_per_eu=1,matrix_instr_nonkdim=16,kpack=2`。decode 只对
`(B,H,HV,K,V)=(1,16,48,128,128)` BF16 使用 `BV=32,4 warps,1 stage`。

state 保留全无、全有、mixed 三态；只有全无 state 时传 `initial_state=None`。把输出
从 `zeros` 改为 `empty` 后，必须清零 padding tail。fused RMSNorm 只接管
`x=[T,48,128]`、BF16、`z.stride=(16384,128,1)`，否则调用官方 norm。

## 6. 官方 vLLM 上修改的实用技巧

1. 先用 `git show OFFICIAL:path` 阅读官方函数，再用 `rg` 找当前函数；不要整文件复制
   优化版，避免把实验状态机和无关改动带回去。
2. 每次只加一个 fast path，先完成 same-input 数值测试和非法 shape fallback，再叠加
   下一项；不要等 600 行全部写完才定位文本漂移。
3. 宿主层只做一次性/廉价 gate。token 循环中的 `view`、slice、GPU-to-CPU 读取和新
   tensor 分配都可能吃掉 kernel 收益。
4. ABI 能复用就复用：GEMV 使用官方 `_rocm_C.LLMM1`，GDN 使用官方 raw Triton
   kernel，Attention 宿主保留官方 AITER。这是压到 600 行的关键。
5. shape、stride 或功能条件拿不准时宁可 fallback；优化路径应 fail closed，而不是
   为了命中率放宽条件。
6. kernel allclose 只是第一关。Attention/GDN 必须覆盖完整生成文本，因为规约树、
   recurrent state 和 LSE merge 的微小差异可能改变 greedy decode。
7. 修改 native/Triton/Python 运行时后都构建新 wheel、使用新编译缓存冷启动，再用同一
   wheel/cache 热重启测试；不要把旧 `.so` 或旧 JIT cache 当成新代码结果。

## 7. 闭卷完成顺序与停止线

建议按以下顺序落地：

1. `_rocm_C`、共享 gfx936 gate 和所有 fallback；
2. K5120，再 K17408，并完成微基准；
3. GQA6 基础 prefill，先测 compact/interleaved stride；
4. page784 三态 LSE merge；
5. 五行 profile、静态 4096；
6. GDN prefill 固定 launch；
7. GDN decode/state/fused norm；
8. M-RoPE；
9. 三档小样本、全量吞吐和 110 条精度。

时间到达停止线时，宁可保留已验证的 Attention+GEMV，也不要提交未完成 fallback 或
没有文本验证的融合路径。

## 8. 最终十句背诵稿

1. 基线 `fa718036`，只在官方函数里插精确 gfx936 fast path。
2. 任意 dtype、shape、stride、功能条件不匹配都回官方实现。
3. 最大收益是 Q24/KV4/D256 的单请求 GQA6 prefill。
4. cache stride 必须运行时传入，绝不能按 page shape 写死。
5. page784 拆 768 main、16 tail/residual、current，再用 FP32 LSE 合并。
6. K5120 用 640 threads、十 wave 合五 wave，M96 四行，其余两行。
7. GateUp 两遍复用 K5120，第二遍融合 BF16-staged SwiGLU。
8. K17408 每行 1024 threads，16 wave leader 经 LDS 最终规约。
9. GDN 复用官方数学 kernel，只固定 launch；state mixed 和 padding 必须走安全语义。
10. 新源码必须新 wheel、新缓存冷编译、热重启，再测吞吐和完整文本精度。
