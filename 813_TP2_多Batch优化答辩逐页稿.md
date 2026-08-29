# Qwen3.5-27B：从单 Batch / TP=1 到多 Batch / TP=2 的优化演进

> 答辩 PPT 逐页内容，共 6 页：GDN、GQA Attention、MLP GEMV 各 2 页。  
> 第 1/3/5 页介绍初赛单 Batch、TP=1 的优化；第 2/4/6 页介绍决赛如何适配多 Batch、TP=2，并突出优化原因与性能来源。

## 统一口径（不单独占 PPT 页）

- 模型：Qwen3.5-27B，BF16，64 层；其中 48 层为 GDN 线性注意力、16 层为全量 GQA Attention，64 层均包含 Dense MLP。
- 初赛目标：单 Batch、TP=1，在 gfx936 上针对固定模型 shape 做专用 fast path。
- 决赛目标：TP=2、最大并发 Batch=10、`max_num_batched_tokens=4096`。TP=2 后每张卡只计算一半的 head、MLP 中间维度和词表分片；多 Batch 又使每一步的 token 数动态变化。
- 基线代码：`vllm_cscc/`，commit `fa718036bdb9dfd80a872b86c8ac16c9d02bfd31`。
- 决赛代码：`813/`，branch `single2`，commit `7b469aa90dc19a6006f6258fd6be4bdae4f94f2f`。
- 核心设计原则：**固定 shape 命中专用 kernel；动态 Batch 或不支持的 layout 继续走官方 AITER/GEMM fallback。** 决赛优化不是让单 token kernel 强行处理所有 Batch，而是按工作负载选择更合适的计算路径。

---

## 第 1 页：GDN 初赛优化——融合高频、小维度的 Norm + Gate

### 页面标题

**GDN：48 个线性注意力层的高频 Epilogue 定形优化**

### 页面主结论

GDN 核心递推保持官方实现，只接管输出端的 `RMSNorm + SiLU Gate`。官方路径已经用一个通用 Triton kernel 融合二者；初赛优化的关键是在固定 `[T,48,128]` shape 上进一步定形化，减少通用路径的辅助开销、额外 `rstd` 写回和 CTA 数量。

### 页面建议布局

左侧放“优化前/优化后”数据流，右侧放“为什么有效”和“性能来源”。

### 页面可见内容

#### 1. 优化前：通用算子处理固定 shape

```text
GDN input projection
        ↓
official gdn_attention_core
        ↓
reshape → 通用 fused RMSNormGated(x,z) → reshape
        ↓
out_proj
```

- 目标张量固定为 `core_attn_out = [T, 48, 128]`。
- 原路径需要 reshape，并通过支持 bias、group norm、多种 activation 的通用 RMSNormGated kernel 完成归一化与门控。
- 通用 kernel 会计算并写出辅助 `rstd`，且 `ROWS_PER_BLOCK` 最多为 4；对于当前纯推理 epilogue，这些通用能力不是必需的。
- 归一化维度只有 128，算术量小，host 侧 guard/reshape、CTA 数量和辅助访存的占比更突出。

#### 2. 初赛实现：用固定 shape Triton kernel 替换通用 fused kernel

```text
result = RMSNorm(x) × SiLU(z)
       = x × rsqrt(mean(x²) + eps) × weight × z × sigmoid(z)
```

- 一个 Triton program 同时处理 16 个 head-row，每行固定 128 维；相比通用路径最多 4 row/CTA，减少 program 数量。
- `x`、`z`、`weight` 转为 FP32 完成平方和、归一化和门控，最后写回 BF16。
- 不创建/写回推理后续不需要的 `rstd` buffer，也不保留 bias、group 或 activation 分支。
- 只替换 GDN Part 3 epilogue；输入投影、递推状态和 `gdn_attention_core` 均保持官方逻辑。
- 非 gfx936 或 shape/layout 不匹配时，回退到官方 `norm(...)`。

#### 3. 性能来源

| 原因 | 性能来源 |
|---|---|
| 48 个 GDN 层重复执行 | 小优化可在全模型中高频累计 |
| 128 维 reduction 很小 | 降低 host guard、reshape 和通用调度的相对开销 |
| 通用 kernel 最多 4 row/CTA | 专用 kernel 处理 16 row/CTA，减少 CTA 数量 |
| 通用路径写回辅助 `rstd` | 专用推理 kernel 省去无后续消费者的辅助写回 |
| shape 固定 | 编译期确定列宽、program 大小和 reduction 方式，去掉无关分支 |

### 讲解话术

> GDN 部分我们没有改最复杂的递推核心，而是抓住了 48 层都会执行的输出 epilogue。需要说明的是，官方已经把 RMSNorm 和 Gate 放在一个通用 Triton kernel 中；我们的增量不是再少一个 kernel，而是利用固定的 128 维和 48 head 去掉通用分支与 `rstd` 辅助写回，并把每个 CTA 处理的 row 从最多 4 个提高到 16 个。它的改动边界很窄、调用频率很高，也不影响 GDN recurrent state。

### 代码证据

- `813/vllm/model_executor/layers/fla/ops/gfx936.py`：`_gdn_rmsnorm_silu_gate`、`qwen35_gdn_rmsnorm`。
- `813/vllm/model_executor/models/qwen3_5.py`：在 `Qwen3_5GatedDeltaNet.forward` 的 Part 3 接入融合函数。

> 数据口径提醒：当前仓库没有独立的 GDN 端到端消融吞吐数据；本页描述的是代码机制与性能来源，最终系统收益见第 6 页。

---

## 第 2 页：GDN 决赛优化——从固定 48 Head 到 TP=1/2、动态 Token 通用

### 页面标题

**GDN 决赛演进：参数化 Head/Stride，让融合在 TP=2 与多 Batch 下继续命中**

### 页面主结论

TP=2 后每卡 GDN value head 从 48 变为 24；多 Batch 下本轮 token 总数 `T` 也会动态变化。决赛将初赛 kernel 中与 48 head 绑定的地址计算改为运行时参数，并为尾部 row 增加 mask，使同一融合 kernel 同时支持 TP=1、TP=2 和动态 token 数。

### 页面建议布局

上半部分放 shape 演进，下半部分放地址计算变化和性能来源。

### 页面可见内容

#### 1. 场景变化

| 场景 | 本卡 GDN 输出 shape | 初赛硬编码的问题 |
|---|---|---|
| TP=1 | `[T, 48, 128]` | 固定除以 48、固定 gate token stride=16384 可以工作 |
| TP=2 | `[T, 24, 128]` | token/head 映射和 gate offset 全部变化 |
| 多 Batch | `T = 本轮所有调度 token 的扁平化总数` | `T×head` 不一定整除原 grid，需要尾部保护 |

#### 2. 决赛代码改造

```text
初赛：token = row // 48
      head  = row % 48
      gate token stride = 16384

决赛：token = row // num_heads
      head  = row % num_heads
      gate token stride = gate.stride(0)
      valid = row < num_rows
```

- 支持 `x.shape[1] in {24, 48}`，即 TP=2 和 TP=1 两种本地 head 数。
- 运行时传入 `num_rows`、`num_heads`、`gate_token_stride`。
- grid 改为 `ceil(num_rows / 16)`，并对最后一个 program 使用 `valid_rows` mask。
- `gate.stride()[1:] == (128, 1)` 时可处理不同 token stride，不再绑定单一布局。

#### 3. 为什么适合多 Batch、TP=2

- vLLM 将本轮多个请求的 token 扁平化为 `T`；该 kernel 只依赖总 row 数和实际 stride，不依赖请求数量。
- TP 只改变本卡 head 数，不改变每个 head 的 128 维 RMSNorm 数学；因此本地计算可以直接参数化复用。
- 融合发生在每卡的 GDN local tensor 上，不引入额外跨卡通信；后续 `out_proj` 和 TP collective 仍由官方线性层负责。

#### 4. 决赛性能来源

```text
TP=2：每卡 head 48 → 24，单卡本地工作量下降
   +  保留固定 128 维、16 row/CTA 的定形 kernel，避免退回通用 fused kernel
   +  动态 T/尾部 mask，使多 Batch 调度下仍可稳定命中
```

### 讲解话术

> 初赛版本最快，但它把 48 个 head 和 16384 的 token stride 写死了。TP=2 后每卡只有 24 个 head，如果只改 shape gate，地址就会读错。决赛真正的改造是把 token/head 映射、token stride 和总 row 数全部参数化，并增加尾部 mask。这样无论这一轮调度了多少 token，同一个 kernel 都能处理；TP 通信则继续由官方 out projection 管理，我们没有改变并行语义。

### 代码证据

- `813/vllm/model_executor/layers/fla/ops/gfx936.py`：`num_rows`、`num_heads`、`gate_token_stride`、`valid_rows`。
- `813` 的决赛提交 `7b469aa`：该文件由固定 `[T,48,128]` 扩展为 `[T,24/48,128]`。

---

## 第 3 页：GQA Attention 初赛优化——利用 GQA6 复用与 page784 分解

### 页面标题

**GQA Attention：针对 Q24/KV4/D256 与 784-token Hybrid Page 定形优化**

### 页面主结论

全量注意力的两个结构特征可以被利用：一是 6 个 Query Head 共享 1 个 KV Head；二是 Hybrid KV Cache 的 page size 为 784，不适合直接套用规则的 FlashAttention tile。初赛分别用 GQA6 kernel 和 page784 分解路径解决。

### 页面建议布局

左侧画 GQA6 的 KV 复用，右侧画 `784 = 768 + 16` 的 page 分解。

### 页面可见内容

#### 1. GQA6：一个 KV Head 服务 6 个 Query Head

```text
Q0 ─┐
Q1 ─┤  同一 CTA，共享一次 K/V tile 读取
    ├── KV0
Q2 ─┤  其余 Q head 按“两两成对”计算
... ┘
```

- TP=1 目标 shape：`Q24 / KV4 / head_dim=256`，固定 GQA ratio 为 6。
- 将同一 KV head 对应的 6 个 Q head 分为 3 对；一个 CTA 同时计算两个 Q head。
- grid 第二维为 `4 KV heads × 3 head pairs = 12`。
- 保留 FP32 online softmax：`max_score → correction → denominator → P@V`。
- 短 query 使用 `BLOCK_ROWS=16`；长 query 使用 `BLOCK_ROWS=64`，并拆为两个 32-token KV subtile。

**性能来源：** 两个 Q head 复用同一 KV tile，减少 K/V 重复加载；固定 Q/KV 比例后，减少通用分支和索引开销。

#### 2. page784：把不规则 page 转化为规则主干

```text
一个 784-token page
┌──────────────────── 768 ────────────────────┬─ 16 ─┐
│ main：规则 paged FlashAttention，non-causal │ tail │
└──────────────────────────────────────────────┴──────┘

residual = 每页 16-token tail + boundary + current K/V
output   = LSE-weighted merge(main, residual)
```

- `main`：每个完整 page 的前 768 token，直接调用高性能官方 FlashAttention。
- `residual`：每页 16-token 尾部、边界 token 和当前 K/V，由 Triton kernel 重新打包为规则 page64。
- 分别计算 main/residual attention，并使用两组 FP32 LSE 稳定合并，不能直接平均。
- workspace 和 metadata 缓存复用，避免 16 个 full-attention 层重复申请大 tensor。

**性能来源：** 让长上下文的大部分 token 进入成熟的 FlashAttention 主路径，只为少量不规则尾部支付 pack 与 merge 成本。

### 讲解话术

> GQA Attention 的优化来自两个结构事实。第一，模型是 Q24/KV4，也就是 6 个 Q head 共用一个 KV head，所以我们让一个 CTA 同时算两个 Q head，复用 K/V tile。第二，Hybrid 模型统一后的 KV page 是 784，而高性能 Attention 更喜欢规则块。我们把每页拆成 768 个主干 token 和 16 个尾部 token，主干走官方 FlashAttention，尾部与当前 KV 打包成 residual，最后按 LSE 权重合并。这相当于把不规则问题隔离到很小的一部分数据上。

### 代码证据

- `813/vllm/v1/attention/ops/rocm_aiter_unified_attention_gqa6.py`：GQA6 Triton kernel。
- `813/vllm/v1/attention/ops/rocm_aiter_unified_attention_page784.py`：pack、两次 FlashAttention、LSE merge。
- `813/vllm/v1/attention/backends/rocm_aiter_unified_attn.py`：`page784 → GQA6 → official AITER` 路由。

> 数据口径提醒：page784 与 GQA6 是同一 Attention 入口下的互斥 fast path，不是串行叠加的两个算子。

---

## 第 4 页：GQA Attention 决赛优化——TP=2 参数化与多 Batch 安全分流

### 页面标题

**GQA 决赛演进：Q12/KV2 复用同一 GQA6 结构，多 Batch 按收益边界分流**

### 页面主结论

TP=2 后每卡从 Q24/KV4 变为 Q12/KV2，但 GQA ratio 仍然是 6。决赛保留“两个 Q head 共享 KV tile”的计算结构，将 head 数、stride、workspace 和 merge grid 全部参数化；对于真正的多序列混合 Batch，则安全回退到官方 AITER。

### 页面建议布局

上半部分用表格对比 TP=1/TP=2，下半部分画运行时路由图。

### 页面可见内容

#### 1. TP=2：变化的是本地 shape，不变的是 GQA ratio

| 项目 | TP=1 | TP=2（每卡） | 决赛改造 |
|---|---:|---:|---|
| Query heads | 24 | 12 | 从 `query.shape[1]` 读取 |
| KV heads | 4 | 2 | 从 `key_cache.shape[2]` 读取 |
| Q heads / KV head | 6 | 6 | 继续两两成对复用 KV tile |
| Query token stride | 6144 | 3072 | 使用 `query.stride()`，不再硬编码 |
| merge rows | `T×24` | `T×12` | `ceil(T×num_query_heads/4)` |

#### 2. GQA6 与 page784 的参数化

- GQA6 kernel 新增 `QUERY_STRIDES`、`OUTPUT_STRIDES`、`Q_HEADS_PER_KV`。
- grid 第二维由固定 12 改为 `num_query_heads / 2`：TP=1 为 12，TP=2 为 6。
- page784 的 token workspace 从固定 `[4096,24,256]` 改为按本地 Query head 数创建。
- packed page 从固定 4 个 KV head 改为 `num_kv_heads`；workspace cache key 同时包含 Q/KV head 数。
- backend 的目标 cache 从固定 `(784,4,256)` 改为 `(784,self.num_kv_heads,self.head_size)`。

#### 3. 保留 784-token Hybrid Page

```text
原通用 ROCm AITER 逻辑：可能把 block size 覆盖为 64
决赛逻辑：模型已选择 block_size=784 时不覆盖
```

- Qwen3.5 的 GDN state page 与 Attention KV page 需要统一物理页大小。
- `784` 不能被 `64` 整除，强制覆盖会破坏 Hybrid KV page 统一。
- 决赛仅当原 block size `<=64` 时使用 64；否则保留模型选择的 784。

#### 4. 多 Batch 的真实处理边界

```text
Qwen3.5 + gfx936 + BF16 + page784 + 单序列 prefill？
        ├─ 是：page784（长上下文收益区间）
        │       └─ 不命中：GQA6
        └─ 否：官方 AITER unified_attention
```

- 专用 Attention 路径要求 `query_start_loc.numel()==2`，即本轮只有一个 prefill 序列。
- 决赛长 prompt 且 token budget 为 4096，一个 chunked-prefill 请求通常即可占满本轮预算，专用路径仍能覆盖主要 prefill 热点。
- 多序列 prefill、混合 Batch、decode、FP8、ALiBi、window、sink 等场景继续走官方 AITER，避免把单序列 kernel 错用到动态 Batch。

#### 5. 性能来源

- TP=2 后每卡 Q/KV head 数减半，专用 kernel 的本地并行规模同步缩小。
- GQA ratio=6 不变，因此 KV tile 复用策略无需重新设计。
- 长上下文越长，768-token main 占比越大，page784 的 pack/merge 固定开销越容易被摊薄。
- 不强行优化多序列 Batch：高收益单序列 prefill 用定形 kernel，复杂 Batch 用成熟 AITER。

### 讲解话术

> TP=2 并没有破坏 GQA6 的核心假设，因为 Q head 和 KV head 都除以 2，比例仍然是 6。我们做的是彻底消除 Q24、KV4、6144 stride 等硬编码，让 kernel 根据本地 tensor shape 启动。多 Batch 方面我们设置了明确边界：当前专用 Attention 只处理单序列 prefill；在本评测的长 prompt 和 4096 token budget 下，一个请求就能占满一个 prefill step，因此仍可命中主要热点。真正的多序列或 decode 则交给官方 AITER，这个分流既保证正确性，也避免专用 kernel 在不适合的 shape 上负优化。

### 代码证据

- `813/vllm/v1/attention/ops/rocm_aiter_unified_attention_gqa6.py`：动态 Q/KV head 与 stride。
- `813/vllm/v1/attention/ops/rocm_aiter_unified_attention_page784.py`：动态 workspace、pack stride 和 merge grid。
- `813/vllm/v1/attention/backends/rocm_aiter_unified_attn.py`：同时接受 `(Q12,KV2)` 与 `(Q24,KV4)`。
- `813/vllm/platforms/rocm.py`：保留模型选择的 784 block size。

---

## 第 5 页：MLP GEMV 初赛优化——固定 K=5120 的 Decode 专用 Kernel

### 页面标题

**MLP GEMV：为单 Token Decode 重构 K=5120 的访存与规约，并融合 SwiGLU**

### 页面主结论

单 Batch decode 时矩阵乘退化为 `N=1` 的 GEMV。通用 GEMM 很难在这种 skinny shape 下充分利用硬件，因此初赛为固定输入维度 K=5120 编写 HIP kernel，并将 MLP 的 GateUp 与 SwiGLU 融合。

### 页面建议布局

左侧展示 HIP kernel 的线程映射，右侧展示 GateUp/SwiGLU 融合前后。

### 页面可见内容

#### 1. 为什么要从 GEMM 切换到专用 GEMV

```text
Decode：x = [1, 5120]
Linear：W[M,5120] × x[5120]
```

- `N=1` 时缺少可复用的 activation 行，通用 GEMM 的 tile 利用率低。
- 每个输出元素要读取 5120 个 BF16 权重，算术强度低，主要受显存带宽和 reduction 效率限制。
- MLP 在 64 层中均出现，单 token decode 每层重复执行，优化可累积到每个输出 token。

#### 2. 固定 K=5120 的 HIP kernel

```text
5120 / 8 = 640 threads
每个 thread：一次 128-bit float4 load → 8 个 BF16 乘加
320 对 lane 合并 → 5 个 wave leader → FP32 二级规约 → BF16 输出
```

- 固定 640 threads，一次覆盖整行 K=5120 权重。
- 使用 128-bit 权重加载，一个 thread 完成 8 个 BF16 product。
- 后 320 个 lane 写 LDS，前 320 个 lane 合并；再由 5 个 wave leader 完成第二级规约。
- 大多数 shape 每 CTA 计算 2 个输出 row；小输出 `M=96` 每 CTA 计算 4 row，提高并行度。

#### 3. MLP GateUp + SwiGLU 融合

```text
优化前：GEMV[34816] → 写 gate+up → SiluAndMul kernel → 写 [17408]

优化后：
Pass 1：计算 gate[17408]，暂存在最终 output buffer
Pass 2：计算 up[17408]，同时完成 SiLU(gate) × up，原位写回
```

- 避免生成完整的 34816 维 GateUp 中间张量。
- 消除独立 `SiluAndMul` kernel launch。
- 减少中间结果的显存写入和再次读取。
- 之后仍调用官方 `down_proj`，不改变 MLP 数学和 TP 语义。

#### 4. 同一 K5120 kernel 的覆盖范围

| 投影 | TP=1 输出 M | 层数/频率 |
|---|---:|---|
| Full Attention QKV | 14336 | 16 层 |
| GDN QKVZ | 16384 | 48 层 |
| GDN BA | 96 | 48 层 |
| MLP GateUp | 34816 | 64 层 |
| LM Head | 248320 | 每个需要 logits 的 step |

> 本页主题是 MLP GEMV；QKV、GDN 投影和 LM Head 复用了同一固定 K=5120 基础 kernel。

#### 5. 性能来源

- 针对 `N=1` 消除通用 GEMM 的无效 tile 与调度开销。
- 128-bit 连续加载提高权重访存效率。
- 两级规约适配 gfx936 wave64。
- GateUp/SwiGLU 融合减少一个 kernel 和大尺寸中间张量流量。
- 64 层 MLP 每个 decode token 都会执行，收益高频累积。

### 讲解话术

> Decode 阶段的关键是 N=1，此时矩阵乘实际上是 GEMV，通用 GEMM 的优势发挥不出来。因为模型 hidden size 固定为 5120，我们让 640 个线程每个读取 8 个 BF16 权重，一次覆盖整行 K 维，并用 wave64 友好的两级规约求和。MLP 中最大的进一步收益来自 GateUp 和 SwiGLU 融合：不再先写 34816 维中间结果再启动激活 kernel，而是第二遍 GEMV 直接读取 gate、计算 SiLU 并与 up 相乘。

### 代码证据

- `813/csrc/rocm/skinny_gemms.cu`：`qwen35_gemv_k5120<ROWS,FUSE_SILU>`。
- `813/vllm/model_executor/models/qwen3_next.py`：`Qwen3NextMLP.forward` 接入 fused GateUp fast path。
- `813/vllm/model_executor/layers/utils.py`：普通无 bias 线性层接入 K5120 fast path。
- `813/vllm/model_executor/layers/fla/ops/gfx936.py`：Python shape/device/dtype gate。

---

## 第 6 页：MLP GEMV 决赛优化——TP=2 Local Shape + Batch-aware Dispatch + 最终收益

### 页面标题

**GEMV 决赛演进：扩展 TP=2 分片 shape，按 Batch 大小选择 GEMV 或 GEMM**

### 页面主结论

TP=2 将每张卡的 column-parallel 输出维度减半；决赛为所有本地 shape 增加 kernel 路由。多 Batch 时不继续使用 N=1 GEMV，而是回退到官方 GEMM，让更大的 `N` 提升矩阵乘的算术强度。这种按 Batch 分流是决赛稳定提速的关键。

### 页面建议布局

上半部分放 TP=1/TP=2 shape 表和分流图，下半部分放系统实测结果。

### 页面可见内容

#### 1. TP=2 本地输出 shape 扩展

| 投影 | TP=1 M | TP=2 每卡 M | 决赛处理 |
|---|---:|---:|---|
| GDN BA | 96 | 48 | 小 M 使用 4 rows/CTA |
| Full Attention QKV | 14336 | 7168 | 使用 2 rows/CTA |
| GDN QKVZ | 16384 | 8192 | 使用 2 rows/CTA |
| MLP GateUp | 34816 | 17408 | 两遍 GEMV + fused SwiGLU，输出 8704 |
| LM Head | 248320 | 124160 | 对本地词表分片做 GEMV |

- Python gate 和 HIP `LLMM1` 同时保留 TP=1、TP=2 两组 M，便于复用和 fallback。
- fused SwiGLU 从只接受 `M=34816` 扩展为 `M in {17408,34816}`。
- kernel 的 K 仍为 5120，线程映射和规约结构不变，只调整输出 row 数和 grid。

#### 2. 多 Batch：不是“批量 GEMV”，而是 GEMV/GEMM 自适应分流

```text
x.numel() == 5120，也就是 N=1？
        ├─ 是：固定 K5120 专用 GEMV；MLP 可融合 SwiGLU
        └─ 否：官方 ROCm GEMM / AITER / TunableOp 路径
```

- 专用 helper 明确要求 `x.numel()==5120`，因此只处理单 token decode。
- 并发请求形成 `N>1` decode batch 后，官方 GEMM 可以复用 activation tile，算术强度明显高于 GEMV。
- 若强行逐 token 启动 GEMV，会产生更多 kernel launch、重复读权重并降低 GPU 利用率。
- 决赛保留官方 fallback，使运行时自动覆盖 Batch=1、Batch>1 以及动态 Batch 收缩过程。

#### 3. TP 通信边界

- GateUp、QKV 和 LM Head 先在每卡本地分片上计算，专用 kernel 不增加跨卡数据。
- MLP `down_proj` 及其后续 collective 仍由官方 Tensor Parallel 线性层负责。
- 本版本优化计算 kernel，没有宣称消除或融合 All-Reduce；系统收益来自本地计算、Attention 与正确分流的共同作用。

#### 4. TP=2、最大并发 10 的系统级实测

测试共同配置：BF16、TP=2、DP=1、最大并发 10、`max_num_batched_tokens=4096`、最大输出 1024。

| 数据集 | Baseline 总吞吐 | 813 总吞吐 | 提升 | Baseline P95 TTFT | 813 P95 TTFT | 降低 |
|---|---:|---:|---:|---:|---:|---:|
| 4–8K | 1567.93 tok/s | 1928.52 tok/s | **+23.0%** | 15.74 s | 8.92 s | **-43.3%** |
| 8–16K | 1706.42 tok/s | 2438.31 tok/s | **+42.9%** | 45.15 s | 27.39 s | **-39.3%** |
| 16–32K | 1180.73 tok/s | 2126.64 tok/s | **+80.1%** | 140.20 s | 67.98 s | **-51.5%** |

补充结果：

- 210/210 条吞吐请求成功，无失败、OOM、HTTP 5xx 或 engine crash。
- 95 条精度预测完整；HotpotQA F1 `67.8414`、检索准确率 `100%`、关键词聚合 `75%`，与基线一致；GovReport ROUGE-L F1 为 `33.2311`，基线为 `32.8629`。

#### 5. 如何解释整体性能趋势

| 场景 | 主要性能来源 |
|---|---|
| Prefill | GDN Norm+Gate 定形 kernel；GQA6 的 KV 复用；page784 将大部分长上下文送入规则 FlashAttention |
| 单 token decode | 固定 K5120 GEMV；GateUp+SwiGLU 融合；TP=2 本地 M 减半 |
| 多 token decode | 回退官方 GEMM，通过更大的 N 获得更高算术强度 |
| 长上下文 | Attention 占比上升，page784/GQA 优化被更充分放大，因此 16–32K 提升最大 |

> 上表是三个类别共同作用后的**系统级结果**，不能把 +23.0%～+80.1% 全部归因于 GEMV，也不能当作某一个 kernel 的独立加速比。

### 讲解话术

> TP=2 后每卡的 GateUp、QKV、GDN 投影和词表都减半，因此我们把对应的本地 M shape 全部加入 K5120 kernel。更关键的是多 Batch 分流：这个专用 kernel 只在 N=1 时进入；并发产生 N>1 后，官方 GEMM 反而更合适，所以我们主动 fallback。最终 TP=2、并发 10 的全量测试中，总 token 吞吐提升 23% 到 80%，而且上下文越长提升越明显，说明主要增量不仅来自 decode GEMV，也来自 page784 和 GQA prefill。所有 210 条吞吐请求成功，95 条精度预测完整，证明优化没有用正确性换速度。

### 代码与数据证据

- `813/csrc/rocm/skinny_gemms.cu`：TP=2 M 集合、`M=17408` fused SwiGLU。
- `813/vllm/model_executor/layers/fla/ops/gfx936.py`：TP=1/2 shape 集合与 `x.numel()==5120` 的 Batch=1 gate。
- `testdata-offical/results_baseline_tp2_batch10/SUMMARY.md`：官方基线全量吞吐。
- `testdata/results_optimized_tp2_full/summary.md`、`summary.json`：813 全量吞吐。
- `testdata-offical/accuracy_baseline_full_20260813/SUMMARY.md`、`testdata-single2/accuracy_single2_full_20260813/SUMMARY.md`：精度对比。

---

## 答辩收束句（不单独占页）

> 我们的优化主线不是简单堆叠自定义 kernel，而是把 Qwen3.5 的固定结构转化为可验证的 fast path：GDN 用专用 Norm+Gate kernel 减少通用 fused kernel 的辅助开销，GQA Attention 通过 KV 复用和 page784 分解提升长上下文 prefill，MLP 通过 K5120 GEMV 与 SwiGLU 融合优化单 token decode。进入决赛后，再把固定 TP=1 shape 参数化为 TP=2 本地 shape，并依据实际 Batch 大小在专用 kernel 与官方高性能实现之间分流。最终性能来自“定形特化 + TP 本地化 + Batch-aware fallback”三者共同作用。

## 答辩时需要主动说明的边界

1. Attention 专用路径当前只接管单序列 prefill；多序列/混合 Batch 走官方 AITER。
2. K5120 GEMV 当前只接管 `N=1`；`N>1` 走官方 GEMM，这不是覆盖缺失，而是有意的性能选择。
3. page784 和 GQA6 是互斥路由，不是两个串行算子。
4. GDN 优化只融合输出 epilogue，没有改写 GDN recurrent core。
5. TP All-Reduce 没有被该版本融合；TP=2 收益主要来自本地 shape 减半、专用 kernel 适配以及 Attention 路径优化。
6. 第 6 页吞吐数字是三类优化共同作用的端到端结果，不是单算子消融数据。
