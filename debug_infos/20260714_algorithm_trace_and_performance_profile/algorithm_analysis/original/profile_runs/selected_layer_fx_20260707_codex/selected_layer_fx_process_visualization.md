# Selected-Layer FX Process Visualization

本文档基于 9 个 selected-layer FX trace 的 `fx_process_reconstruction.md/json` 和 `fx_process_nodes.csv` 手工解释 layer 内部 process。三组上下文、三个筛选 layer 的 FX DAG 都是 155 个节点、12 个 process；除 `8-16K/input4_layer31` 的 `S=1685` 外，其余事件的 `S=4096`。这里的 process label 是重建规则，不是 FX 或 vLLM 官方模块归属。

## 覆盖的 FX Event

| Context | Event | Layer | q_len `S` | Reconstruction |
| --- | --- | ---: | ---: | --- |
| `4-8K` | `input1_layer3` | 3 | 4096 | `contexts/4-8K/fx_trace/traces/input1_layer3/fx_process_reconstruction.md` |
| `4-8K` | `input1_layer31` | 31 | 4096 | `contexts/4-8K/fx_trace/traces/input1_layer31/fx_process_reconstruction.md` |
| `4-8K` | `input1_layer59` | 59 | 4096 | `contexts/4-8K/fx_trace/traces/input1_layer59/fx_process_reconstruction.md` |
| `8-16K` | `input1_layer3` | 3 | 4096 | `contexts/8-16K/fx_trace/traces/input1_layer3/fx_process_reconstruction.md` |
| `8-16K` | `input3_layer59` | 59 | 4096 | `contexts/8-16K/fx_trace/traces/input3_layer59/fx_process_reconstruction.md` |
| `8-16K` | `input4_layer31` | 31 | 1685 | `contexts/8-16K/fx_trace/traces/input4_layer31/fx_process_reconstruction.md` |
| `16-32K` | `input1_layer3` | 3 | 4096 | `contexts/16-32K/fx_trace/traces/input1_layer3/fx_process_reconstruction.md` |
| `16-32K` | `input2_layer31` | 31 | 4096 | `contexts/16-32K/fx_trace/traces/input2_layer31/fx_process_reconstruction.md` |
| `16-32K` | `input4_layer59` | 59 | 4096 | `contexts/16-32K/fx_trace/traces/input4_layer59/fx_process_reconstruction.md` |

共同轴和宽度：

- `S`: 当前 prefill chunk token 数，`4096` 或 `1685`。
- `Hidden`: `5120`。
- Q heads: `24`，KV heads: `4`，head_dim: `256`。
- RoPE rotary band: `64 = 32 + 32`，pass-through band: `192`。
- MLP intermediate: `17408`，fused gate/up projection width: `34816`。

## 1. Runtime FX Inputs

是什么：这是固定输入 FX DAG 的入口，节点 `#000-#002` 分别给出 `positions [3,S]`、`hidden_states [S,5120]` 和 `residual [S,5120]`。

为什么需要：后续所有 layer 内计算都依赖这三个 runtime 采样值；`positions` 驱动 MROPE lookup，`hidden_states` 和 `residual` 先相加形成本 layer 的 pre-attention residual。

怎么做/计算：placeholder 节点不做数值计算，只把 sampled tensor 暴露给后续 ATen 节点。`arg0_1` 被 `aten.index.Tensor` 用来查 rotary table；`arg1_1` 和 `arg2_1` 被 `aten.add.Tensor` 合成 `add [S,5120]`。

```text
Token axis S (0..S-1)                         Hidden dimension (0..5119)
                                              ▲                              ▲
positions arg0_1 [3,S]                 ──▶   [POS_AXIS_0 | POS_AXIS_1 | POS_AXIS_2]  ◀── MROPE position ids

hidden_states arg1_1 [S,5120]          ──▶   +--------------------------------+
                                             | HIDDEN_STATE_ROWS              |
                                             | HIDDEN_STATE_ROWS              |
                                             | HIDDEN_STATE_ROWS              |
                                             +--------------------------------+
residual arg2_1 [S,5120]               ──▶   +--------------------------------+
                                             | RESIDUAL_ROWS                  |
                                             | RESIDUAL_ROWS                  |
                                             | RESIDUAL_ROWS                  |
                                             +--------------------------------+
examples: token row 0, token row S-1 share the same Hidden axis 0..5119.
```

## 2. Pre-Attention Residual Add And Input RMSNorm

是什么：节点 `#003-#015` 把 `hidden_states + residual` 变成 RMS-normalized attention input `[S,5120]`。

为什么需要：Qwen3.5 decoder layer 在 attention 前使用 residual stream，再通过 RMSNorm 稳定每个 token row 的 Hidden 维尺度。

怎么做/计算：`#005 aten.add.Tensor` 先得到 `add [S,5120]`。`#006 aten._to_copy` 转 fp32；`#007 pow` 对 Hidden 元素平方；`#008 mean.dim([-1], keepdim=True)` 将每行 Hidden 归约成 `[S,1]` RMS 能量；`#009 add` 加 `1e-6`；`#010 rsqrt` 得到 inverse RMS；`#011 mul` 对每行广播缩放。`#012-#014` 把 norm weight 转 fp32 后加 1，再乘到每个 Hidden 位置；`#015` 转回 bf16。

```text
Token axis S (compressed)                     Hidden dimension
                                              0                            5119
                                              ▲                              ▲
residual stream add [S,5120]           ──▶   +--------------------------------+
                                             | TOKEN_ROW_0: h+r over Hidden   |
                                             | TOKEN_ROW_i: h+r over Hidden   |
                                             | TOKEN_ROW_S-1: h+r over Hidden |
                                             +--------------------------------+

RMS per token [S,1]                    ──▶   +----+
                                             |RMS |
                                             |RMS |
                                             |RMS |
                                             +----+  ◀── mean(square(row)) + eps, then rsqrt

normalized attention input [S,5120]    ──▶   +--------------------------------+
                                             | NORMED_TOKEN_ROW_0             |
                                             | NORMED_TOKEN_ROW_i             |
                                             | NORMED_TOKEN_ROW_S-1           |
                                             +--------------------------------+
formula: normed[s,h] = (hidden[s,h] + residual[s,h]) * rsqrt(mean_h(square(row)) + eps) * weight[h]
```

## 3. Fused Q/Gate/K/V Projection And Head Reshape

是什么：节点 `#017-#031` 对 normalized hidden 做一次大矩阵乘，得到 fused projection `[S,14336]`，再拆成 Q/gate/K/V。

为什么需要：attention 需要 Q/K/V；当前源码编译 vLLM 的 FX 图还显示了一个与 attention output 相乘的 gate branch，它和 Q branch 同在第一段 `[S,12288]` 中。

怎么做/计算：`#018 t` 把 projection weight 转成 `[5120,14336]`；`#019 mm` 计算 `[S,5120] x [5120,14336] -> [S,14336]`。`#020 split_with_sizes([12288,1024,1024])` 划出 Q+gate、K、V。`#024 view` 把第一段变成 `[S,24,512]`，`#025 split(256,-1)` 切成 Q half 和 gate half；`#028-#031 clone/_unsafe_view` 分别得到 flattened Q `[S,6144]` 和 gate `[S,6144]`。

```text
Token axis S                                    Feature / projection width
                                                0       6143 6144 12287 12288 13311 13312 14335
                                                ▲         ▲    ▲    ▲     ▲     ▲     ▲     ▲
normalized hidden [S,5120]               ──▶   +------------------------------------------------+
                                               | NORMED_HIDDEN_ROWS                              |
                                               +------------------------------------------------+

fused projection [S,14336]               ──▶   +----------------+----------------+------+------+
                                               | Q_BRANCH 6144  | GATE 6144      | K1024| V1024|
                                               +----------------+----------------+------+------+

Q head view [S,24,256]                   ──▶   +--------------------------------+
                                               | HEAD_0 ... HEAD_23, Dh=256     |
                                               +--------------------------------+
gate flattened [S,6144]                  ──▶   +--------------------------------+
                                               | ATTENTION_OUTPUT_GATE_ROWS      |
                                               +--------------------------------+
```

## 4. Q Head RMSNorm

是什么：节点 `#032-#045` 对 Q tensor `[S,24,256]` 的每个 token/head 向量做 RMSNorm，并回到 flattened Q `[S,6144]`。

为什么需要：Qwen3.5 的 attention 在 RoPE 前对 Q/K head vector 单独归一化，避免不同 head_dim 上尺度漂移。

怎么做/计算：`#032 view` 得到 `[S,24,256]`。`#035-#040` 和输入 RMSNorm 同构：fp32、平方、对最后一维 256 求 mean、加 eps、rsqrt、广播乘回 Q。`#041-#043` 准备 head norm weight，`#044-#045` 转回 bf16 并 flatten 成 `[S,6144]`。

```text
Token/head axis (S x 24)                       Head dimension Dh
                                               0                            255
                                               ▲                              ▲
Q heads before norm [S,24,256]          ──▶   +--------------------------------+
                                              | Q_HEAD_VECTOR                 |
                                              | Q_HEAD_VECTOR                 |
                                              | Q_HEAD_VECTOR                 |
                                              +--------------------------------+
Q RMS [S,24,1]                          ──▶   +----+
                                              |RMS |
                                              |RMS |
                                              |RMS |
                                              +----+
Q normalized [S,24,256]                 ──▶   +--------------------------------+
                                              | Q_NORMED_HEAD_VECTOR          |
                                              | Q_NORMED_HEAD_VECTOR          |
                                              | Q_NORMED_HEAD_VECTOR          |
                                              +--------------------------------+
```

## 5. K Head RMSNorm

是什么：节点 `#046-#059` 对 K tensor `[S,4,256]` 的每个 token/KV-head 向量做 RMSNorm，并 flatten 成 `[S,1024]`。

为什么需要：K 与 Q 使用相同的 head_dim 归一化规则，后续 RoPE 和 attention custom op 依赖归一化后的 K。

怎么做/计算：`#046 view` 将 K branch reshape 为 `[S,4,256]`；`#049-#054` 对最后一维平方、mean、加 eps、rsqrt、广播缩放；`#055-#058` 应用 K head norm weight；`#059 view` 得到 `[S,1024]`。

```text
Token/KV-head axis (S x 4)                    Head dimension Dh
                                              0                            255
                                              ▲                              ▲
K heads before norm [S,4,256]          ──▶   +--------------------------------+
                                             | K_HEAD_VECTOR                 |
                                             | K_HEAD_VECTOR                 |
                                             | K_HEAD_VECTOR                 |
                                             +--------------------------------+
K normalized [S,4,256]                ──▶   +--------------------------------+
                                             | K_NORMED_HEAD_VECTOR          |
                                             | K_NORMED_HEAD_VECTOR          |
                                             | K_NORMED_HEAD_VECTOR          |
                                             +--------------------------------+
```

## 6. MROPE Cos/Sin Table Lookup And Axis Remap

是什么：节点 `#060-#084` 用 `positions [3,S]` 查 `_tensor_constant0 [1048576,64]`，得到 cos/sin 相关表 `[3,S,64]`，再拆出两个 `[3,S,32]` 半表并重排成 per-token rotary 系数。

为什么需要：RoPE 需要按 token position 提供 cos/sin。这里 positions 有 3 个 axis，FX 图中通过 `select`、`slice`、`copy_` 把 axis 1/2 的部分列写回 axis 0 clone，形成后续 Q/K RoPE 直接使用的 `[S,32]` 系数。

怎么做/计算：`#061 aten.index.Tensor` 完成 table lookup；`#062 split(32,-1)` 得到两个 `[3,S,32]` 半表。`#065-#074` 对第一个半表执行 axis 0 clone，并把 axis 1 的列 `1:33:3` 和 axis 2 的列 `2:30:3` copy 到 clone 对应列。`#075-#084` 对第二个半表做同样操作。最终 `clone_2 [S,32]` 和 `clone_3 [S,32]` 作为 RoPE 的两组系数。

```text
Position axis A=3, token S, rotary half 32

lookup table result [3,S,64]            ──▶   +----------------+----------------+
                                             | HALF_0 [3,S,32] | HALF_1 [3,S,32] |
                                             +----------------+----------------+

axis rows inside one half [3,S,32]
Rotary half dimension                         0      1..32 step3      2..29 step3      31
                                              ▲           ▲                ▲            ▲
axis0 clone [S,32]                     ──▶   +------------------------------------------+
                                             | AXIS0_BASE + COPIED_AXIS1 + COPIED_AXIS2 |
                                             +------------------------------------------+
axis1 selected columns                 ──▶            [AXIS1_COLUMNS]
axis2 selected columns                 ──▶                             [AXIS2_COLUMNS]

examples: slice #068/#069 has shape [S,11]; slice #072/#073 has shape [S,10].
```

## 7. Q RoPE Application

是什么：节点 `#085-#101` 对 Q 的前 64 个 head_dim 位置应用 rotary transform，剩余 192 维保持 pass-through，然后恢复 Q flattened `[S,6144]`。

为什么需要：RoPE 将位置信息注入 Q/K 的 rotary band；vLLM attention custom op 接收的是已经 rotary 后的 Q/K。

怎么做/计算：`#085 view` 得到 Q `[S,24,256]`。`#086 slice` 取 rotary band `[S,24,64]`，`#087 slice` 保留 `[S,24,192]`。`#090 split(32,-1)` 拆成 left/right 半维；`#088/#089 unsqueeze` 将 `[S,32]` 系数广播为 `[S,1,32]`。`#093-#099` 计算 `left*cos - right*sin` 与 `right*cos + left*sin`，`#100 cat` 拼回 `[S,24,256]`，`#101 view` flatten 到 `[S,6144]`。

```text
Q tensor [S,24,256]                         Head dimension
                                             0        31 32       63 64                         255
                                             ▲         ▲ ▲         ▲ ▲                            ▲
Q rotary split                         ──▶  +----------+-----------+-----------------------------+
                                            | Q_LEFT32 | Q_RIGHT32 | Q_PASS_THROUGH_192          |
                                            +----------+-----------+-----------------------------+

rotated Q band [S,24,64]              ──▶  +----------------------+-----------------------------+
                                            | LEFT*cos - RIGHT*sin | RIGHT*cos + LEFT*sin         |
                                            +----------------------+-----------------------------+

Q after RoPE [S,24,256]               ──▶  +----------------------------------------------------+
                                            | ROTATED_Q_64 | Q_PASS_THROUGH_192                 |
                                            +----------------------------------------------------+
```

## 8. K RoPE Application

是什么：节点 `#102-#118` 对 K `[S,4,256]` 执行与 Q 相同的 rotary transform，然后 flatten 为 `[S,1024]`。

为什么需要：attention score 由 Q 和 K 共同决定，两者必须在同一 rotary position 坐标系中。

怎么做/计算：`#102 view` 得到 K `[S,4,256]`；`#103/#104` 分出 rotary/pass-through；`#105/#106` 准备 `[S,1,32]` cos/sin；`#107-#117` 执行 left/right 旋转和 concat；`#118 view` 回到 `[S,1024]`。

```text
K tensor [S,4,256]                          Head dimension
                                             0        31 32       63 64                         255
                                             ▲         ▲ ▲         ▲ ▲                            ▲
K rotary split                         ──▶  +----------+-----------+-----------------------------+
                                            | K_LEFT32 | K_RIGHT32 | K_PASS_THROUGH_192          |
                                            +----------+-----------+-----------------------------+

K after RoPE [S,4,256]                ──▶  +----------------------------------------------------+
                                            | ROTATED_K_64 | K_PASS_THROUGH_192                 |
                                            +----------------------------------------------------+
```

## 9. vLLM Attention And KV Cache Update

是什么：节点 `#119-#125` 把 Q/K/V 和输出 buffer reshape 成 vLLM custom op 需要的 head layout，调用 `vllm.unified_kv_cache_update` 和 `vllm.unified_attention_with_output`。

为什么需要：FX 图没有展开 attention kernel 内部；当前证据只能看到 custom op 边界。KV cache update 先把 K/V 写入缓存相关路径，attention custom op 再把 attention result 写入 `view_10 [S,24,256]`。

怎么做/计算：`#119 empty` 分配 attention output flat buffer `[S,6144]`；`#120/#121/#122/#123 view` 分别得到 Q `[S,24,256]`、output `[S,24,256]`、K `[S,4,256]`、V `[S,4,256]`。`#124` 产生 cache update token，`#125` 使用 Q/K/V/output view 和 cache token 调用 attention。由于这是 vLLM custom op，QK score、mask、softmax、weighted V 在 FX 节点中不可见。

```text
Token axis S                                  Heads and head_dim
                                              Q heads 24 x Dh256        KV heads 4 x Dh256
                                              ▲                         ▲
Q after RoPE [S,24,256]                ──▶   +-----------------------+  |
                                             | Q_HEADS_FOR_ATTENTION |  |
                                             +-----------------------+  |
K after RoPE [S,4,256]                 ──▶                             +----------------+
                                                                        | K_CACHE_INPUT  |
                                                                        +----------------+
V branch [S,4,256]                     ──▶                             +----------------+
                                                                        | V_CACHE_INPUT  |
                                                                        +----------------+
attention output buffer [S,24,256]     ──▶   +-----------------------+
                                             | ATTENTION_OUTPUT_HEADS |
                                             +-----------------------+
custom op boundary: unified_attention_with_output writes ATTENTION_OUTPUT_HEADS in place.
```

## 10. Attention Gate, Output Projection, And Residual

是什么：节点 `#016,#126-#132,#135` 把 attention output flatten 后与 gate branch 相乘，投影回 Hidden 宽度，再加回 pre-attention residual。

为什么需要：Qwen3.5 当前 FX 路径显示 attention output 经过一个 sigmoid gate 调制，再通过输出投影回 residual stream 的 Hidden 宽度。

怎么做/计算：`#126 view` 将 attention output buffer 变成 `[S,6144]`；`#127 sigmoid` 将 gate branch `_unsafe_view_1 [S,6144]` 压到 `[0,1]`；`#128 mul` 做逐元素 gate。`#130 t_1` 把 output projection weight 转成 `[6144,5120]`，`#131 mm` 得到 `[S,5120]`，`#132 copy_` 写入早先分配的 `empty_like [S,5120]`；`#135 add` 与 pre-attention residual `add [S,5120]` 相加，得到 post-attention residual。

```text
Token axis S                                  Feature width
                                              0                         6143       5119
                                              ▲                           ▲          ▲
attention output [S,6144]              ──▶   +-----------------------------+
                                             | ATTENTION_OUTPUT_ROWS        |
                                             +-----------------------------+
gate sigmoid [S,6144]                  ──▶   +-----------------------------+
                                             | GATE_VALUES_0_TO_1           |
                                             +-----------------------------+
gated attention [S,6144]               ──▶   +-----------------------------+
                                             | ATTENTION_OUTPUT * GATE      |
                                             +-----------------------------+
projected hidden [S,5120]              ──▶   +--------------------------------+
                                             | OUTPUT_PROJECTION_ROWS          |
                                             +--------------------------------+
post-attention residual [S,5120]       ──▶   +--------------------------------+
                                             | PROJECTED_ROWS + PRE_RESIDUAL   |
                                             +--------------------------------+
```

## 11. Post-Attention RMSNorm

是什么：节点 `#133-#145` 对 post-attention residual `[S,5120]` 再做一次 RMSNorm，输出 MLP 输入 `[S,5120]`。

为什么需要：MLP 前需要规范化 residual stream，和 transformer decoder 的 post-attention norm 路径对应。

怎么做/计算：`#136 aten._to_copy` 转 fp32；`#137 pow`、`#138 mean.dim([-1])`、`#139 add(eps)`、`#140 rsqrt` 得到每 token 的 inverse RMS；`#141 mul` 缩放 residual row。`#142-#144` 应用 norm weight，`#145` 转回 bf16，供 MLP `mm_2` 使用。

```text
Token axis S (compressed)                     Hidden dimension
                                              0                            5119
                                              ▲                              ▲
post-attn residual [S,5120]            ──▶   +--------------------------------+
                                             | POST_ATTENTION_RESIDUAL_ROWS   |
                                             | POST_ATTENTION_RESIDUAL_ROWS   |
                                             +--------------------------------+
RMS per token [S,1]                    ──▶   +----+
                                             |RMS |
                                             |RMS |
                                             +----+
MLP input [S,5120]                     ──▶   +--------------------------------+
                                             | POST_NORM_MLP_INPUT_ROWS       |
                                             | POST_NORM_MLP_INPUT_ROWS       |
                                             +--------------------------------+
```

## 12. MLP And Layer Output Tuple

是什么：节点 `#146-#154` 运行 MLP：fused gate/up projection `[S,34816]`、`silu_and_mul` 得到 intermediate `[S,17408]`、down projection 回 `[S,5120]`，最后输出 tuple。

为什么需要：attention 后的 FFN/MLP 提供 token-wise nonlinear transformation。当前 FX 图的 `output` 返回 `(mm_3, add_9)`：`mm_3` 是 MLP down projection 结果，`add_9` 是 post-attention residual，下一层会继续使用这两个张量。

怎么做/计算：`#147 t_2` 转置 fused MLP weight 成 `[5120,34816]`；`#148 mm_2` 计算 `[S,5120] -> [S,34816]`。`#149 empty_1` 分配 `[S,17408]`，`#150 _C.silu_and_mul` 将 fused projection 的两半做 `silu(gate) * up` 并写入 `empty_1`。`#152 t_3` 转置 down weight `[17408,5120]`，`#153 mm_3` 得到 `[S,5120]`；`#154 output` 打包 `(mlp_output, post_attention_residual)`。

```text
Token axis S                                  MLP feature width
                                              0              17407 17408             34815
                                              ▲                ▲    ▲                  ▲
fused MLP projection [S,34816]         ──▶   +------------------+----------------------+
                                             | GATE_HALF        | UP_HALF              |
                                             +------------------+----------------------+

activated intermediate [S,17408]       ──▶   +-----------------------------------------+
                                             | SILU(GATE_HALF) * UP_HALF               |
                                             +-----------------------------------------+

down projection [S,5120]               ──▶   +--------------------------------+
                                             | MLP_OUTPUT_ROWS                |
                                             +--------------------------------+
output tuple                            ──▶   [MLP_OUTPUT_ROWS, POST_ATTENTION_RESIDUAL_ROWS]
```

## Process-Level 结论

9 个 selected-layer event 的重建结果一致：每个 event 都被分成 12 个 process，155 个 FX 节点全部分配，无重复分配。`layer 3`、`layer 31`、`layer 59` 在这些 full-attention prefill chunk 上的 layer 内部固定输入 DAG 结构相同；差异只体现在 event 的上下文位置、layer id、forward id 和 `S`。当前 FX 图没有展开 vLLM attention custom op 内部，所以 attention 的 QK score、mask、softmax、weighted V 只能作为 custom op 边界理解，不能从这些 FX 节点中读取内部 kernel 过程。
