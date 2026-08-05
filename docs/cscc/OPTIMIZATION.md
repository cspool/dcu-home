# 保留的优化栈

本文只描述当前 `repro-minimal` 运行时仍可达、且有性能/精度证据的路径。
阶段编号仅用于追溯，不再作为启动接口。

## 1. stride-aware GQA6 长上下文 prefill

主要文件：

- `vllm/v1/attention/backends/rocm_aiter_unified_attn.py`
- `vllm/v1/attention/ops/rocm_aiter_unified_attention_gqa6.py`

精确目标为 gfx936、BF16、head size 256、GQA6、单序列 causal prefill、
cache block 784。GQA6 将每个 KV head 的 6 个 query heads 分成三个双头组；
逻辑 64-token K/V tile 按两个 32-token 数值子块执行 FP32 online softmax。

kernel 直接使用 block table 和真实 tensor stride 读取 KV cache；当一个数值
tile 跨越物理页边界时，在同一 Triton program 中读取下一页，不展开完整 KV
cache，也不再保留独立 page784 wrapper。非目标 shape 或 decode 路径回退到
AITER/vLLM 原实现。

## 2. GDN prefill 固定配置与完整预热

主要文件：

- `vllm/model_executor/layers/fla/ops/{chunk,chunk_o}.py`
- `vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py`
- `vllm/model_executor/layers/fla/ops/gfx936.py`
- `vllm/model_executor/layers/fla/ops/{solve_tril,wy_fast,utils}.py`
- `vllm/model_executor/models/qwen3_next.py`

`gfx936.py` 统一 gate gfx936 Qwen3.5 GDN 的张量 shape、dtype、layout、
`cu_seqlens=int32` 和 T=4096 条件。命中时绕过大规模在线 autotune，使用离线
验证的 Triton meta/compiler 配置；短 residual prefill 按 BT=16/32/64 使用
固定 `chunk_o` 配置。

初始化阶段预热 16/32/64/4096 token、无初态/有初态和是否输出末态的实际
变体，避免首次长请求承担编译开销。

## 3. GDN decode 与 Qwen3.5 数据搬运

主要文件：

- `vllm/model_executor/layers/fla/ops/fused_recurrent.py`
- `vllm/model_executor/layers/rocm_qwen35_gdn.py`
- `vllm/model_executor/models/qwen3_5.py`
- `vllm/model_executor/models/qwen3_next.py`
- `vllm/v1/attention/backends/gdn_attn.py`

单 token packed decode 使用 4-warp、BV=32、1-stage 配置。内部调用来自已
验证的模型元数据，因此热路径跳过重复 Python shape 校验。

Qwen3.5 专用路径还包含：

- 无历史 state 的 prefill 直接传 `None`，避免读取、复制并清零大 state；
- `core_attn_out` 延迟到真实 token 写入后只清 padded tail；
- gfx936 BF16 `(48,128)` GDN output 使用 strided-z fused RMSNorm+SiLU；
- metadata builder 在 CPU 侧计算 `has_initial_state_any/all`，避免 GPU 同步。

## 4. 固定形状 BF16 decode GEMV

主要文件：

- `csrc/rocm/{ops.h,skinny_gemms.cu,torch_bindings.cpp}`
- `vllm/_custom_ops.py`
- `vllm/model_executor/layers/rocm_qwen35_gemv.py`
- `vllm/model_executor/layers/utils.py`

外层统一从 `rocm_unquantized_gemm` 按精确 shape 分发。原来需要手写 load、
FP32 FMA、wave shuffle、LDS 和 launch bounds 的 output projection 已改为
Triton reduction；K=5120 的多行 pair-reduction 仍保留现有 HIP kernel。其
源码只做语法等价压缩，640 threads、320 pairs、wave64、5 个归约 wave、
K chunk=640、2/4 rows per block 及 load/FMA/reduction 顺序均不变：

| Weight `(M,K)` | 实现 |
| --- | --- |
| `(5120,17408)` | 16-warp Triton reduction，`BLOCK_K=2048` |
| `(96,5120)` | HIP，4 rows/block，640-thread pair reduction |
| `(14336/16384/34816/248320,5120)` | HIP，2 rows/block，640-thread pair reduction |

两条路径都只接受 gfx936、BF16、单 token、连续且无 bias 的精确 shape；其他
GEMM 走原路径。`qwen35_bf16_gemv(weight, input)` 现在只承载 K=5120 HIP
shape，C++ 自行选择 rows/block。旧 K=17408 手写 kernel、generic 320-thread
kernel、旧 `LLMM1Strided` ABI、SwiGLU 融合实验和所有 disable 环境变量均已
删除。

## 5. M=4096 rocBLAS profile 与静态编译图

主要文件：

- `vllm/platforms/rocm.py`
- `vllm/platforms/rocm_tunableop.py`
- `vllm/platforms/tunable_profiles/gfx936_qwen3_5_27b_bf16_tn_m4096.csv`
- `vllm/v1/worker/gpu_worker.py`

精确目标自动设置 `compile_sizes=[4096]`。显式 opt-in 的 profile 固定 5 个
rocBLAS solutions，关闭 online tuning/record，并在初始化和捕图前两次验证。
profile SHA、validators、运行配置或 API 状态漂移均直接失败，不静默降级到
未知方案。

## 6. 连续 M-RoPE staging

文件：`vllm/v1/worker/gpu_model_runner.py`。

固定 4096 图使用持久连续 M-RoPE buffer，H2D 直接写入图输入地址；其他动态
长度使用按长度缓存的连续 staging buffer。token 位置值和模型输入语义不变。

## 已删除路径

- gate/up+SwiGLU 自定义 MLP 和 `LLMM1StridedSilu`；
- generic 320-thread strided GEMV；
- runtime no-op backend 强制切换；
- 端到端回归的单 token K6144 pair-reduce；
- 12,000 余行仅用于开发期 trace/实验的仓库内脚本。

上述删除不改变当前 Qwen3.5 服务解析的模型类或实际热内核。三档最差样本
门禁的生成文本逐条一致，详见 [RESULTS.md](RESULTS.md)。
