# C：GEMV、TunableOp、M-RoPE 与 DP2 闭卷实施卡

## 负责范围

- `qwen35_rocm_opt/gemv.py`、`native.py`、`runtime.py`、`profiles/`
- `qwen35_rocm_opt/csrc/`（迁移备用；当前 vLLM 主路径仍可用 `_rocm_C`）
- `csrc/rocm/skinny_gemms.cu`、`setup.py`
- `vllm/model_executor/layers/rocm_qwen35_gemv.py`、`layers/utils.py`
- `qwen3_next.py` 文件顶部 MLP alias/子类这一独立 hunk
- `vllm/platforms/rocm.py::set_device`
- `vllm/v1/worker/gpu_model_runner.py` 的 M-RoPE buffer
- DP2 启动与容量检查脚本

不要修改 GQA6 或 GDN state/FLA 逻辑。

## vLLM 依赖

1. `rocm_unquantized_gemm_impl` 是无量化 linear 的最终 dispatch 点；portable
   GEMV 必须在 rocBLAS fallback 之前、且仅在 `bias is None` 时调用。
2. 当前主路径从 `torch.ops._rocm_C.LLMM1` 取得 K=5120 provider。若新 vLLM
   不构建该扩展，可预编译并加载 `qwen35_rocm_opt.native`，调用 ABI 相同。
3. `Qwen2MoeMLP`/`Qwen3NextMLP` 必须仍暴露 `gate_up_proj.weight`、`act_fn`、
   `down_proj` 和可选 `expert_gate`。
4. `RocmPlatform.set_device` 或等价初始化 hook 必须早于第一个 TunableOp GEMM。
5. M-RoPE 依赖 runner 持有成对 CPU/GPU buffer，并在 CUDA Graph 输入地址稳定
   后执行 H2D copy。

## 固定 GEMV 契约

| Weight `(M,K)` | 实现 | rows/warps |
| --- | --- | --- |
| `(96,5120)` | HIP pair-reduction | 4 rows/block，640 threads |
| `(14336/16384/34816/248320,5120)` | HIP pair-reduction | 2 rows/block，640 threads |
| `(5120,17408)` | Triton reduction | 16 warps，BLOCK_K=2048 |

禁止用 Triton 替换 K=5120：现有实验没有成功候选，wave10 与 persistent-row
变体均已否决。HIP 的 load/FMA/reduction 顺序必须保持。

## 闭卷步骤

1. 将 K=5120 精确 gate 加入 `_rocm_C.LLMM1`，构建 ROCm extension；或使用
   独立 native provider。两者 provider 签名均为 `(weight,input,rows)->output`。
2. 在统一 GEMM dispatch 中调用 portable `qwen35_gemv`；失败必须返回官方
   GEMM，不得接管多 token、bias、FP16、非连续或非 gfx936 情况。
3. dense MLP 的 gate/up 可直接调用同一 GEMV，再保留宿主 `act_fn/down_proj`。
4. device 初始化时加载 portable package 内的 5 validators+5 results profile。
5. 固定 `compile_sizes=[4096]` 时使用连续 `[3,4096]` M-RoPE buffer；其他模式
   保留 `+1` dummy column，并在必要时复制到复用 scratch。
6. DP2 固定 TP=1、DP=2、batch token=4096、memory utilization=0.95；首次编译
   后同参数重启并核对两 rank KV cache token 容量。

## 独立验收

- K=5120 五个 M 的 shape gate、三 seed 数值和 provider dispatch 计数。
- 独立 provider 与 499 `_rocm_C`：M=96/14336/16384/34816 逐位一致；可执行
  verifier 的 2026-08-05 最终同卡 1000 次×11 轮交错计时中位差为
  `+0.857%/-0.002%/+0.002%/-0.000%`，全部低于 1%。
- K=17408 Triton 与冻结 kernel/torch linear 数值通过，并复用既有微基准。
- profile 解析必须恰好得到 5 validators 和 5 results，任一不匹配 fail closed。
- M-RoPE 固定图返回 contiguous 且地址稳定；动态图不在每步分配新 tensor。
- DP2 concurrency 2/4/8×三档全部成功、无 OOM，再运行 110/110 精度。
