# 从官方原版实现 600 行压缩优化

## 1. 文档目的

本文回答“如何尽可能参照官方原版代码，通过局部修改得到当前高性能实现”。
唯一代码基线是 OpenDAS `vllm_cscc`：

```text
official commit: fa718036bdb9dfd80a872b86c8ac16c9d02bfd31
framework: vLLM 0.18.1
model: Qwen3.5-27B BF16
primary target: one gfx936, TP/PP/DP=1
```

`pra2026-bh408` 只提供历史高性能实现和实验结论，不是补丁基线；不要从该目录
整文件复制，也不要恢复已经删除的 `qwen35_rocm_opt` 独立包。当前策略是：

- 优先在官方已有函数中增加短 gate 和 fast path；
- 条件不满足时继续执行原官方代码；
- 只有官方没有合适承载位置时才新增文件；
- 用相对 `fa718036` 的 diff，而不是相对当前 HEAD，审计修改量。

性能与精度结果另见 [MODULAR_3K_PARITY.md](MODULAR_3K_PARITY.md)。

## 2. 先阅读的官方材料

以下文档都来自本仓库携带的上游 vLLM 官方文档，可直接作为实现依据：

| 官方文档 | 对本实现的参考价值 |
| --- | --- |
| [CustomOp](../design/custom_op.md) | 设备专用实现必须有明确 dispatch，并保留 native/backend fallback |
| [Attention backend](../design/attention_backends.md) | backend 选择、手动选择和平台 priority 的边界 |
| [Paged Attention](../design/paged_attention.md) | block table、KV page、Q/K/V 线程组织和 softmax 数据流 |
| [torch.compile integration](../design/torch_compile.md) | `compile_sizes`、静态 shape、graph 输入地址和预编译语义 |
| [Profiling vLLM](../contributing/profiling.md) | 用少量请求定位 kernel 占比，profile 本身不进入正式吞吐 |
| [Incremental compilation](../contributing/incremental_build.md) | 修改 `csrc/` 后的 CMake 增量构建与 native 安装思路 |
| [ROCm installation](../getting_started/installation/gpu.rocm.inc.md) | PyTorch/ROCm/Triton/AITER 版本必须配套 |
| [Optimization and tuning](../configuration/optimization.md) | chunked prefill、并行策略和 attention backend 的通用配置边界 |

阅读官方代码时固定到基线提交，避免把当前实现误认成官方内容：

```bash
git show fa718036:vllm/platforms/rocm.py | less
git show fa718036:vllm/model_executor/models/qwen3_next.py | less
git show fa718036:csrc/rocm/skinny_gemms.cu | less
```

## 3. 最小文件边界

相对官方原版，运行时代码共有 18 个文件：15 个官方文件局部修改，3 个必要新增
文件。统计范围固定为 `csrc/ + setup.py + vllm/`。

### 3.1 直接修改官方文件

| 类别 | 官方文件与原函数锚点 | 当前局部修改 |
| --- | --- | --- |
| Native GEMV | `csrc/rocm/skinny_gemms.cu::LLMM1` | 在原 ABI 内加入 K17408、K5120 和 GateUp 分支，保留原 kernel 尾部 |
| GEMM dispatch | `layers/utils.py::rocm_unquantized_gemm_impl` | 在官方通用 GEMM 前尝试精确 GEMV，失败继续原流程 |
| Qwen MLP/GDN | `models/qwen3_next.py` | 短 MLP 子类、GDN warmup、state 和 padding fast path |
| Qwen3.5 output | `models/qwen3_5.py::Qwen3_5GatedDeltaNet.forward` | `empty` output 与 fused RMSNorm+SiLU |
| GDN prefill | `chunk_o.py::chunk_fwd_o` | 固定 launch 配置 |
| GDN prefill | `chunk_scaled_dot_kkt.py::chunk_scaled_dot_kkt_fwd` | 固定 launch 配置 |
| GDN prefill | `solve_tril.py::solve_tril` | 固定 launch 配置 |
| GDN prefill | `wy_fast.py::recompute_w_u_fwd` | 固定 launch 配置 |
| GDN decode | `fused_recurrent.py::fused_recurrent_gated_delta_rule_packed_decode` | 精确 shape 下 4 warp/1 stage |
| GDN metadata | `gdn_attn.py::GDNAttentionMetadataBuilder.build` | CPU 侧求 initial-state uniformity |
| Attention host | `rocm_aiter_unified_attn.py::RocmAiterUnifiedAttentionImpl` | 精确 GQA6 gate 与官方 AITER fallback |
| ROCm platform | `rocm.py::_get_backend_priorities`、`set_device`、`apply_config_platform_defaults` | backend、profile 和静态 4096 |
| Model runner | `gpu_model_runner.py` | 连续 M-RoPE buffer 与 scratch |
| Worker | `gpu_worker.py::Worker.init_device` | distributed 选卡后调用 platform loader |
| Build | `setup.py` | 构建 `_rocm_C` 并打包 profile CSV |

### 3.2 必要新增文件

| 新文件 | 必要性 |
| --- | --- |
| `vllm/model_executor/layers/fla/ops/gfx936.py` | 共享设备 gate、GDN launch helper、fused norm 和 GEMV 分派；避免在多个官方文件重复实现 |
| `vllm/v1/attention/ops/rocm_aiter_unified_attention_gqa6.py` | 官方基线没有 Q24/KV4/D256/page784 的窄范围 Triton op |
| `vllm/platforms/tunable_profiles/gfx936_qwen3_5_27b_bf16_tn_m4096.csv` | 固定五个 M=4096 rocBLAS solution，不能硬编码进通用 Python 逻辑 |

除此之外不需要新增 Python package、C++ binding 层、权重格式或 scheduler 模块。

## 4. 优先级：性能、难度与实施顺序

### 4.1 证据口径

当前 R24 是组合版本，最终全量测试证明整栈有效，但并非每项都在 R24 上重新做
单独消融。下表只引用已有可归因证据：

- GQA6 长 prefill 的历史 same-input raw kernel 相对上一实现为
  `1.68x–3.11x`，对应三档 mean TTFT 约下降
  `15.9% / 20.5% / 24.7%`；
- K5120 640-thread pair reduction 相对 320-thread 版本快
  `7.7%–8.5%`，同文本小样本 mean TPOT 约下降 `4.9%–5.2%`；
- GDN T4096 四个目标 kernel 按 profile 权重合计降时 `14.06%`，但折算全
  trace 仅约 `0.48%`；
- TunableOp 五 key portfolio 的 isolated wall reduction 为约 `5.72%`，其
  单项端到端贡献未独立闭环；
- M-RoPE 专项历史理论端到端上限约 `0.18%–0.35%`。

K17408、GDN decode/RMSNorm 和 page784 的当前收益包含在最终整栈中，没有同一
R24 wheel 的独立官方-base 消融，因此不写伪精确百分比。

### 4.2 纯性能排序

| 性能名次 | 优化簇 | 主要指标 | 结论 |
| ---: | --- | --- | --- |
| 1 | GQA6 prefill + page784 | TTFT，长上下文 | 最大结构性收益，8--32K 最关键 |
| 2 | K5120 GEMV + GateUp/SwiGLU | TPOT | 每层多次命中，decode 主收益之一 |
| 3 | K17408 output/down GEMV | TPOT | 单 token 重复命中，收益小于 K5120 组合但不可缺失 |
| 4 | 五行 TunableOp + 静态 4096 | TTFT/图内 GEMM | 代码短、工具链相关，组合收益明显 |
| 5 | GDN prefill 固定配置 + fused norm | TTFT | 48 个 GDN 层累计，但单独 Amdahl 上限较小 |
| 6 | GDN decode 4 warp + state/清零优化 | TPOT/host sync | 算子局部可快，整服务占比低且数值路径敏感 |
| 7 | 连续 M-RoPE staging | 图稳定性/小幅 TPOT | 收益最小，主要消除 strided copy/compile 干扰 |

### 4.3 相对官方修改难度

难度 1 表示单文件配置改动，5 表示需要新 kernel、内存布局和端到端数值审计。

| 从易到难 | 优化簇 | 难度 | 主要风险 |
| ---: | --- | ---: | --- |
| 1 | profile 打包、loader、静态 4096 | 2/5 | 工具链 validator 或加载时机漂移 |
| 2 | 连续 M-RoPE staging | 3/5 | CUDA Graph 输入地址、动态 slice 连续性 |
| 3 | GDN prefill 固定 launch | 3/5 | 绕过 autotune 后 shape/meta 不完整 |
| 4 | K17408 GEMV | 4/5 | BF16 规约树、1024-thread 资源和 ABI gate |
| 5 | GDN decode/state/fused norm | 4/5 | recurrent state、padding、mixed batch 与文本漂移 |
| 6 | K5120 GEMV + GateUp | 5/5 | wave/LDS 规约、BF16 staging、融合输出 ownership |
| 7 | GQA6 + page784 | 5/5 | block table、动态 stride、跨页读取、causal softmax/LSE merge |

### 4.4 推荐实施顺序

综合性能、依赖和调试成本，建议按以下顺序从官方原版落地：

1. **基础门与构建**：启用官方 `_rocm_C`，新增 `gfx936.py` 的设备 gate，先保证
   所有非目标 shape 回官方路径。
2. **Decode GEMV**：先 K5120，再 K17408，逐 shape 做 same-input 和非法输入
   fallback；这是最容易用微基准闭环的高收益路径。
3. **GQA6 基础 prefill**：先实现不含 page784 merge 的精确 GQA6 kernel，核对
   compact/interleaved cache stride，再加入 page784 三态 LSE merge。
4. **静态 prefill 运行时**：加载五行 TunableOp profile、设 4096 compile shape，
   再处理 M-RoPE 连续 buffer。
5. **GDN prefill**：只替换 launch schedule，不重写官方五个数学 kernel。
6. **GDN decode 与融合**：最后处理 4 warp、CPU metadata、tail 清零和 RMSNorm，
   因为这组最容易出现“算子正确但生成文本漂移”。
7. **组合小样本 → 全量**：每叠加一簇先做 3–5 条三档小样本；全部方向对齐后
   才运行 50×3 吞吐与 110 条精度。

若目标是最快获得可见性能，而不是最稳妥开发，优先做第 2、3 步；若目标是最小
风险建立可运行版本，先做第 1、4 步。

## 5. 按官方函数实施

### 5.1 `_rocm_C` 与共享 gate

官方 `setup.py` 已保留被注释的 `_rocm_C` 扩展入口，直接恢复 ROCm 条件分支，
并在 `package_data["vllm"]` 增加 profile glob。不要新建第二个 extension。

`gfx936.py` 的公共 gate 只缓存
`gcnArchName.startswith("gfx936:")`；具体优化还必须检查 model、BF16、shape、
stride、连续性和功能条件。helper 返回 `None` 或原 kernel，宿主据此自然回到
官方实现。

### 5.2 K5120 与 K17408 GEMV

官方锚点是 `skinny_gemms.cu::LLMM1`。在该函数之前添加两个窄 kernel，在官方
dtype/rank 校验之后添加 early return；原 `AT_DISPATCH_REDUCED_FLOATING_TYPES`
及通用 `LLGemm1_kernel` 保持为最后 fallback。

| Weight `(M,K)` | 当前 launch |
| --- | --- |
| `(96,5120)` | 640 threads，4 rows/CTA |
| `(14336/16384/34816/248320,5120)` | 640 threads，2 rows/CTA |
| `(5120,17408)` | 1024 threads，1 output row/CTA |

K5120 把 10 个 wave 两两规约为 5 个 wave；K17408 用 16 个 wave 的 LDS leader
规约。全部读取官方 BF16 权重，FP32 累加，写回 BF16，不创建量化权重。

在官方 `rocm_unquantized_gemm_impl` 获取 `n/m/k` 后、通用 backend 判断前，
仅在 `bias is None` 时调用 `qwen35_gemv`。MLP GateUp 只在
`expert_gate is None` 时调用融合分支，失败执行 `super().forward(x)`。

### 5.3 GDN prefill：只改 wrapper launch

官方数学 kernel 不复制。`gdn_kernel()` 在精确目标下解包 autotune wrapper 的
raw kernel 并注入固定配置；不命中时返回官方 wrapper 和空 options。

| 官方 wrapper | 固定目标 |
| --- | --- |
| `chunk_fwd_o` T16 | BK32/BV32，2 warps，2 stages |
| `chunk_fwd_o` T32 | BK32/BV32，2 warps，3 stages |
| `chunk_fwd_o` T64 | BK32/BV64，4 warps，2 stages |
| `chunk_fwd_o` T4096 | BK128/BV128，4 warps，1 stage |
| `chunk_scaled_dot_kkt_fwd` T4096 | BK128，4 warps，1 stage |
| `solve_tril` T4096 | 2 warps，1 stage |
| `recompute_w_u_fwd` T4096 | 2 warps，1 stage |

单 stage 再传 `waves_per_eu=1,matrix_instr_nonkdim=16,kpack=2`。模型 warmup
覆盖 T16/32/64/4096、state/no-state，避免正式请求首次 JIT。

### 5.4 GDN decode、state 与 fused norm

在官方 packed recurrent wrapper 计算 `B/H/HV/K/V` 后，只对
`(1,16,48,128,128,BF16)` 选择 `BV=32,4 warps,1 stage`；其他情况保留官方
`3 stages/1 warp`。

metadata builder 使用已有 CPU context length 得到三态：全无 state、全有 state、
mixed。只有“全无且 native GDN”时传 `initial_state=None`；mixed 仍 gather 并按
mask 清零。`core_attn_out` 可以用 `empty`，但真实 token 写完后必须显式清零
padding tail。

fused RMSNorm 只接管 `[T,48,128]` BF16 和精确 z stride
`(16384,128,1)`；其余布局调用官方 norm。不要只做 operator allclose，必须覆盖
state/no-state、prefill/decode 和最终生成文本。

### 5.5 GQA6 与 page784

官方宿主锚点是 `RocmAiterUnifiedAttentionImpl.__init__/forward`：初始化时建立
一次性 model/head gate，forward 时再检查 query、cache、output、block size、
dtype、stride 和单序列条件。

命中顺序：

1. later-prefill 满足 page784 条件时尝试 `page784_prefill`；
2. 其余目标 prefill 调用 GQA6 Triton kernel；
3. 任一条件失败调用官方 `self.unified_attention`；
4. decode 始终保留官方 AITER 路径。

GQA6 kernel 每 CTA 处理两个 Q heads，使用动态 `key_cache.stride()`；不能把
服务中的交错 cache 首维写死为 `784*4*256`。page784 把每页拆成 768 main 与
16 tail，分别计算 main、residual、current 三态，并用 FP32 LSE 稳定合并。

最低数值门是：同一输入的 compact cache 与 interleaved cache 结果一致；同时
覆盖跨页 boundary、query 128/4096、fallback 和完整服务文本。R23 的最大性能
遗漏正是错误要求 cache contiguous，导致所有目标 prefill 回退通用 kernel。

### 5.6 TunableOp、静态图和 M-RoPE

官方 `RocmPlatform.set_device()` 原本只设置设备。worker 完成 distributed
初始化后再次调用该入口，确保 profile 在正确 DCU 上加载；loader 关闭 tuning 和
record、设置 CSV，并要求结果数为 5。

官方 `apply_config_platform_defaults()` 中只对 gfx936、Qwen3.5、BF16、4096、
world/DP=1、无 speculative、用户未指定 compile sizes 的组合设 `[4096]`。

固定 4096 图下把 M-RoPE buffer 宽度设为 4096；动态图保持官方额外 dummy
column 语义，并把非连续动态 slice 复制到持久 scratch。不要每 token 新分配
scratch，也不要修改 position 值。

## 6. 生成可直接应用到官方原版的补丁

在当前 modular 工作树执行：

```bash
OFFICIAL=fa718036bdb9dfd80a872b86c8ac16c9d02bfd31
PATCH=/path/to/output/qwen35-gfx936-r24.patch

git diff --binary --full-index "$OFFICIAL" -- csrc setup.py vllm > "$PATCH"
git diff --check "$OFFICIAL" -- csrc setup.py vllm
git diff --name-status "$OFFICIAL" -- csrc setup.py vllm
git diff --numstat "$OFFICIAL" -- csrc setup.py vllm | \
  awk '{a += $1; d += $2; n += 1} END {print "files=" n, "add=" a, "del=" d, "churn=" a+d}'
```

预期为：

```text
files=18 add=567 del=33 churn=600
```

在独立官方 worktree 验证补丁，不污染当前目录：

```bash
git worktree add /path/to/official-r24 fa718036
git -C /path/to/official-r24 apply --check "$PATCH"
git -C /path/to/official-r24 apply "$PATCH"
git -C /path/to/official-r24 diff --check
```

这里的 patch 是最直接的“参考官方修改得到”交付物：每个上下文 hunk 都以官方
代码为锚，不依赖 3k Git 历史或当前 HEAD 的中间重构提交。

## 7. 验证顺序

### 7.1 静态与构建

1. `git diff --check fa718036 --`；
2. 文件数与 churn 为 `18 / 600`；
3. Python 文件可编译，Triton import 不在非 ROCm 环境提前初始化设备；
4. 干净 wheel 含 `_rocm_C`、GQA6、gfx936 helper 和 CSV；
5. wheel 不含独立实验包、`.pyc`、build cache 或权重。

### 7.2 算子

1. GEMV：所有目标 M/K、多 seed、zeros/特殊值、repeat、非法 shape fallback；
2. GDN：T16/32/64/4096、state/no-state、padding、decode；
3. GQA6：短/长 query、compact/interleaved stride、跨 page、fallback；
4. page784：main/residual/current 单独与合并结果、LSE finite、workspace 上限；
5. profile：恰好 5 validators + 5 results，禁止 online tuning。

page784 与只读 3k 历史的同输入计时可使用
`scripts/benchmark_page784_parity.py --history /path/to/results.json --output /path/to/output.json`；
脚本只读取显式提供的历史 JSON，不启动或修改 3k 归档。

### 7.3 服务

先用每档 3–5 条做方向筛选，再做：

- DCU 0 三档各 2 warmup + 50 正式请求；
- 官方原始吞吐 `total_output_tokens / duration`，不做长度归一化；
- TTFT P99、全局 TPOT P99、完成率和日志审计；
- HotpotQA 20、GovReport 30、Retrieval 30、Aggregation 30，共 110 条精度。

当前完整结果和哈希见 [MODULAR_3K_PARITY.md](MODULAR_3K_PARITY.md)。

## 8. 规则边界

当前代码加载官方 BF16 safetensors，kernel 内仅做 BF16 load、FP32 累加和 BF16
store；没有创建持久量化权重、剪枝、重排压缩或模型格式转换。没有修改采样、
输出长度、层/head/token 数、batch scheduler 或统一 API。

初赛规则要求单卡、并发 1；可选 DP=2 仅为决赛多卡准备，配置与当前验证边界见
[构建与启动简明流程](BUILD_SERVE_CACHE_QUICKSTART.md)。决赛具体卡数与评分规则公布后再
复验，不能将 DP=2 历史吞吐计入初赛成绩。

## 9. 不要恢复的旧路径

- `qwen35_rocm_opt` 独立 package 和重复 native binding；
- `rocm_qwen35_gemv.py` 中间分派层；
- 80-CTA/seg20、decode-bank/BV8、attention 20 segments 等已拒绝探索；
- 持久权重量化、投机解码、prefix cache 或 scheduler 修改；
- 只因单个 operator 快就跳过服务文本与全量精度验证。

这些路径要么与官方原版直接修改目标冲突，要么没有形成当前可接受的性能/精度
闭环。
