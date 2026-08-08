# 600 行 modular 对 3k 最优版复现报告

本文是最终性能、精度和规则证据；如何以官方 `fa718036` 为起点逐文件实现、
以及各优化的性能/难度优先级，见
[官方原版优化实施指南](OFFICIAL_BASE_OPTIMIZATION_GUIDE.md)。可选 DP=2 配置
单独保存在 [DP2_MULTI_REQUEST.md](DP2_MULTI_REQUEST.md)，不参与本页单卡结果。

## 1. 目标、对象与边界

- 官方源码基线：`fa718036bdb9dfd80a872b86c8ac16c9d02bfd31`。
- 3k 最优归档：`pra2026-bh408`，提交
  `0abe1e1d9b911e68dfdd8c2c9ba91f46b94306cd`；本轮只读历史结果和源码，
  没有启动、构建或修改该归档。
- 最终源码：`pra2026-bh408-modular`；运行候选为
  `modular_validation/candidates/modular_compact3k_r24/site`。
- 设备与服务：只暴露物理 DCU 0，BF16，TP/PP/DP=1，
  `max_num_batched_tokens=4096`、`max_num_seqs=128`、
  `gpu_memory_utilization=0.95`。
- 禁用项：无权重量化、无权重重排/压缩缓存、无投机解码、无 prefix cache、
  无采样参数或 scheduler 语义修改。
- 目标：相对官方源码的运行时代码 churn 不超过 600 行；三档官方原始输出吞吐
  分别不低于 3k 的 99%；四项精度损失不超过 1%。

最终运行时代码口径为 `csrc/ + setup.py + vllm/`，相对官方源码新增 567 行、
删除 33 行，churn **正好 600 行**；文档、测试脚本、结果和构建产物不计入。

## 2. 压缩实现总览

### 2.1 gfx936 目标门与短分派

`vllm/model_executor/layers/fla/ops/gfx936.py` 用一个缓存的
`gcnArchName.startswith("gfx936:")` 判定复用所有目标门；每条优化还同时检查
Qwen3.5 的精确 shape、BF16、连续性/stride 和功能条件，不匹配立即回官方路径。
这取代了 3k 中分散在多个宿主文件的重复设备、模型和 shape 判断。

### 2.2 K=17408 output GEMV（HIP）

- 入口只接管 `M=5120,K=17408,N=1,BF16,rows_per_block=1`。
- `LLGemm1_k17408_kernel` 每个输出行一个 CTA、1024 threads；每次以 `float4`
  读取 8 个 BF16 权重，16 个 wave 先 shuffle 再经 16-float LDS 做最终规约。
- `qwen35_dot8<true>` 用 gfx936 原生 `v_fmac_f32`，保持 BF16 输入、FP32 累加、
  BF16 输出，不改变权重格式。
- 短实现直接放入既有 `_rocm_C.LLMM1` ABI，不新增 Python 扩展或绑定层。

关键位置：`csrc/rocm/skinny_gemms.cu:236,255,352,372`。

### 2.3 K=5120 GEMV 与 GateUp/SwiGLU（HIP）

- 精确 shape 为 `K=5120` 且
  `M in {96,14336,16384,34816,248320}`，只处理 `N=1,BF16`。
- `LLGemm1_k5120_pairreduce640_kernel` 固定 640 threads：先把十个 wave 两两合并
  成 320 lanes，再由五个 wave 完成行规约；`M=96` 每 CTA 算 4 行，其余每 CTA
  算 2 行，减少 CTA 数和权重重复加载。
- `M=34816` 的 GateUp 通过 `rows_per_block=-2` 复用同一 ABI：第一遍算 17408 行
  gate，第二遍算 up，并在写回前执行 BF16-staged `SiLU(gate) * up`，输出直接是
  17408 维，省掉独立激活 kernel 和 34816 维中间张量。
- 通用线性层只在 `bias is None` 时尝试该分派；Qwen3.5 MLP 仅在没有
  `expert_gate` 时尝试融合 GateUp，失败即回父类实现。

关键位置：`csrc/rocm/skinny_gemms.cu:284-395`、
`vllm/model_executor/layers/fla/ops/gfx936.py:60`、
`vllm/model_executor/layers/utils.py:128`、
`vllm/model_executor/models/qwen3_next.py:112`。

### 2.4 GDN prefill 固定调度（Triton）

3k 的核心不是重写五个数学 kernel，而是绕过目标 shape 的 autotune wrapper，
直接使用已知最优 launch。压缩版用 `gdn_kernel()` 统一解包 `kernel.fn.fn` 并注入
配置，五处调用各只增加两三行：

| kernel | 目标配置 |
| --- | --- |
| chunk-o T=16 | `BK=32,BV=32,num_warps=2,num_stages=2` |
| chunk-o T=32 | `BK=32,BV=32,num_warps=2,num_stages=3` |
| chunk-o T=64 | `BK=32,BV=64,num_warps=4,num_stages=2` |
| chunk-o T=4096 | `BK=128,BV=128,num_warps=4,num_stages=1` |
| scaled KKT T=4096 | `BK=128,num_warps=4,num_stages=1` |
| solve-tril T=4096 | `num_warps=2,num_stages=1` |
| recompute-w/u T=4096 | `num_warps=2,num_stages=1` |

所有单 stage 配置再固定
`waves_per_eu=1,matrix_instr_nonkdim=16,kpack=2`；门控要求 batch=1、目标 GDN
head shape、varlen metadata 和 gfx936。T=4096 以及有/无 initial state 在模型 warmup
阶段预编译，正式计时不承担首次 JIT。

关键位置：`gfx936.py:8-32`、`chunk_o.py:167`、
`chunk_scaled_dot_kkt.py:142`、`solve_tril.py:546`、`wy_fast.py:140`、
`qwen3_next.py:739-812`。

### 2.5 GDN decode 四 warp 与宿主同步消除

- packed decode 只对
  `(B,H,HV,K,V,dtype)=(1,16,48,128,128,BF16)` 设
  `BV=32,num_stages=1,num_warps=4`；其他 shape 保留官方 `3 stages/1 warp`。
- 已由精确门保证输入时，调用传 `validate=False`，避免每 token 重复 Python 检查。
- metadata builder 从已有 CPU context length 计算
  `has_initial_state_uniform`；全无初始状态时直接传 `initial_state=None`，避免 GPU
  mask、清零和读回同步；混合状态仍走官方安全路径。
- 输出 buffer 从 `zeros` 改为 `empty`，只对 padding tail 显式清零；真实 token 全部
  被 kernel 覆盖，因此不改变语义。

关键位置：`fused_recurrent.py:437-440`、`gdn_attn.py:305-317`、
`qwen3_next.py:818,985,1094`、`qwen3_5.py:198`。

### 2.6 GDN RMSNorm + SiLU gate 融合（Triton）

目标输入为 `x=[T,48,128]`、`z.stride=(16384,128,1)`、BF16。一个 CTA 处理
16 行 × 128 列，4 warps，在 FP32 中完成 RMS、乘 weight、再乘 `z*sigmoid(z)`，
直接写最终 BF16；grid 为 `T*3`，恰好覆盖 48 heads。非目标布局调用官方 norm。

关键位置：`gfx936.py:36-57`、`qwen3_5.py:215`。

### 2.7 GQA6 full-attention prefill（Triton）

- 仅接管 gfx936、BF16、`24 query heads / 4 KV heads / head_dim=256`、单请求
  prefill；decode 保持 AITER。
- 每 CTA 同时处理同一 KV head 的两个 query heads；4 个 KV heads × 3 个 head
  split 形成 grid 第二维 12，第一维为
  `ceil(max_query_len / (BLOCK_M/2))`。
- 长 query 用 `BLOCK_M=64`，内部按两个顺序 32-token subtile 做 online softmax，
  `4 warps,1 stage,waves_per_eu=1,matrix_instr_nonkdim=16,kpack=2`；短 query 用
  `BLOCK_M=16,4 warps,2 stages`。
- R23 曾把 cache 首维写死为 `784*4*256=802816`，并要求 cache contiguous；服务
  实际 K/V 交错布局 stride 为 `1605632`，因此 64 次 prefill 全部误走通用 kernel。
  R24 把四个 cache stride 作为 Triton constexpr 传入，同时要求 K/V stride 相同。
  交错与紧凑 cache 的同输入算子结果逐 bit 相同，最大绝对误差 0。

关键位置：`rocm_aiter_unified_attn.py:131,221-234`、
`rocm_aiter_unified_attention_gqa6.py:88-181`。

### 2.8 page784 later-prefill（Triton + FlashAttention）

每个 784-token cache page 拆为连续的 768-token main 和 16-token tail；所有 tail
以及最后 boundary 被 `_pack_page784` 按逻辑顺序压成 page64。随后分别计算：

1. 768 main 的历史上下文；
2. 打包后的 residual；
3. 本轮 current K/V 的 causal attention；

三态以各自 FP32 LSE 在 `_merge_page784` 中做稳定权重合并。workspace 按设备和
dtype 复用，容量上限为 4096 query tokens 和 96 residual pages；超界或非单请求
直接返回 False。该实现保留 3k 的 page784 数学拆分，但把大段宿主适配压成一个
181 行文件中的前 84 行。

关键位置：`rocm_aiter_unified_attention_gqa6.py:11-84`。

### 2.9 TunableOp 五行静态 profile

环境变量
`VLLM_ROCM_TUNABLEOP_PROFILE=gfx936_qwen3_5_27b_bf16_tn_m4096` 启用内置 CSV。
`RocmPlatform.set_device()` 在 distributed 初始化选定 DCU 0 后创建一次空 tensor，
关闭 runtime tuning/record，使用 `insert_device_ordinal=False` 加载文件，并强制
`get_results()` 恰好为 5 行，否则启动失败。五行均是 M=4096 BF16 TN GEMM：

- N=14336/16384/34816、K=5120：`Gemm_Rocblas_20981`；
- N=5120、K=17408：`Gemm_Rocblas_20979`；
- N=5120、K=6144：`Gemm_Rocblas_20981`。

关键位置：`rocm.py:567-576`、`gpu_worker.py:270`、
`platforms/tunable_profiles/gfx936_qwen3_5_27b_bf16_tn_m4096.csv`。

### 2.10 静态 4096 编译形状与 M-RoPE 连续输入

- 仅当 gfx936、Qwen3.5、BF16、`max_num_batched_tokens=4096`、world/DP=1、
  无 speculative 且用户未指定 compile sizes 时，自动设
  `compilation_config.compile_sizes=[4096]`。
- 对该静态路径，M-RoPE graph buffer 宽度直接为 4096 而不是故意构造的 4097
  非连续 buffer；H2D 每轮固定复制 4096，动态小 slice 才复制到预分配 scratch
  形成连续 view。这样保持 graph shape 固定并消除模型内部的 strided position
  输入，不改变 position 数值。

关键位置：`rocm.py:633-647`、`gpu_model_runner.py:696-711,898-906,1825-1833`。

### 2.11 ROCm backend、构建与 fail-closed 集成

ROCm backend priority 加入 AITER unified attention；`setup.py` 构建既有
`vllm._rocm_C` 并打包 tunable CSV。所有高性能路径都复用官方 ABI；任何设备、
dtype、shape、stride、模型功能条件不匹配都回官方 vLLM/AITER/FLA 实现。

## 3. R23 根因和修复证据

同一 8--16K 请求的 R23 profile 中，K5120、K17408 和全部 GDN kernel 已与 3k
处于约 1% 内；唯一大差距是 64 次通用 `kernel_unified_attention_2d` 累计
4479 ms，而 3k 的 GQA6 + page784 组合约 382 ms。一次性 gate 日志确认 cache
shape 正确但首维 stride 是 `1605632`，R23 的 contiguous 条件为 False。

R24 使用 runtime stride 后，同一 13962-token 样本 TTFT 从 R23 的约 7--9 s
恢复到 3.140 s，且 92-token 文本与历史结果逐字一致；8--16K 五条热跑 mean TTFT
为 3023.87 ms，相对 3k 历史 3038.78 ms 为 -0.49%。

## 4. 全量官方原始吞吐

参数完全沿用官方脚本：每档 2 条 warmup、50 条正式请求、并发 1、request-rate 1、
temperature 0、最大输出 1024。官方原始吞吐即结果 JSON 的
`total_output_tokens / duration`，未做长度归一化或反事实替换。

| 输入档 | 3k 历史 tok/s | R24 tok/s | 相对 3k | mean TTFT | mean TPOT | 成功 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4--8K | 21.60766 | 21.50590 | -0.471% | 1439.48 ms | 40.80 ms | 50/50 |
| 8--16K | 19.12615 | 19.06339 | -0.328% | 3003.02 ms | 41.50 ms | 50/50 |
| 16--32K | 15.92680 | 15.84538 | -0.511% | 5047.31 ms | 42.64 ms | 50/50 |

三档均满足“不低于 3k 99%”；按 20%/50%/30% 权重的相对变化为约 -0.412%。
150/150 正式请求成功，三个 client log 无 traceback、OOM、engine death 或非零
failed request。

结果目录：
`modular_validation/results/parity/full_singlecard_r24_20260808/work/test`。

## 5. 全量精度

同一热服务运行 OpenCompass 四项固定精度集，110/110 API 请求成功：

| 数据集 | 3k 历史 | R24 | 精度损失 |
| --- | ---: | ---: | ---: |
| HotpotQA | 77.96 | 77.96 | 0.00% |
| GovReport | 32.98 | 33.17 | 无损失（略高） |
| Retrieval Multi Point | 100.00 | 100.00 | 0.00% |
| Aggregation Keyword | 100.00 | 100.00 | 0.00% |

结果目录：
`modular_validation/results/parity/full_singlecard_r24_20260808/accuracy_work/accuracy_debug/output/local_accuracy_qwen35/20260808_182838`。

最终后处理表中的 Aggregation 100% 是比赛脚本按无序关键词集合重新核算的结果；
OpenCompass 自带的顺序敏感 `AccEvaluator` 中间表为 0，不是最终比赛口径。

## 6. 约束审计

- 模型仍从官方 BF16 safetensors 加载；代码没有创建、保存或替换权重文件。
- HIP/Triton kernel 只读取原 BF16 权重/激活，使用 FP32 寄存器累加后写 BF16；
  这不是持久化权重量化、剪枝、重排压缩或格式转换。
- 没有修改 sampling、max tokens、输入、层数或 batch scheduler；吞吐测试使用统一
  服务接口和原始脚本参数。
- `speculative_config is None` 是静态优化的必要门，不是启用 speculative；服务
  日志中 `speculative_config=None`、`enable_prefix_caching=False`、
  `quantization=None`。
- 未缓存测试 prompt、答案或可跨样本复用的模型中间结果；page workspace 只复用
  空输出 buffer，不保存请求内容。
- 归档 HEAD 保持 `0abe1e1...`，本轮没有写入 `pra2026-bh408`；其中已有的其他
  perf-trace 工作树改动不属于本轮。

## 7. 未保留的尝试

- 80-CTA/seg20、GDN decode bank/BV8、attention 20 segments 等探索因文本漂移、
  收益不足或不属于“压缩复现 3k”而未并入最终源码。
- 量化仅保留为“若决赛规则允许”的独立备选结论，当前 R24 完全不含量化实现。
- 3k 的大段 GDN direct-plan 宿主适配被统一 `gdn_kernel()` + 四处短调用替代；
  算子 profile 已拉平，因此没有复制其长状态机。
- R23 的硬编码 cache stride 已被删除；这是遗漏 3k 高性能实现关键，而不是新的
  优化方向。

## 8. 可复核哈希

- R24 `_rocm_C.abi3.so`：
  `ee13e2e71904c21049cf4a1ca37e5d819c6b655e3c0963668992496fb813f5fd`。
- 4--8K result：
  `9f49b86bc90c317906e6acd882499177965be515a1171d16ba1e8705e568f870`。
- 8--16K result：
  `4a57e2519af9f33c77e0ad676908cf86e88ce59a176c543eea2a1efe669fa456`。
- 16--32K result：
  `516b816c963f0fb244759c69b1f6f525881f448e7e52f9dff791f54655edba00`。
- 比赛脚本最终精度表 `accuracy_client.log`：
  `0d879e42327a56b5fa136527c9086c31e8c37906f893cb227bf7f94cf8c4e083`。
