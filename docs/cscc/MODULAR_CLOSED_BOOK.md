# Qwen3.5 ROCm 优化模块化与三人闭卷实施总册

## 1. 冻结基线与验收线

已验证的 499 行版本保持冻结，不在模块化工作中继续压缩或改写：

- 工作树：`pra2026-bh408-repro-800`
- 分支：`repro-800`
- 提交：`773645327ace48233980a3eba85d7df8db65c91e`
- 相对 OpenDAS `fa718036`：新增 469 行、删除 30 行，共 499 行

模块化工作位于独立的 `repro-modular` 分支。官方单卡历史口径仍以压缩前最优
分数 `91.676381851` 为基准；允许下降 1% 时硬下限为 `90.759618032`，冻结
499 版的 `91.080201612` 已占用约 `0.6503%`。本轮用户指定的是双卡交付，
因此双卡门禁单独与 3k/2656 行版本的匹配 DP2 c8 三档比较，要求总体比例
不低于 99%。必须复跑 110 条精度以及双卡全矩阵，不能仅用 kernel 微基准
宣称达标。若后续改回官方单卡计分拓扑，还必须追加单卡三档复测。

## 2. 模块边界

```text
                         qwen35_rocm_opt（不导入 vLLM）
                  ┌────────────────┬────────────────┐
                  │                │                │
          A. Attention       B. GDN/FLA       C. GEMV/Runtime
          attention.py       gdn.py           gemv.py/native.py
          target.py          target.py        runtime.py/profile
                  │                │                │
                  └────── vLLM 薄适配层与明确张量契约 ──────┘
                         official vLLM / future SGLang
```

允许依赖 vLLM，但依赖必须停留在适配层，并在本册中按“符号、字段、shape、
回退路径”记录。性能 kernel 不直接导入 vLLM。这样决赛换版本时只重写适配层，
不重写已经验证的 kernel。

## 3. 三人划分

| 人员 | 独占模块 | 主要宿主文件 | 独立验收 |
| --- | --- | --- | --- |
| A | GQA6 prefill、page64/page784、AITER fallback | `rocm.py` 的 backend priority；`rocm_aiter_unified_attn.py` | 短 prefill、长 prefill/page784 跨页、decode fallback |
| B | GDN 配置、packed decode、state 快路径、output 复用、RMSNorm+SiLU | FLA 五个 op 文件、`gdn_attn.py`、Qwen GDN 两个模型文件 | GDN kernel 配置、B=1/2/4/8 decode、state/no-state、tail 清零 |
| C | K=5120 HIP、K=17408 Triton、TunableOp profile、M-RoPE、DP2 启动 | GEMM dispatch、`skinny_gemms.cu`、`gpu_model_runner.py`、profile loader | GEMV shape/数值/延迟、profile 5+5、连续 positions、DP2 无 OOM |

三人从同一官方基线分别工作。A 与 C 都修改 `vllm/platforms/rocm.py`，但锚点
分别是 `_get_backend_priorities` 和 `set_device`；B 与 C 都修改
`qwen3_next.py`，但锚点分别是 GDN 类和文件顶部的 MLP alias。它们是独立
hunk，禁止跨模块顺手整理，以保证合并可自动完成。

## 4. vLLM 依赖总表

| 依赖 | 使用模块 | 依赖内容 | 换版本时的迁移判断 |
| --- | --- | --- | --- |
| `RocmAiterUnifiedAttentionImpl` | A | cache update 后的 Q/K/V、output buffer 和 metadata | 核对 forward 是否仍在同一层更新 cache；字段名变化只改适配层 |
| `FlashAttentionMetadata` | A | `query_start_loc`、`seq_lens`、`max_query_len`、`max_seq_len`、`block_table` | 建立字段映射；不能从 Python request 对象重新推导热路径 metadata |
| AITER `unified_attention` | A | decode 与不满足精确 gate 的 fallback | AITER 不存在时保持官方 backend；不得让自定义 prefill 接管 decode |
| FLA Triton kernel decorators | B | `early_config_prune` 接入五个 autotune kernel | 若装饰器 API 变化，移植 `_CHUNK_O` schedule，不复制整份 FLA |
| FLA packed recurrent kernel | B | host 将 kernel object 注入 `launch_packed_decode` | 核对参数顺序、stride 参数和 grid；portable core 不拥有该 kernel |
| `GDNAttentionMetadataBuilder` | B | CPU 侧 state 是否统一、spec/non-spec mask | 字段变化时只维护 `has_initial_state_uniform` 的产生与消费链 |
| `Qwen3NextGatedDeltaNet` | B | warmup、state gather、output alias、tail 清零 | 以方法名和张量语义为锚点，不以行号移植 |
| `rocm_unquantized_gemm_impl` | C | N=1、无 bias 时的精确 GEMV dispatch | 若入口改名，在最终 rocBLAS/linear 前插入同一 shape gate |
| `torch.ops._rocm_C.LLMM1` | C | 主路径 K=5120 HIP provider | vLLM 扩展不可用时使用独立 `qwen35_rocm_opt.native` provider |
| `GPUModelRunner` M-RoPE buffers | C | CPU/GPU persistent buffer、graph input、H2D copy | 映射 buffer owner 和 capture 时机；不要在 hot path 新分配 tensor |
| `RocmPlatform.set_device` | C | 加载冻结 TunableOp profile | 若宿主无该 hook，在创建首个 GEMM 前调用 `load_tunable_profile` |
| vLLM internal MP DP | C | `--data-parallel-size 2`、TP=1、两 rank KV cache | SGLang 迁移时只迁移启动参数与容量检查，不迁移 vLLM scheduler 类型 |

### 4.1 已验证版本与依赖级别

下表记录的是本轮实际构建、数值和微基准验证过的环境，不表示可接受的版本范围。
升级后是否兼容，应以第 4 节的符号、字段和张量契约为准，不能只比较版本号。

| 依赖 | 已验证版本/目标 | 级别 | 可替换性 |
| --- | --- | --- | --- |
| Python | 3.10 | L0 必需 | 可升级，但要重建 native extension 与 vLLM wheel |
| PyTorch ROCm | `2.10.0`，`torch.version.hip=6.3.26093` | L0 必需 | tensor/stream/extension ABI 变化时重测全部 kernel |
| Triton | `3.4.0` | L0 必需 | A/B 和 K=17408 使用；编译参数或数值顺序变化时重测 |
| ROCm/DTK | gfx936；`hipcc` 报告 HIP 6.2、DTK 6.3.3 LLVM 工具链 | L0 native 必需 | 换卡或工具链后 K=5120 必须重新反汇编/数值/延迟验证 |
| vLLM | `0.18.1+das.dtk2604`；宿主基线 `fa718036` | L1/L2 当前宿主 | 允许更换；只重做薄适配层和服务层 |
| AITER | `0.1.dev1+g9daa788.d20260401` | L1 attention fallback | 可换成新 vLLM/SGLang 的等价 decode/fallback backend |
| vLLM vendored FLA | 随 `fa718036` 源码 | L1 GDN host kernel | 没有独立版本；按五个 kernel 与 packed recurrent ABI 迁移 |
| `torch.ops._rocm_C.LLMM1` | 当前 vLLM wheel 内扩展 | L1 首选 provider | 非 L0 必需；可换独立 `qwen35_rocm_opt.native` provider |
| SGLang | 当前未依赖、未导入 | 无 | 迁移时新写 L1/L2，不修改 L0 kernel |

依赖故障采用 fail-closed：精确 gate 不满足就返回宿主官方路径；只有当前运行
明确命中 K=5120 且宿主没有 provider 时才要求预加载独立 native library，不能
静默换成未经验证的 Triton K=5120 实现。

机器可读版本见
[`integration_manifest.json`](closedbook/integration_manifest.json)。三个模块的逐步
闭卷说明分别见 [A](closedbook/PERSON_A_ATTENTION.md)、
[B](closedbook/PERSON_B_GDN.md) 和 [C](closedbook/PERSON_C_GEMV_RUNTIME.md)。

## 5. 合并纪律

1. A/B/C 各自只提交清单内文件；公共 `target.py` 只读。
2. 每人先运行模块自己的独立验证，保存命令、结果和源码 SHA-256。
3. 合并顺序不影响功能；若同文件冲突，只按锚点拼接，不做格式整理。
4. 合并后先运行静态依赖检查，再构建 clean wheel。
5. clean wheel 和 GPU 数值门禁通过后才验收 DP2；两 rank 首次编译后用相同
   参数重启，确认 KV cache 容量一致再计时；交付前在同一热服务补齐全量精度。
6. 任一性能点相对冻结 499 版超过 `0.35%` 的可重复回退，都需要定位；最终
   相对 3k 最优版超过 1% 则不得交付。

## 6. 迁移层级

- L0：`qwen35_rocm_opt` core，仅依赖 PyTorch、Triton、ROCm。
- L1：当前 vLLM adapter，允许依赖上述 vLLM 符号，但不得把 kernel 复制回宿主。
- L2：服务启动、scheduler、DP2 和 KV cache 容量检查，必然依赖宿主框架。

换到新 vLLM 通常只重做 L1；换到 SGLang 时重做 L1 和 L2。L0 只有张量 layout
发生变化时才允许修改，并必须重新完成数值与性能验证。

## 7. 2026-08-05 模块化版本验收

### 7.1 构建、独立性与冻结边界

- 冻结工作树 `pra2026-bh408-repro-800` 的 `git status --short` 为空，模块化
  工作没有改写 499 版。
- full vLLM wheel：
  `modular_validation/dist2/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl`，
  SHA-256 `0c70323f20c36bd4ae1fe835332fe4bf97abb56872540a08352147c8e4077980`。
- 独立 core wheel：
  `modular_validation/standalone-dist2/qwen35_rocm_opt-0.1.0-py3-none-any.whl`，
  SHA-256 `20a20e7b45bb62aedd780ff7fa79f9dc043ace372b7e4bf44620971a92df4708`。
- full wheel 安装到隔离目录后，`qwen35_rocm_opt`、`vllm._rocm_C` 和
  `vllm._C` 均从该目录加载；core-only import 没有加载 vLLM 或 SGLang。
- `scripts/verify_closedbook_modules.py`、`git diff --check` 均通过。静态门禁覆盖
  三模块文件、宿主锚点、禁止 import、profile 5+5 和四份闭卷文档。

### 7.2 GPU 算子门禁

隔离 wheel 与冻结 499 `_rocm_C` 在 gfx936 上复测：page784 attention 逐元素
一致，GDN、K=17408 Triton GEMV 与 K=5120 native provider 数值通过。K=5120
四个输出形状的交错 1000 次、11 轮中位延迟如下：

| M | portable（ms） | frozen 499（ms） | 变化 |
| ---: | ---: | ---: | ---: |
| 96 | 0.010106 | 0.010020 | +0.857% |
| 14336 | 0.099481 | 0.099484 | -0.002% |
| 16384 | 0.113272 | 0.113269 | +0.002% |
| 34816 | 0.237733 | 0.237733 | -0.000% |

四项都低于 1% 门禁；主服务仍优先调用 full wheel 的
`torch.ops._rocm_C.LLMM1`，独立 provider 是迁移/缺失扩展时的等价实现。

### 7.3 双卡启动与 OOM

服务使用 `TP=1, DP=2, backend=mp`、`gpu-memory-utilization=0.95`、
`max_num_batched_tokens=4096`、`compile_sizes=[4096]`。首次 AOT 编译后，以
相同环境和缓存键 `3d032b4893` 热重启；两个 rank 均直接加载 AOT，KV cache
均为 `27,440` token，随后完成 mixed/decode graph capture。

`27,440` 比冻结 499 热服务记录的 `28,224` 少一个 784-token page，但两个
rank 完全对称；这只是当次进程可用显存向下取整后的物理 page 数，不改变
模型 KV 语义。下面 72 个正式性能请求和 110 个精度请求全部成功，服务日志
没有 OOM，证明该容量足够当前验收负载。

### 7.4 双卡九组性能

同一热服务、每组 2 条 warmup 加 8 条正式请求、每请求固定输出 1024 token：

| 并发 | 4–8K tok/s | 8–16K tok/s | 16–32K tok/s | 成功 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 48.211699 | 44.573554 | 41.661494 | 24/24 |
| 4 | 81.615319 | 74.004958 | 63.708134 | 24/24 |
| 8 | 142.836577 | 117.198289 | 95.561931 | 24/24 |

九组共 `72/72`、失败 0、无 OOM。原始 summary SHA-256 为
`9219825dcb56c7e1f0683185824a817c4b907d568c7313f043fee128149da6de`。

权威文档给出的 2656 行/3k 匹配参考为 4–8K `145.244267`、16–32K
`98.068154`；8–16K 匹配参考由冻结 499 的 `115.695862 / 100.360%`
反推为约 `115.280851`。不挑选复跑结果时，首次完整 c8 三档相对参考分别为
`98.342% / 101.663% / 97.444%`，几何平均 `99.134%`，按三档总时长聚合为
`99.071%`：两种总体口径的下降分别约 `0.866%` 和 `0.929%`，都小于 1%。

为排除边缘时钟/系统负载波动，又在同一服务单独复跑两个慢项，4–8K 为
`143.734095 tok/s`，16–32K 为 `96.568554 tok/s`，均 8/8 成功。与首次
8–16K 组成匹配集后，三项相对参考为
`98.960% / 101.663% / 98.471%`，几何平均 `99.688%`，按总时长聚合为
`99.657%`。复跑 summary SHA-256 为
`1e4950e459b71bfbd6a7bfd5e909f051d7bf7ffb423b279c726f5c52f80bc2d5`。

因此模块化版本满足“相对 3k 最优版总体下降不超过 1%”；单项仍应如实保留：
16–32K/c8 的复跑约低 `1.53%`，由 8–16K/c8 的正收益抵消，不能把总体通过
误写成所有单项都低于 1%。

### 7.5 全量精度

官方 `run_accuracy.sh` SHA-256 为
`2e641672a45ac96318c2118df8df4dae2babf87c16afd49cbe4b037ff9beed4e`。
同一双卡热服务的 OpenCompass 结果：

| 数据集 | 请求 | 结果 | 精度系数 |
| --- | ---: | ---: | ---: |
| HotpotQA | 20/20 | 77.96 | 1.00 |
| GovReport | 30/30 | 33.07 | 1.00 |
| Retrieval Multi Point | 30/30 | 100.00 | 1.00 |
| Aggregation Keyword | 30/30 | 100.00 | 1.00 |

总计 `110/110`，四项 `k=1.00`。summary CSV SHA-256 为
`b0dbcd6961dd72e73316799092a01373b588aaccd9b39a2609fabe01afff7a33`，
结果目录为
`repro_minimal_validation/final_full/work/accuracy_debug/output/local_accuracy_qwen35/20260805_153828`。
