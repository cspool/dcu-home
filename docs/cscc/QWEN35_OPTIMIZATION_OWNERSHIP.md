# Qwen3.5-27B 精简优化实现分工

本文给出
`pra2026-bh408-gqa-page784-k5120` 的闭卷实现分工。实现方法遵循
`pra2026-bh408-modular/docs/cscc/CLOSED_BOOK_HIGH_IMPACT_CHEATSHEET.md`：
每项优化只记住并维护四件事——**官方入口、目标 shape、fast path、官方 fallback**。

## 1. 基线与范围

### 1.1 官方 upstream 与实际 OpenDAS 基线

本文所称“官方基线”是比赛提供的 OpenDAS vLLM 源码
`fa718036bdb9dfd80a872b86c8ac16c9d02bfd31`。它声明的包版本是 vLLM 0.18.1，
但它是带 DCU/OpenDAS 修改的 downstream commit，不等同于 upstream vLLM 0.18.1 tag
的 commit。实现时以 `fa718036` 的文件内容为最终基准，upstream URL 用于理解官方设计。

| 项目 | 详细版本 | URL |
| --- | --- | --- |
| vLLM upstream 仓库 | vLLM Team，Apache-2.0 | <https://github.com/vllm-project/vllm> |
| upstream 发布版 | `v0.18.1`，tag commit `a26e8dc7ff2111a005144d775ecf9cebf56c45b2` | <https://github.com/vllm-project/vllm/releases/tag/v0.18.1> |
| upstream 固定源码 | tag `v0.18.1` | <https://github.com/vllm-project/vllm/tree/v0.18.1> |
| upstream 固定文档 | vLLM 0.18.1 | <https://docs.vllm.ai/en/v0.18.1/> |
| 比赛/OpenDAS 基线 | commit `fa718036bdb9dfd80a872b86c8ac16c9d02bfd31`；本地 ref `refs/baselines/opendas-v0.18.1`；提交时间 `2026-04-13T21:02:44+08:00` | <https://gitlab.eduxiji.net/T2026100069912215/pra2026-bh408/-/commit/fa718036bdb9dfd80a872b86c8ac16c9d02bfd31> |
| 当前精简实现 | branch `repro-gqa-page784-k5120`；优化提交 `0e764a1cd6fba986be35edbc64cbd531154e24f4` | <https://github.com/cspool/dcu-home/tree/repro-gqa-page784-k5120> |

OpenDAS `setup.py` 把 Python 包基础版本固定为 `0.18.1`，并按构建环境形成以下 wheel
版本：

```text
默认                 0.18.1+das
ADD_GIT_VERSION=1    0.18.1+das.<7位提交号>
同时设置 ROCM_PATH   0.18.1+das[.<提交号>].dtk2604
```

基线与当前验证环境的详细依赖如下：

| 类别 | 版本/约束 |
| --- | --- |
| Python 声明 | `>=3.10,<3.14`；当前环境 `3.10.12` |
| PyTorch | 基线严格固定 `2.10.0`；当前环境 `2.10.0` |
| torchvision | `0.25.0` |
| flash-attn | OpenDAS要求`2.8.3`；当前环境`2.8.3+das.opt1.dtk2604.torch2100.20260330.g3f0061` |
| Triton | OpenDAS源码构建约束为`3.6.0`；当前比赛容器预装DCU版本`3.4.0+git1ef59765` |
| AITER | 当前比赛容器预装`0.1.dev1+g9daa788.d20260401`；官方attention backend原本已使用 |
| HIP runtime | 当前 `6.3.26093` |
| DTK | 当前 `26.04`，wheel 后缀写作 `dtk2604` |
| HCU target | `gfx936:sramecc+:xnack-` |

page784 使用的公开入口 `flash_attn.varlen_fwd_unified` 来自上述外部 wheel，
最终调用已编译的 `flash_attn_2_cuda`。它是OpenDAS环境声明并预装的DCU版本，不是本仓库
内置源码，也不是三人自行从公开PyPI下载的CUDA wheel。只修改本项目的page784调用不需要
重编FlashAttention。

这里的“官方提供”是指比赛/OpenDAS源码或比赛容器提供，不表示所有组件都由vLLM项目
自行开发。四组优化没有引入新的第三方依赖：

| 使用方 | 依赖 | 来源 | 当前容器是否需要额外下载 |
| --- | --- | --- | --- |
| A：GQA6、GDN | `torch`、`vllm.triton_utils`及其Triton | OpenDAS官方源码入口；PyTorch/Triton由比赛容器预装 | 否 |
| B：page784 | 上述依赖，加`flash_attn`和`flash_attn_2_cuda` | `requirements/rocm.txt`正式声明；比赛容器预装DCU wheel | 否 |
| C：K5120/SwiGLU | PyTorch C++ ABI、HIP/DTK、CMake、Ninja；本仓库`_rocm_C`源码 | OpenDAS仓库和比赛容器 | 否 |
| 公共官方fallback | AITER | OpenDAS官方backend原本已有；比赛容器预装 | 否 |

因此，在当前比赛容器中禁止为本分支另行执行`pip install triton`、
`pip install flash-attn`、`git clone`第三方kernel或下载公开CUDA wheel。Triton源码约束与
容器实装版本不同不是本分支的新依赖；当前算子已用容器预装版本验证，应以比赛镜像为准，
不能由个人替换。若上述import或`hipcc/cmake/ninja`缺失，应判定容器不匹配并恢复官方
环境，而不是自行补包。ROCm/HIP构建只编译仓库已有源码；CMake中的外部FetchContent
分支只在CUDA构建启用，本目标不会下载这些依赖。

读取代码时优先使用精确 baseline object，而不是从当前 HEAD 反推官方实现：

```bash
OFFICIAL=fa718036bdb9dfd80a872b86c8ac16c9d02bfd31
git show "$OFFICIAL":vllm/v1/attention/backends/rocm_aiter_unified_attn.py
git show "$OFFICIAL":vllm/model_executor/models/qwen3_5.py
git show "$OFFICIAL":csrc/rocm/skinny_gemms.cu
```

### 1.2 本文实现范围

- 本文行号对应在优化提交`0e764a1cd6fba986be35edbc64cbd531154e24f4`之上完成的
  attention执行路径拆分；代码变化后仍应优先按函数名定位。
- 目标环境：单卡 gfx936、Qwen3.5-27B、BF16、TP/PP/DP=1。
- 模型配置：hidden size 5120，64 层，Q24/KV4，attention head dim 256，
  GDN K/V head 数为 16/48、head dim 为 128，MLP intermediate size 为 17408。
- 本文只覆盖 GQA6、page784、GDN RMSNorm+SiLU、K5120 GEMV 和
  GateUp/SwiGLU；不覆盖 GDN 固定 schedule、K17408、TunableOp 或 M-RoPE。
- 本版可执行实现 churn 为 626 行；经确认允许相对原 600 行目标超出 26 行。
  `docs/` 和代码注释不计入该口径。代码行号变化后应优先按函数名定位，
  不应机械依赖旧行号。

任何 fast path 只接管已经验证的输入。不匹配时必须返回 `False`/`None`，或从宿主函数
继续执行官方实现，不能让优化条件改变非目标输入的行为。

## 2. 总体责任边界

行数使用 cheatsheet 的 runtime churn 口径：相对 `fa718036` 的 additions 和 deletions
均计入工作量，不是净新增行数；本表排除后来按要求补充的注释。可执行实现是
608 additions + 18 deletions = 626，允许超过原目标 26 行。物理 diff 另含 77 行
调用链/职责/工作逻辑注释，因此不能直接用最终 `git diff --numstat` 反推本表。

| 人员 | 负责代码行数 | 语言 | 行数分解 | 主责 |
| --- | ---: | --- | --- | --- |
| A | 215 | Python；Triton Python DSL | GQA6 完整路径 123；ROCm backend gate/router与AITER适配60；GDN helper 与 Qwen 接入 churn 32 | GQA6；attention 公共入口；GDN RMSNorm+SiLU |
| B | 215 | Python；Triton Python DSL | page784 完整路径 215 | page784 gate、数据准备、两次 FlashAttention 与 LSE merge |
| C | 196 | HIP C++ 125；Python 71 | `skinny_gemms.cu` churn 125；gfx936 公共门/K5120、linear/MLP/setup Python churn 71 | 公共 gfx936 门；K5120；GateUp/SwiGLU；ROCm extension 构建 |
| **合计** | **626** | Python、Triton Python DSL、HIP C++ | 608 additions + 18 deletions；注释另计 77 行但不进入预算 | 本分支全部运行时优化 |

责任排除项：A 不负责 page784 内部实现或 K5120；B 不负责 GQA6、attention 宿主 gate、
GDN epilogue 或 native GEMV；C 不负责 Attention、GDN
Norm+SiLU 或 K17408 down projection。

attention 按执行路径拆成独立文件，不再从函数中间划分负责人：

- `rocm_aiter_unified_attention_gqa6.py`：A 独立负责 GQA6 kernel 与 host launch。
- `rocm_aiter_unified_attention_page784.py`：B 独立负责从 gate、pack 到两次
  FlashAttention 和 LSE merge 的完整 page784 路径。
- `rocm_aiter_unified_attn.py`：A 是 attention 公共入口 owner；先调用 B 的 page784，
  返回 `False` 时再调用 A 的 GQA6，宿主 gate 不命中时继续官方实现。
- `gfx936.py`：C 是文件 owner，负责模块骨架、公共 `is_gfx936` 和 K5120；A 负责其中
  GDN Norm+SiLU 行段。修改共享 import 或公共接口时，A/C 必须互相 review。

page784 仍有可读重构空间，但不能用空白压缩代替分工。保留当前四行/CTA 自定义 merge
时，预计可去除约 15--25 行重复样板；改用官方 `merge_attn_states` 可减少更多代码，
但 launch 结构不同，必须先做相同输入消融。无论是否继续精简，page784 的 pack layout、
两次 attention 和 LSE merge 都由 B 一人维护，避免跨负责人协议和合并冲突。

## 3. 命中的 Qwen layer

Qwen3.5-27B 的 64 个 decoder layer 中：

- `full_attention`：层
  `[3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]`，共 16 层。
- `linear_attention`：其余层，即 `0 <= i < 64 and i % 4 != 3`，共 48 层。
- dense MLP：全部 `0..63` 层都有一组 `Qwen3NextMLP`。

| 优化 | 实际命中的类/权重 | 层范围 | 阶段 |
| --- | --- | --- | --- |
| GQA6 | `Qwen3NextAttention.attn`，Q24/KV4/D256 | 16 个 `full_attention` 层 | 单请求 prefill |
| page784 | 同一 `Qwen3NextAttention.attn` 的 later-prefill；B 负责完整执行路径 | 16 个 `full_attention` 层 | context 已有至少 784 token 的 prefill |
| GDN Norm+SiLU | `Qwen3_5GatedDeltaNet.norm(core_attn_out, z)` | 48 个 `linear_attention` 层 | prefill 和 decode |
| K5120 QKV | `Qwen3NextAttention.qkv_proj`，weight `(14336,5120)` | 16 个 `full_attention` 层 | 单 token decode |
| K5120 GDN QKVZ | `Qwen3_5GatedDeltaNet.in_proj_qkvz`，weight `(16384,5120)` | 48 个 `linear_attention` 层 | 单 token decode |
| K5120 GDN BA | `Qwen3_5GatedDeltaNet.in_proj_ba`，weight `(96,5120)` | 48 个 `linear_attention` 层 | 单 token decode |
| K5120 GateUp/SwiGLU | `Qwen3NextMLP.gate_up_proj`，weight `(34816,5120)` | 全部 64 层 | 单 token decode |
| K5120 LM head | `ParallelLMHead`，weight `(248320,5120)` | decoder 之后一次 | 需要 logits 的 decode step |

这些输出维度直接由模型配置得到：

```text
full-attention QKV+gate = 2*24*256 + 2*4*256 = 14336
GDN QKVZ               = 2*(16*128) + 2*(48*128) = 16384
GDN BA                 = 2*48 = 96
MLP GateUp             = 2*17408 = 34816
LM head                = vocab_size = 248320
```

### 调用链

```text
full_attention layer
  -> Qwen3NextAttention.forward
  -> Attention backend
  -> A: RocmAiterUnifiedAttentionImpl 公共 gate 与路径路由
  -> B: page784 gate、metadata、residual pack、两次 FlashAttention 与 LSE merge
       （命中即返回）
  -> A: GQA6 prefill（page784 不命中时）
  -> 官方 AITER unified_attention（宿主 gate 不命中时）

linear_attention layer
  -> Qwen3_5GatedDeltaNet.forward
  -> 官方 gdn_attention_core
  -> A: qwen35_gdn_rmsnorm
  -> 官方 out_proj

单 token linear/MLP/logits
  -> C: qwen35_k5120_gemv
  -> 官方 _rocm_C.LLMM1 ABI
  -> C: qwen35_gemv_k5120
  -> 不命中时继续官方 GEMM/MLP 路径
```

## 4. A：GQA6、attention 公共入口与 GDN Norm+SiLU

A 的实现范围、官方源码参照、代码索引、专项验收和构建命令已拆至
[A 实现和构建文档](QWEN35_OWNER_A_IMPLEMENTATION_AND_BUILD.md)。

## 5. B：page784 完整执行路径

B 的实现范围、官方源码参照、代码索引、专项验收和构建命令已拆至
[B 实现和构建文档](QWEN35_OWNER_B_IMPLEMENTATION_AND_BUILD.md)。

## 6. C：K5120、GateUp/SwiGLU 与 native 接入

C 的实现范围、官方源码参照、代码索引、专项验收和构建命令已拆至
[C 实现和构建文档](QWEN35_OWNER_C_IMPLEMENTATION_AND_BUILD.md)。

## 7. 构建、少量数据验证与合并

### 7.1--7.4 分工实现与专项构建

原第 7.1 通用构建命令已经复制进每份分工文档；原第 7.2、7.3、7.4 专项命令分别移入
A、B、C 文档，使每位负责人只阅读一份文档即可完成实现、构建和算子验收。

| 负责人 | 独立文档 | 包含的原章节 |
| --- | --- | --- |
| A | [GQA6、attention 路由与 GDN](QWEN35_OWNER_A_IMPLEMENTATION_AND_BUILD.md) | 4、7.1、7.2 |
| B | [page784](QWEN35_OWNER_B_IMPLEMENTATION_AND_BUILD.md) | 5、7.1、7.3 |
| C | [K5120、GateUp/SwiGLU](QWEN35_OWNER_C_IMPLEMENTATION_AND_BUILD.md) | 6、7.1、7.4 |

### 7.5 小数据服务冒烟（DP1/DP2）

先用少量数据验证“新 wheel -> 服务 -> 路由 -> kernel -> HTTP 答案”的闭环。脚本生成
4880-token 文本，chat template 后本轮实际为 4892 tokens；在
`max_num_batched_tokens=4096` 下会形成首个 4096-token prefill 和后续 chunk，从而覆盖
GQA6、page784、GDN，并在单 token decode 覆盖 K5120/GateUp/SwiGLU。该步骤不是吞吐测试。

先在任意终端设置共同路径；`SITE_DIR` 必须指向对应分工文档 2.1 刚安装的新 wheel：

```bash
REPO_ROOT=/public/home/tangyu408/Qwen_DCU_Worker_0/pra2026-bh408-gqa-page784-k5120
MODEL_DIR=/root/Qwen3.5-27B
SMOKE_SCRIPT="$REPO_ROOT/docs/cscc/smoke_qwen35_service.py"
```

DP1 服务终端：

```bash
DP1_ROOT="$ARTIFACT_ROOT/service-dp1-eager"
mkdir -p "$DP1_ROOT"/{triton,inductor,vllm,pycache}
RUN_DIR="$(mktemp -d /tmp/qwen35-dp1-run.XXXXXX)"
cd "$RUN_DIR"
export HIP_VISIBLE_DEVICES=0
export PYTHONPATH="$SITE_DIR"
export TRITON_CACHE_DIR="$DP1_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$DP1_ROOT/inductor"
export VLLM_CACHE_ROOT="$DP1_ROOT/vllm"
export PYTHONPYCACHEPREFIX="$DP1_ROOT/pycache"
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
python3 -m vllm.entrypoints.cli.main serve "$MODEL_DIR" \
  --served-model-name Qwen3.5-27B --port 18011 --trust-remote-code \
  --dtype bfloat16 --tensor-parallel-size 1 --block-size 784 \
  --max-num-seqs 128 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.95 --enforce-eager \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --reasoning-parser qwen3 --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder 2>&1 | tee "$DP1_ROOT/server.log"
```

DP1 请求终端只发 1 条数据：

```bash
python3 "$SMOKE_SCRIPT" \
  --base-url http://127.0.0.1:18011 --model Qwen3.5-27B \
  --tokenizer "$MODEL_DIR" --requests 1 --target-prompt-tokens 4880 \
  --output "$DP1_ROOT/smoke.json"
```

JSON 中 `contains_expected_code` 为 `true` 后，在服务终端按 `Ctrl-C` 正常停止。确认进程
退出后再启动 DP2。

DP2 服务终端：

```bash
DP2_ROOT="$ARTIFACT_ROOT/service-dp2-eager"
mkdir -p "$DP2_ROOT"/{triton,inductor,vllm,pycache}
RUN_DIR="$(mktemp -d /tmp/qwen35-dp2-run.XXXXXX)"
cd "$RUN_DIR"
export HIP_VISIBLE_DEVICES=0,1
export PYTHONPATH="$SITE_DIR"
export TRITON_CACHE_DIR="$DP2_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$DP2_ROOT/inductor"
export VLLM_CACHE_ROOT="$DP2_ROOT/vllm"
export PYTHONPYCACHEPREFIX="$DP2_ROOT/pycache"
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
python3 -m vllm.entrypoints.cli.main serve "$MODEL_DIR" \
  --served-model-name Qwen3.5-27B --port 18012 --trust-remote-code \
  --dtype bfloat16 --tensor-parallel-size 1 --block-size 784 \
  --data-parallel-size 2 --data-parallel-backend mp \
  --max-num-seqs 128 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.95 --enforce-eager \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --reasoning-parser qwen3 --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder 2>&1 | tee "$DP2_ROOT/server.log"
```

DP2 请求终端并发 2 条数据，分别验证两个副本：

```bash
python3 "$SMOKE_SCRIPT" \
  --base-url http://127.0.0.1:18012 --model Qwen3.5-27B \
  --tokenizer "$MODEL_DIR" --requests 2 --target-prompt-tokens 4880 \
  --output "$DP2_ROOT/smoke.json"
```

这里的 `--enforce-eager` 是 vLLM 官方功能开关，只用于少量数据的功能冒烟；目标
Triton/HIP kernel 仍真实执行和 JIT，但本轮耗时不能作为性能结果。正式吞吐测试必须去掉
该开关、完成冷/热缓存，并覆盖 4--8K、8--16K、16--32K 全数据。在任何 OOM、worker
重启、异常 fallback 或重复 JIT 消除前，不得开始正式计时。

### 7.6 2026-08-10 实测记录

本轮没有下载或替换依赖，直接使用比赛容器中的官方配套包：

| 组件 | 实测版本/入口 |
| --- | --- |
| vLLM | `0.18.1+das.dtk2604` |
| PyTorch | `2.10.0+das.opt1.dtk2604.20260325.g6b060a` |
| Triton | `3.4.0+git1ef59765` |
| FlashAttention | `2.8.3+das.opt1.dtk2604.torch2100.20260330.g3f0061`，`flash_attn.flash_attn_interface.varlen_fwd_unified` |
| AITER | `0.1.dev1+g9daa788.d20260401` |
| 设备 | `gfx936:sramecc+:xnack-` |

新构建和 native extension 身份：

| 项目 | 实测值 |
| --- | --- |
| wheel | `/tmp/qwen35-focused-wheel-20260810-r7/dist/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl` |
| wheel SHA256 | `ecc3bf7b86c74d550e1f9a9c80dde10665a9ac13f2bcd561914d6a8be83f8c10` |
| 隔离导入 | `/tmp/qwen35-focused-wheel-20260810-r7/site/vllm/__init__.py` |
| 新 `_rocm_C` | `/tmp/qwen35-focused-wheel-20260810-r7/site/vllm/_rocm_C.abi3.so` |
| `_rocm_C` SHA256 | `4226920e3de4ec0188de61b0b54d28b6c55225755283868189c9e2c249df5fe1` |

从源码树外导入该隔离 site，确认 `vllm` 和 `_rocm_C` 都来自上述 r7 目录；wheel 内所有
相关 Python 文件也与本工作树逐字节一致。`verify_qwen35_optimizations.py all` 使用全新
Triton cache 运行，schema 为 `qwen35-focused-small-matrix-v3`，少量算子矩阵全部通过：

| 优化 | 覆盖 | 最大绝对误差 | 结果 |
| --- | --- | ---: | --- |
| GQA6 | 4组普通cache和1组交错cache | 0.015625；交错cache为0.00390625 | pass |
| page784 | context=`784/800`，含1组交错cache | 0.00390625 | pass |
| page784 fallback | 4 个边界，均返回 `False` 且输出不变 | 0 | pass |
| GDN Norm+SiLU | T=`16/32/64/128/4096` | 0 | pass |
| GDN fallback | 非目标 stride | 0 | pass |
| K5120 | M=`96/14336/16384/34816/248320` | 0.0078125 | pass |
| GateUp/SwiGLU | 输出维度 17408 | 0.0625 | pass |
| K5120 fallback | dtype/stride/shape/bias/expert gate 矩阵 | 0 | pass |

服务级少量数据结果：

| 模式 | 请求 | 实际 prompt / completion | 答案 | HTTP墙钟 | 结果 |
| --- | ---: | --- | --- | ---: | --- |
| DP1 | 1 | 4892 / 8 | `9342002` | 129.678 s | HTTP 200，精确匹配 |
| DP2 | 2 并发 | 每条 4892 / 8 | 两条均为 `9342002` | 15.753 s | ApiServer 0/1 各完成一条，精确匹配 |

两次服务都显式设置目标环境变量和`--block-size 784 --enforce-eager`，日志确认选择
`ROCM_AITER_UNIFIED_ATTN`。DP1使用全新cache冷JIT，engine初始化约192.31 s；请求完成后
生成了`_gqa6_prefill`、`_pack_page784`、`_merge_page784`三个kernel cache文件。DP2为减少
重复JIT，将已验证的DP1 Triton cache复制为起点，约38.7 s完成初始化；因此上表墙钟只证明
功能闭环，不能比较DP1/DP2性能。DP1/DP2日志扫描均未发现traceback、OOM、worker重启或
EngineCore failure，并已在请求完成后正常停止。结果保存在：

- `/tmp/qwen35-focused-wheel-20260810-r7/operator-validation.log`
- `/tmp/qwen35-focused-wheel-20260810-r7/service-dp1/{server.log,smoke.json}`
- `/tmp/qwen35-focused-wheel-20260810-r7/service-dp2/{server.log,smoke.json}`
- `/tmp/qwen35-focused-wheel-20260810-r7/service-dp1/triton/`中的三个目标kernel cache证据

额外停止线：本轮也尝试了 `compile_sizes=[4096]` 的非 eager 启动。两段图成功编译，
但官方 runner 的 M-RoPE position buffer 按设计具有 `(4097,1)` 非连续 stride，而本环境
Torch AOT 图把它固化为 `(4096,1)`，在 profile 阶段触发 stride assertion。该问题位于
官方 `vllm/v1/worker/gpu_model_runner.py` 的 M-RoPE buffer/compile 交界，不是上述算子
数值失败。因此本节只确认官方 eager 功能闭环；在这个 AOT stride 问题解决并重新验证前，
不能把本轮结果写成正式性能数据或“DP2 全量完成”。

### 7.7 合并顺序

建议按下面顺序集成，便于定位回归：

1. C 先提供公共 gfx936 设备门和 `gfx936.py` 模块骨架；A/B 只消费该接口。
2. B 独立完成 page784 的 `gate -> pack -> two FA -> merge -> bool` 路径。
3. A 接 GQA6 和 attention 宿主 gate/router，并保持
   `page784(True) -> return`、`page784(False) -> GQA6`、宿主 gate 不命中走 official。
4. C 启用 `_rocm_C`，接普通 K5120，再单独接 fused GateUp/SwiGLU。
5. A 最后接 GDN Norm+SiLU；它不应改变 GDN core/state，也不应与 C 的输入投影冲突。
6. 每合一项先做 same-input 算子正确性，再使用全新 wheel、`.so` 和编译缓存做服务测试。

联合完成标准：

- `git diff --check`、Ruff、Python compile 和改动行的 clang-format 通过。
- 可执行实现churn相对`fa718036`为626（经允许超过原600目标26行）；77行代码注释和
  `docs/`不计入预算。禁止通过压缩空白字符降低该数字。
- 4-8K、8-16K、16-32K 固定请求精度全部通过，再执行 DP2 全量测试。
- decode 验证必须同时覆盖 full-attention 层、linear-attention 层、全部 MLP 和 LM head，
  不能只证明某一个 `(M,5120)` microbenchmark 成功。
- 任一 shape、stride、dtype 或功能开关不满足时，结果必须来自官方 fallback。

### 7.8 备注：可能出问题的位置与验收检查

下面的内容用于失败定位和最终验收，不要混入各分工文档 2.1 的主构建命令。

- **临时目录已存在**：`git worktree add` 或 wheel 安装会拒绝复用旧目录。确认没有服务或
  构建进程使用它后，清理旧 worktree，或给 `/tmp/qwen35-build-source` 和
  `/tmp/qwen35-build` 加新的后缀。
- **改动没有进入 wheel**：临时 worktree 只包含 `HEAD`；构建前应先提交要验证的改动。
  如果 wheel 中的 Python 文件与开发目录不同，优先检查这一项。
- **误从开发目录导入**：在仓库目录执行 Python 时，当前目录会遮蔽 `PYTHONPATH`。应从
  `/tmp` 等源码树外运行；`vllm.__file__` 和 `vllm._rocm_C.__file__` 都必须位于同一个
  `/tmp/qwen35-build/site`。
- **复用了旧 native 库**：C 修改后必须重新构建 wheel。记录 wheel 和
  `site/vllm/_rocm_C.abi3.so` 的 SHA256；K5120 测试日志也会打印实际 `.so` 路径和哈希。
- **依赖或编译器缺失**：当前实测依赖为 PyTorch
  `2.10.0+das.opt1.dtk2604.20260325.g6b060a`、Triton `3.4.0+git1ef59765`、
  FlashAttention `2.8.3+das.opt1.dtk2604.torch2100.20260330.g3f0061`、AITER
  `0.1.dev1+g9daa788.d20260401`；编译器使用容器中的 `hipcc`、CMake 和 Ninja。不要执行
  `pip install` 下载替代包。
- **服务未命中目标 attention**：必须同时设置
  `VLLM_ROCM_USE_AITER=1`、`VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1`，并传入
  `--block-size 784`。日志应出现 `Using ROCM_AITER_UNIFIED_ATTN attention backend`；
  缺少环境变量会回到默认 backend，缺少 page size 会在 hybrid KV cache 初始化时报错。
- **连续cache假通过**：比赛hybrid KV cache可能是交错view；只用`torch.empty`直接创建
  `(pages,784,4,256)`会掩盖地址错误。GQA6/page784专项必须额外从
  `(pages,2,784,4,256)`切出K/V，并与官方FA比较；当前记录的stride是
  `(1605632,1024,256,1)`。
- **答案正确但fast path未命中**：HTTP答案可能来自官方fallback。目标服务请求完成后，
  除日志中的backend名称外，还要在本轮`TRITON_CACHE_DIR`确认新生成
  `_gqa6_prefill`、`_pack_page784`和`_merge_page784`缓存文件。
- **AITER 函数签名不一致**：本容器官方 AITER 的 `unified_attention` 不接收 `sinks` 和
  `output_scale`。本目标版本声明不支持这两个融合能力，并且 Qwen3.5-27B 两项实际均为
  `None`；如果换成 sink attention 或输出量化模型，应改走支持这些参数的官方 backend。
- **小数据通过但不能计时**：`--enforce-eager` 仅用于功能闭环。冷 JIT、启动 profile 和
  HTTP 墙钟都不是吞吐数据；正式性能测试需去掉该开关并完成冷/热缓存区分。
- **非 eager 在 profile 阶段失败**：本环境曾因官方 M-RoPE position buffer 的
  `(4097,1)` stride 与 AOT 图固化的 `(4096,1)` 不一致而停止。这是正式吞吐测试的停止线，
  不能把 eager 冒烟写成“DP2 全量完成”。
- **最终检查**：依次确认 `git diff --check`、Ruff、Python compile、C++ 格式、
  `verify_qwen35_optimizations.py all`（含交错cache）、DP1一条请求、DP2两条并发请求，
  并检查三个attention kernel的命中证据；服务日志不得出现traceback、OOM、worker重启
  或EngineCore failure。
