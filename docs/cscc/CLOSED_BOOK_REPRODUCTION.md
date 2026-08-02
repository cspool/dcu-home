# 闭卷复现手册

本页是当前分支的权威操作入口。只依赖仓库、平台镜像、官方模型和官方评测
数据，不依赖开发期 goal、trace 或聊天记录。

## 1. 身份与前提

```text
branch: repro-minimal
direct code baseline: OpenDAS vllm_cscc
baseline commit: fa718036bdb9dfd80a872b86c8ac16c9d02bfd31
model: Qwen3.5-27B BF16
primary device topology: one gfx936
primary parallelism: TP=1, PP=1, DP=1
optional multi-request topology: two local gfx936, TP=1, PP=1, DP=2, mp
```

仓库历史以精简提交快照 `67f44ab405d8efed30a42f04f6e74ae2e8370884`
为根；OpenDAS commit 是树比较对象，不是该精简历史的 Git ancestor。

要求平台预装 `BUILD.md` 中的 DTK/PyTorch/Triton/AITER 组合，并提供官方
模型、tokenizer 和评测数据。不要联网升级 Python 依赖。

## 2. checkout 后静态检查

```bash
git status --short
git rev-parse HEAD
bash scripts/verify_cscc_repro.sh
```

最后一行必须是 `verify_cscc_repro: PASS`。若 OpenDAS comparison object 未
随 clone 提供，脚本会注明改用提交快照验证；这不影响当前完整源码构建。

## 3. 从空 build tree 生成 wheel

```bash
mkdir -p /short/public/path/tmp /path/to/repro-output/dist
export TMPDIR=/short/public/path/tmp

DIST_DIR=/path/to/repro-output/dist \
MAX_JOBS=16 \
bash scripts/build_cscc_wheel.sh 2>&1 | tee /path/to/repro-output/build.log

WHEEL="$(find /path/to/repro-output/dist -maxdepth 1 \
  -name 'vllm-*.whl' -print -quit)"
bash scripts/verify_cscc_repro.sh "$WHEEL"
sha256sum "$WHEEL"
```

不要复用仓库旧 `build/`。脚本默认使用 `mktemp` 创建空 build tree；需要保留
native build 供调试时，传入一个尚不存在的 `BUILD_BASE`。

## 4. 安装与 native ABI 检查

```bash
python3 -m pip install --force-reinstall --no-deps "$WHEEL"

python3 - <<'PY'
import torch
import vllm
import vllm._rocm_C

assert vllm.__version__ == "0.18.1"
assert hasattr(torch.ops._rocm_C, "qwen35_bf16_gemv")
assert not hasattr(torch.ops._rocm_C, "LLMM1Strided")
assert not hasattr(torch.ops._rocm_C, "LLMM1StridedSilu")
print("native ABI: PASS")
PY
```

## 5. 前台启动服务

首次复现建议前台运行，直接观察编译进度：

```bash
source scripts/cscc_gfx936_env.sh
export HIP_VISIBLE_DEVICES=0
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"

vllm serve /path/to/Qwen3.5-27B \
  --served-model-name Qwen3.5-27B \
  --port 8001 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.95 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

另一个终端等待健康检查：

```bash
until curl -fsS --noproxy '*' http://127.0.0.1:8001/health >/dev/null; do
  sleep 5
done
echo ready
```

首次启动需要生成 Triton/TorchInductor/AOT cache。不要在 `/health` 成功前跑
性能脚本，也不要把冷编译时间计为请求 TTFT。

### 5.1 可选双卡 DP=2 服务

多请求测试可以启动两个完整的 TP=1 副本：

```bash
MODEL_DIR=/path/to/Qwen3.5-27B \
HIP_VISIBLE_DEVICES=0,1 \
bash scripts/serve_cscc_dp2.sh 2>&1 | tee service-dp2.log
```

首次启动会分别生成两个 DP rank 的 compile/AOT cache。健康检查成功后用完全
相同的 wheel、cache 和参数重启；只有日志直接加载 compiled graph、AOT 分别
命中 `rank_0_0`/`rank_0_1` cache，两个 rank 均报告
`GPU KV cache size: 28,224 tokens`，并通过 TunableOp INIT/PRE_CAPTURE 验收后
才开始计时。完整 benchmark 和实测见
[DP2_MULTI_REQUEST.md](DP2_MULTI_REQUEST.md)。

## 6. 启动日志验收

单卡主路径日志必须同时满足：

```text
Resolved architecture: Qwen3_5ForConditionalGeneration
speculative_config=None
tensor_parallel_size=1
data_parallel_size=1
Setting attention block size to 784 tokens
Using the validated 4096-token static compile shape
VLLM_ROCM_TUNABLEOP_INIT status=ready
VLLM_ROCM_TUNABLEOP_PRE_CAPTURE status=ready
page784 later-Prefill wrapper enabled
```

DP2 外层配置为 `data_parallel_size=2`；dense worker 内部会规范化为 DP1，
TunableOp 日志仍必须保留 `parent_dp=2`、`parent_local_dp=2` 和各自 rank，避免
把错误拓扑误认作单卡。

出现 `VLLM_ROCM_TUNABLEOP_* status=error`、profile SHA mismatch、旧 GEMV
symbol 缺失或 engine process 退出时，停止评测并修复环境；不要静默禁用 profile
后继续计分。

## 7. 固定评测

进入组委会提供的 testdata 目录，保持脚本逐字节不变：

```bash
MODEL_DIR=/path/to/Qwen3.5-27B bash run_throughput.sh all \
  2>&1 | tee performance.log
MODEL_DIR=/path/to/Qwen3.5-27B bash run_accuracy.sh all \
  2>&1 | tee accuracy.log
```

性能契约为三档各 50 条、单并发、request-rate=1、temperature=0、1024 最大
输出 token、2 条 warmup。精度应实际完成 20+30+30+30=110 条；GovReport
文件末尾可能没有换行，`wc -l=29` 不表示少测一条。

验收至少检查：

- 每档 `completed=50`、`failed=0`；
- TTFT P99 和全局逐请求 TPOT P99 低于官方门限；
- HotpotQA、GovReport、Retrieval、Aggregation 的精度系数均为 1；
- 服务日志在评测期间无 traceback、engine death 或 profile error。

## 8. 常见故障

| 现象 | 原因与处理 |
| --- | --- |
| wheel 含大量 `.pyc` | 复用了旧 build tree；只用 `build_cscc_wheel.sh` 的空 `BUILD_BASE` |
| HIP 编译长时间停在少数目标 | 干净构建会为多个 arch 编译；先看子进程 CPU，再检查 `TMPDIR` 空间 |
| `/tmp` 100% | 把 `TMPDIR` 指到公共盘，不删除来源不明的系统临时目录 |
| ZMQ IPC path 超过 107 字符 | `TMPDIR` 路径过深；改用公共盘上的短目录 |
| 首次服务启动很慢 | 冷 Triton/Inductor/AOT 编译；等待 `/health`，后续可复用相同源码缓存 |
| 单条 TTFT 多出约 30 秒 | 检查是否有漏预热编译；保留原结果，待服务稳态后用同一完整档复测 |
| `watch` 无输出 | 保证整个命令在同一对引号内；优先 `tail -F service.log` |
| profile 被拒绝 | 核对 `cscc_gfx936_env.sh`、工具链 validator、单卡和 4096 配置 |
| DP2 两侧 KV cache 容量不同 | 首次双 rank 编译造成瞬态显存不对称；待 cache 生成完毕后停止并以相同参数重启 |

## 9. 重构可追溯性

核心瘦身提交：

```text
0557e7f  remove inactive experiment paths
a2e80e9  compact frozen TunableOp loader
e82db99  keep generated version metadata untracked
621fde4  specialize the frozen Qwen3.5 GEMV op
```

这些提交没有改动原最高性能工作树。当前分支的 source manifest、clean wheel
哈希和最终全量结果见 [RESULTS.md](RESULTS.md) 与
`evidence/manifests/`。
