# 源码构建、启动与冷热缓存简明流程

本文是单卡和可选 DP=2 的统一操作入口，适用于 gfx936、Qwen3.5-27B BF16。
核心原则只有两条：**源码变化必须重建 wheel 并使用新缓存目录；正式运行必须在一次
冷编译完成后，用完全相同的 wheel、参数和缓存目录热重启。**

冷启动用于生成当前源码对应的 Triton、TorchInductor 和 AOT 编译缓存，首次 JIT
耗时不能计入性能。热重启命中这些编译缓存，才具有稳定、可比较的运行条件。这里
复用的是编译产物，不是 prompt、KV、答案或模型中间结果；prefix cache 仍然关闭。

## 1. 准备路径

在仓库根目录执行，将示例路径替换为实际路径：

```bash
cd /home/rog/dcu-home

export MODEL_DIR=/path/to/Qwen3.5-27B
export RUN_ID="$(git rev-parse --short=12 HEAD)-r1"
export ARTIFACT_ROOT="/path/to/artifacts/$RUN_ID"
export CACHE_ROOT="/path/to/compile-cache/$RUN_ID"
export TMPDIR=/path/to/build-scratch

test -f "$MODEL_DIR/config.json"
test ! -e "$ARTIFACT_ROOT"
test ! -e "$CACHE_ROOT"
mkdir -p "$ARTIFACT_ROOT/dist" "$CACHE_ROOT"

git rev-parse HEAD >"$ARTIFACT_ROOT/source-commit.txt"
git status --short >"$ARTIFACT_ROOT/source-status.txt"
git diff --check
```

源码再次修改后，把 `r1` 改成 `r2` 或使用新的提交 SHA，并创建新的
`ARTIFACT_ROOT`、`CACHE_ROOT`。不要让新源码复用旧 revision 的编译缓存。

## 2. 从源码构建 wheel

评测环境应使用已配套的 PyTorch、Triton、AITER 和 DTK，不要从公开 PyPI 覆盖它们。

```bash
DIST_DIR="$ARTIFACT_ROOT/dist" \
MAX_JOBS=16 \
bash scripts/build_cscc_wheel.sh

mapfile -t WHEELS < <(
  find "$ARTIFACT_ROOT/dist" -maxdepth 1 -type f -name 'vllm-*.whl' -print
)
test "${#WHEELS[@]}" -eq 1
export WHEEL="${WHEELS[0]}"

sha256sum "$WHEEL" | tee "$ARTIFACT_ROOT/wheel.sha256"
bash scripts/verify_cscc_repro.sh "$WHEEL"
```

构建脚本使用空的临时 build tree，避免仓库内旧 `.so` 混入 wheel。修改 `csrc/`、
Triton/Python 运行路径、profile CSV 或 `setup.py` 后，都应重新生成 wheel。

## 3. 安装并确认没有导入源码树

```bash
python3 -m pip install --force-reinstall --no-deps "$WHEEL"

export SOURCE_ROOT="$(pwd -P)"
VERIFY_CWD="$(mktemp -d /tmp/vllm-wheel-verify.XXXXXX)"
(
  cd "$VERIFY_CWD"
  PYTHONPATH= python3 - <<'PY'
import os
from pathlib import Path

import vllm
import vllm._rocm_C

source = Path(os.environ["SOURCE_ROOT"]).resolve() / "vllm"
python_file = Path(vllm.__file__).resolve()
native_file = Path(vllm._rocm_C.__file__).resolve()
print(f"vllm={python_file}")
print(f"_rocm_C={native_file}")
assert source not in python_file.parents
assert source not in native_file.parents
PY
)
rmdir "$VERIFY_CWD"
```

必须从源码树外且清空 `PYTHONPATH` 验证，否则可能出现“新 Python + 旧 native
extension”或直接导入工作树的假安装。

## 4. 设置当前版本的独立缓存

冷启动和热重启都使用同一组变量：

```bash
source scripts/cscc_gfx936_env.sh

export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
export PYTHONPYCACHEPREFIX="$CACHE_ROOT/pycache"
```

不要清空系统共享缓存，也不要执行系统级 `drop_caches`。解决冷热缓存问题的方法是：

1. 每个源码版本使用一个全新的 `CACHE_ROOT`；
2. 用该目录完成一次冷启动和目标 shape warmup；
3. 正常停止全部 worker；
4. 用相同 wheel、环境、参数和 `CACHE_ROOT` 热重启；
5. 若热启动仍反复 JIT，先检查缓存变量、写权限和导入路径，不要记录该轮性能。

## 5. 单卡启动：TP/PP/DP=1

建议从源码树外启动，避免当前目录优先导入 `vllm/`：

```bash
export HIP_VISIBLE_DEVICES=0
export PORT=8001
RUN_CWD="$(mktemp -d /tmp/vllm-run.XXXXXX)"
cd "$RUN_CWD"

vllm serve "$MODEL_DIR" \
  --served-model-name Qwen3.5-27B \
  --port "$PORT" \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.95 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  2>&1 | tee "$ARTIFACT_ROOT/cold-single.log"
```

这是第一次冷启动。等待模型 warmup/JIT 完成和健康检查成功：

```bash
curl --fail "http://127.0.0.1:$PORT/health"
```

再用正式输入范围各发少量非计时请求，使 4--8K、8--16K、16--32K 会用到的 shape
完成编译。随后在启动终端按 `Ctrl-C` 正常停止服务，确认 worker 全部退出。

热重启时原命令完全不变，只把日志名改为：

```text
$ARTIFACT_ROOT/hot-single.log
```

不要增加量化、speculative decoding 或 prefix-caching 参数。单卡模式在精确门下会
自动使用 `compile_sizes=[4096]`；日志中应确认 BF16、TP/PP/DP=1、
`quantization=None`、`speculative_config=None` 和 prefix cache 关闭。

## 6. 可选双卡启动：TP=1、DP=2

DP=2 是两个完整的 TP=1 模型副本，由 MP data-parallel frontend 分发请求；它不是
把单卡 kernel 改成 TP kernel。该配置用于多卡准备，不应与单卡成绩混用。

回到仓库根目录并使用仓库脚本启动：

```bash
cd /home/rog/dcu-home

MODEL_DIR="$MODEL_DIR" \
HIP_VISIBLE_DEVICES=0,1 \
PORT=8001 \
bash scripts/serve_cscc_dp2.sh \
  2>&1 | tee "$ARTIFACT_ROOT/cold-dp2.log"
```

脚本固定以下关键参数：

```text
BF16, TP=1, DP=2, backend=mp
max_num_seqs=128
max_num_batched_tokens=4096
compile_sizes=[4096]
gpu_memory_utilization=0.95
```

DP=2 必须显式给出 `compile_sizes=[4096]`，因为源码中的单卡自动门只接管
world/DP=1。第一次启动会为两个 rank 分别生成缓存；等待两侧编译完成、`/health`
成功，并确认两个 rank 报告一致且足够的 KV cache token capacity。正常停止后，用
完全相同的命令和 `CACHE_ROOT` 热重启，只把日志改为：

```text
$ARTIFACT_ROOT/hot-dp2.log
```

若两个 rank 的 KV capacity 不一致、其中一侧仍在 JIT、出现 OOM/fallback/worker
重启，应停止并检查两侧缓存是否均可写且均命中，不能直接开始正式请求。

## 7. 每次修改后的最短闭环

```text
停止旧服务
  -> 新 RUN_ID、空 dist、空 CACHE_ROOT
  -> 构建并安装新 wheel
  -> 源码树外确认 Python 与 _rocm_C 导入路径
  -> 冷启动并覆盖目标 shape
  -> 正常停止
  -> 相同 wheel/参数/cache 热重启
  -> 再进行性能或精度工作
```

至少保存 commit、工作树状态、wheel SHA256、实际导入路径、缓存目录、冷/热启动
命令和日志。冷启动耗时与首请求耗时不作为正式性能结果。
