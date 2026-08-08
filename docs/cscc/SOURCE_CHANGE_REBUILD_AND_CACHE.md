# 源码修改后的构建、安装与冷热缓存复测

本文规定修改 `csrc/`、`setup.py` 或 `vllm/` 后，如何在 DCU 0 上确认实际运行的
是新代码，并以一致的冷热缓存条件重新测试。它补充 [BUILD.md](../../BUILD.md) 的
干净构建步骤，不替代全量性能与精度验收。

正式性能数据必须来自新构建并安装的 wheel。直接从源码目录运行只适合定位问题，
不能作为可提交或可比较的性能结果。

## 1. 哪些改动需要失效

| 修改类型 | 新 wheel | 新服务进程 | 新编译缓存命名空间 |
| --- | --- | --- | --- |
| `csrc/`、native binding、`setup.py` | 必须 | 必须 | 必须 |
| Triton kernel、模型或 attention/GDN Python 路径 | 正式测试必须 | 必须 | 必须 |
| TunableOp profile CSV 或打包规则 | 必须 | 必须 | 必须 |
| 影响 shape、dtype、并行拓扑或 kernel gate 的启动参数 | wheel 可不变 | 必须 | 必须 |
| 只修改文档 | 不需要 | 不需要 | 不需要 |

即使只改 Python，旧 worker 仍持有已经导入的模块和已编译图，因此不能在旧服务上
继续计时。为了让 Python、Triton 和 native extension 来自同一份交付物，正式测试也
统一重新生成 wheel。

## 2. 为本次源码建立唯一身份

正式全量测试应使用已提交且工作树干净的 revision。小样本调试允许工作树有改动，
但必须为每次改动手工增加新的 `RUN_ID`，且不得把这类结果写成正式数据。

```bash
git status --short
git diff --check

RUN_ID="$(git rev-parse --short=12 HEAD)-r1"
ARTIFACT_ROOT="/path/to/artifacts/${RUN_ID}"
CACHE_ROOT="/path/to/compile-cache/${RUN_ID}"

test ! -e "$ARTIFACT_ROOT"
test ! -e "$CACHE_ROOT"
mkdir -p "$ARTIFACT_ROOT/dist" "$CACHE_ROOT"

git rev-parse HEAD >"$ARTIFACT_ROOT/source-commit.txt"
git status --short >"$ARTIFACT_ROOT/source-status.txt"
git diff --binary fa718036bdb9dfd80a872b86c8ac16c9d02bfd31 -- \
  csrc setup.py vllm >"$ARTIFACT_ROOT/official-runtime.patch"
```

`RUN_ID` 的 `r1` 是本次构建序号；源码再次修改时改为新的序号，不能继续使用原来的
缓存目录。不要通过清空共享缓存根目录制造“冷环境”，只创建并使用本次测试独占的
新目录。

## 3. 停止旧服务

安装新 wheel 前，先正常停止自己启动的 vLLM 服务并等待 worker 全部退出。不要用
宽泛的 `pkill` 影响同机其他任务。安装后也不能恢复旧进程：已经运行的 Python
进程不会因 site-packages 变化自动加载新源码或新 `.so`。

同时记录旧服务的启动命令、环境和 PID，确认正式测试期间只有 DCU 0 被当前服务
使用。单卡流程始终设置：

```bash
export HIP_VISIBLE_DEVICES=0
```

## 4. 从空 build tree 生成并验收 wheel

`scripts/build_cscc_wheel.sh` 每次创建新的 build tree，不复用仓库内的 `build/`、
`dist/` 或旧 native binary。输出目录也应是本次 `RUN_ID` 的空目录：

```bash
TMPDIR=/path/to/build-scratch \
DIST_DIR="$ARTIFACT_ROOT/dist" \
MAX_JOBS=16 \
bash scripts/build_cscc_wheel.sh

mapfile -t WHEELS < <(
  find "$ARTIFACT_ROOT/dist" -maxdepth 1 -type f -name 'vllm-*.whl' -print
)
test "${#WHEELS[@]}" -eq 1
WHEEL="${WHEELS[0]}"

sha256sum "$WHEEL" | tee "$ARTIFACT_ROOT/wheel.sha256"
bash scripts/verify_cscc_repro.sh "$WHEEL"
```

如果修改了 `csrc/` 却没有看到 native extension 重新构建，或 wheel 中 `_rocm_C`
时间戳/哈希没有变化，应停止测试并检查构建输入，不能用服务可启动来替代这项确认。

## 5. 强制安装并验证实际导入路径

```bash
python3 -m pip install --force-reinstall --no-deps "$WHEEL"

SOURCE_ROOT="$(pwd -P)"
export SOURCE_ROOT
VERIFY_CWD="$(mktemp -d /tmp/vllm-wheel-verify.XXXXXX)"
(
  cd "$VERIFY_CWD"
  PYTHONPATH= python3 - <<'PY'
import os
from pathlib import Path

import vllm
import vllm._rocm_C

python_file = Path(vllm.__file__).resolve()
native_file = Path(vllm._rocm_C.__file__).resolve()
source_package = Path(os.environ["SOURCE_ROOT"]).resolve() / "vllm"
print(f"vllm={python_file}")
print(f"_rocm_C={native_file}")
assert python_file.is_file()
assert native_file.is_file()
assert source_package not in python_file.parents
assert source_package not in native_file.parents
PY
)
rmdir "$VERIFY_CWD"
```

验证必须从源码树之外执行，并清空 `PYTHONPATH`。否则当前目录下的 `vllm/` 可能
遮蔽已安装 wheel，形成“新 Python + 旧 `_rocm_C`”或相反的混合运行。保存上述
两个绝对路径、wheel SHA256 和 `python3 -m pip show vllm` 输出作为本次证据。

正式服务也应从源码树之外的空运行目录启动，并使用与上述安装命令相同的 Python
环境。

## 6. 三类缓存的边界

| 缓存 | 本流程如何处理 | 是否能跨源码版本复用 |
| --- | --- | --- |
| CMake/Ninja/wheel build tree | 每次从空目录构建 | 不能 |
| Triton、TorchInductor、AOT 编译缓存 | 同一 wheel 先冷编译，再供热重启使用 | 不能 |
| KV、prefix、prompt/答案和模型中间结果 | 不持久化；prefix cache 保持关闭 | 不能跨请求复用 |

为本次源码显式设置独立缓存根目录：

```bash
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
export PYTHONPYCACHEPREFIX="$CACHE_ROOT/pycache"
```

vLLM 可能在 `VLLM_CACHE_ROOT` 下进一步生成带 graph hash 和 rank 的子目录；这是
正常行为。验收重点是所有编译缓存都属于当前 `RUN_ID`，而不是沿用上一个源码版本
的目录。

这里允许复用的是由当前源码生成的 kernel/graph 编译产物，不是模型推理结果。
`--enable-prefix-caching`、答案缓存、跨请求 hidden state/KV 复用仍然禁止。服务
重启后重新分配的 KV cache 容量也不表示复用了旧请求内容。

## 7. 冷启动：只生成当前版本的编译缓存

1. `source scripts/cscc_gfx936_env.sh`，再导出上一节四个缓存变量和
   `HIP_VISIBLE_DEVICES=0`。
2. 在源码树外启动 [README 单卡命令](../../README.md#构建与单卡启动)，把日志保存为
   `cold-service.log`。
3. 等待 `/health` 成功以及模型 warmup、Triton、Inductor 和 AOT 首次编译完成。
4. 用正式 benchmark 的相同 shape 各执行少量请求，覆盖 4--8K、8--16K、16--32K
   三档可能延迟编译的 attention 路径。
5. 确认日志没有编译失败、fallback、OOM 或 engine death，然后正常停止服务。

冷启动期间的启动耗时、首请求 TTFT 和吞吐不计入正式性能；它们包含首次 JIT 成本，
不能与已经热过的历史结果直接比较。

不要执行系统级 `drop_caches`。当前评分测量的是服务就绪后的推理吞吐，不是模型
文件的磁盘冷启动时间；基线和候选版本只需采用相同的编译缓存与重启协议。

## 8. 热重启：正式性能与精度

使用完全相同的 wheel、模型、Python 环境、启动参数、设备和 `CACHE_ROOT` 再次启动
服务，日志保存为 `hot-service.log`。必须重新确认：

- 导入路径和 wheel SHA256 未变化；
- `HIP_VISIBLE_DEVICES=0`，TP/PP/DP 均为 1；
- BF16、`compile_sizes=[4096]`、`quantization=None`；
- speculative decoding 和 prefix cache 均关闭；
- `/health` 成功，KV cache 容量稳定；
- 正式计时窗口没有新的 JIT、online tuning、fallback、OOM 或 worker 重启。

如果热重启后仍发生新 kernel 编译，先让该 shape 在非计时 warmup 中完成；如果每次
重启都重复编译，则缓存没有正确命中，应先定位原因，不能把该轮写入性能结果。

正式单卡流程保持每档 2 条 warmup、50 条请求、并发 1，并使用官方原始吞吐
`total_output_tokens / duration`。三档吞吐完成后，在同一热服务执行 110 条精度集；
参数和结果要求见 [MODULAR_3K_PARITY.md](MODULAR_3K_PARITY.md)。

## 9. 源码再次修改时

任何影响运行时的源码或配置再次变化后，重复以下闭环：

1. 停止现有服务；
2. 使用新的 `RUN_ID`、空输出目录和空 build tree 重建 wheel；
3. 强制安装并从源码树外验证 Python/native 导入路径；
4. 使用新的 `CACHE_ROOT` 完成一次冷编译；
5. 使用同一新缓存目录热重启，再做 warmup、性能和精度测试。

不得仅修改源码后重启服务却继续使用旧 wheel，也不得为了省编译时间让新源码复用
旧 revision 的 Triton/Inductor/AOT cache。旧目录不必立即删除；保留到结果验收
完成有助于追溯。确需清理时，只处理自己创建且已核对的精确 `RUN_ID` 目录。

## 10. 每轮测试至少保存的证据

- commit SHA、`git status --short` 和相对官方的 runtime patch；
- wheel 文件名、SHA256、`vllm.__file__`、`vllm._rocm_C.__file__`；
- Python/DTK/PyTorch/Triton/AITER 版本；
- 冷、热两次完整启动命令、环境变量和日志；
- `CACHE_ROOT`、DCU 编号、KV cache capacity 和 warmup 数；
- 三档原始结果 JSON、客户端日志及 110 条精度结果。

DP=2 使用相同原则，但两个 rank 还必须确认各自命中 cache 且 KV cache capacity
一致；完整步骤见 [DP2_MULTI_REQUEST.md](DP2_MULTI_REQUEST.md#冷启动后必须热重启)。
