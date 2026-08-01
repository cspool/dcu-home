# 干净构建与安装

## 已验证工具链

| 组件 | 版本 |
| --- | --- |
| Python | 3.10.12 |
| DTK / HIP | 6.2.0-0，运行时 validator 为 HIP 603 |
| PyTorch | 2.10.0+das.opt1.dtk2604.20260325.g6b060a |
| Triton | 3.4.0+git1ef59765 |
| AITER | 0.1.dev1+g9daa788.d20260401 |
| CMake / Ninja | 3.29.0 / 1.11.1 |
| 目标 | gfx936，BF16，单卡 |

评测容器已提供定制依赖。不要用公开 PyPI requirements 覆盖 PyTorch、
Triton、AITER 或 DTK。

## 构建

先确保 `TMPDIR` 所在文件系统有空间。HIP 编译器会在其中创建较大的临时
目标；仓库和输出目录有空间并不能弥补 `/tmp` 已满。该变量也会被服务进程
用于 ZMQ IPC，因此路径应尽量短，连同随机文件名必须小于 107 字符。

```bash
export TMPDIR=/path/on/a-filesystem-with-free-space
DIST_DIR="$PWD/dist-repro" bash scripts/build_cscc_wheel.sh
```

脚本固定 `VLLM_TARGET_DEVICE=rocm`、默认 `MAX_JOBS=16`，并在一个新的空
`BUILD_BASE` 中构建。因此旧 `build/` 内的 `.pyc`、已删除模块或旧 native
binary 不会混入 wheel。默认临时 build tree 在退出时删除。

如需保留 build tree 做运行时检查，传入一个尚不存在的路径：

```bash
MAX_JOBS=16 \
BUILD_BASE=/path/to/new-build-tree \
DIST_DIR=/path/to/dist \
bash scripts/build_cscc_wheel.sh
```

为防止污染，显式 `BUILD_BASE` 已存在时脚本会失败，不会复用。

## 强制校验

```bash
WHEEL="$(find "$PWD/dist-repro" -maxdepth 1 -name 'vllm-*.whl' -print -quit)"
bash scripts/verify_cscc_repro.sh "$WHEEL"
```

校验内容包括：

- OpenDAS 基线对象或提交快照关系；
- 必需优化文件、profile SHA-256、5 个 validators 和 5 个结果；
- Python/shell 语法和补丁格式；
- 已否决 GEMV/SwiGLU 实验不存在；
- wheel 包含 `_rocm_C` 和冻结 profile；
- wheel 不含 `.pyc`、`__pycache__` 或已删除实验模块。

## 安装与导入

```bash
python3 -m pip install --force-reinstall --no-deps "$WHEEL"
python3 - <<'PY'
import torch
import vllm
import vllm._rocm_C

assert vllm.__version__ == "0.18.1"
assert hasattr(torch.ops._rocm_C, "qwen35_bf16_gemv")
assert not hasattr(torch.ops._rocm_C, "LLMM1Strided")
print("vLLM and qwen35_bf16_gemv: OK")
PY
```

`--no-deps` 是必要的：它保留评测镜像内经过配套验证的定制依赖。
