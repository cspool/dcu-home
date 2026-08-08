# 干净构建与安装

## 已验证工具链

| 组件 | 版本 |
| --- | --- |
| Python | 3.10.12 |
| DTK / HIP | 6.2.0-0；TunableOp validator 为 HIP 603 |
| PyTorch | 2.10.0+das.opt1.dtk2604.20260325.g6b060a |
| Triton | 3.4.0+git1ef59765 |
| AITER | 0.1.dev1+g9daa788.d20260401 |
| CMake / Ninja | 3.29.0 / 1.11.1 |
| 目标 | gfx936，Qwen3.5-27B BF16 |

评测容器已提供配套依赖。不要从公开 PyPI 覆盖 PyTorch、Triton、AITER 或
DTK；ROCm wheel 与运行时工具链不一致时，即使构建成功也不能沿用性能结论。

## 构建前核对官方差异

当前实现的唯一代码比较基线为：

```bash
OFFICIAL=fa718036bdb9dfd80a872b86c8ac16c9d02bfd31
git cat-file -e "$OFFICIAL^{commit}"
git diff --check "$OFFICIAL" --
git diff --name-status "$OFFICIAL" -- csrc setup.py vllm
git diff --numstat "$OFFICIAL" -- csrc setup.py vllm | \
  awk '{a += $1; d += $2; n += 1} END {print n, a, d, a + d}'
```

最后一条预期输出为 `18 567 33 600`。不要用当前 `HEAD` 代替 `OFFICIAL`：
`HEAD` 只反映工作树增量，不能回答“相对官方原版修改多少”。完整文件清单见
[官方原版优化实施指南](docs/cscc/OFFICIAL_BASE_OPTIMIZATION_GUIDE.md)。

## 从空 build tree 生成 wheel

先为 HIP 临时目标选择一个空间充足且路径较短的目录：

```bash
export TMPDIR=/path/on/a-filesystem-with-free-space
DIST_DIR=/path/to/output/dist \
MAX_JOBS=16 \
bash scripts/build_cscc_wheel.sh
```

脚本固定 `VLLM_TARGET_DEVICE=rocm`，用 `mktemp` 创建新的 build tree，并在
退出时删除临时构建目录；它不会复用仓库中的 `build/`、`dist/` 或旧 native
binary。如需保留中间目标进行调试，`BUILD_BASE` 必须指向尚不存在的路径：

```bash
BUILD_BASE=/path/to/new-build-tree \
DIST_DIR=/path/to/output/dist \
MAX_JOBS=16 \
bash scripts/build_cscc_wheel.sh
```

## wheel 静态验收

```bash
WHEEL="$(find /path/to/output/dist -maxdepth 1 -name 'vllm-*.whl' -print -quit)"
test -n "$WHEEL"
unzip -Z1 "$WHEEL" | grep -E '/_rocm_C[^/]*[.]so$'
unzip -Z1 "$WHEEL" | \
  grep 'vllm/platforms/tunable_profiles/gfx936_qwen3_5_27b_bf16_tn_m4096.csv'
if unzip -Z1 "$WHEEL" | grep -Eq '(^|/)(__pycache__/|.*[.]pyc$)'; then
  echo 'wheel contains stale bytecode' >&2
  exit 1
fi
sha256sum "$WHEEL"
```

wheel 必须包含 `_rocm_C`、共享 gfx936 helper、GQA6 op 和冻结 profile，且不得
包含 `qwen35_rocm_opt`、`rocm_qwen35_gemv.py`、编译缓存或实验日志。

## 安装与 ABI 验收

```bash
python3 -m pip install --force-reinstall --no-deps "$WHEEL"
python3 - <<'PY'
from pathlib import Path

import torch
import vllm
import vllm._rocm_C

profile = Path(vllm.__file__).parent / (
    "platforms/tunable_profiles/gfx936_qwen3_5_27b_bf16_tn_m4096.csv"
)
assert vllm.__version__ == "0.18.1"
assert hasattr(torch.ops._rocm_C, "LLMM1")
assert profile.is_file()
assert len(profile.read_text(encoding="utf-8").splitlines()) == 10
print("ROCm ABI and TunableOp profile: PASS")
PY
```

`--no-deps` 用于保留评测镜像内已经配套验证的依赖。安装后先做算子数值与
fallback 检查，再启动服务；不要以 wheel 能导入代替端到端精度验证。

## 仓库卫生

`build/`、`dist/`、wheel、`.so`、Triton/Inductor cache、模型权重和评测结果都
不属于源码提交。清理时只删除自己创建且已核对路径的构建目录；不要递归删除
共享缓存根目录。
