# 源码编译与安装

## 已验证环境

本版本在组委会统一容器中完成构建、安装和评测。已验证工具链如下：

| 组件 | 版本 |
| --- | --- |
| Python | 3.10.12 |
| DTK / HIP | 6.2.0-0 |
| Clang | 18.0.0 |
| PyTorch | 2.10.0+das.opt1.dtk2604.20260325.g6b060a |
| Triton | 3.4.0+git1ef59765 |
| AITER | 0.1.dev1+g9daa788.d20260401 |
| CMake | 3.29.0 |
| Ninja | 1.11.1.git.kitware.jobserver-1 |
| transformers | 5.5.0 |
| 目标设备 | gfx936，BF16，单卡 |

评测容器已经提供上述定制依赖。**不要**执行公开 PyPI/ROCm requirements
的全量重装，否则可能覆盖平台定制的 PyTorch、Triton 或 AITER。

## 一键构建

在仓库根目录执行：

```bash
bash scripts/build_cscc_wheel.sh
```

脚本只使用预装环境，默认：

- `MAX_JOBS=16`
- `VLLM_TARGET_DEVICE=rocm`
- 输出目录为仓库内 `dist/`

需要调整构建并行度或输出目录时可以显式覆盖：

```bash
MAX_JOBS=8 DIST_DIR=/tmp/vllm-dist bash scripts/build_cscc_wheel.sh
```

## 最终 evidence 记录的原始构建命令

```bash
python3 setup.py build_py --force
MAX_JOBS=16 python3 setup.py bdist_wheel --dist-dir <dist>
```

原始成功构建由 setup 自动识别 ROCm。提交的一键复现脚本另行显式设置
`VLLM_TARGET_DEVICE=rocm`，用于在统一评测容器中消除目标设备歧义；这不是
对历史 evidence 命令的改写。

## 安装

```bash
python3 -m pip install --force-reinstall --no-deps \
  dist/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl
```

使用 `--no-deps` 是为了保留评测容器预装的定制依赖。

## 已验证产物

```text
vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl
SHA256 03568ba87ff64fd0a8aade299026d7ee78cbf40d9c1ed5884fb584250b2031f2
```

最终 wheel 不提交到 GitLab；评测机应从本仓库源码重新编译。对应源码哈希见
`evidence/manifests/repo_source.sha256`。

## 构建后检查

```bash
python3 -m pip show vllm
sha256sum dist/vllm-*.whl
```

预期版本为 `0.18.1+das.dtk2604`。安装完成后使用组委会提供且未修改的
`start_vllm.sh`、`run_throughput.sh` 和 `run_accuracy.sh` 进行评测。
