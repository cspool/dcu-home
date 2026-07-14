> [!IMPORTANT]
> ## 第三方开源声明
>
> 本作品基于第三方开源项目 **vLLM 0.18.1** 二次开发；本次优化的直接
> 代码基线是 OpenDAS `vllm_cscc` fork 的
> `fa718036bdb9dfd80a872b86c8ac16c9d02bfd31` commit
>（<http://developer.sourcefind.cn/codes/OpenDAS/vllm_cscc.git>）。
> GitHub vLLM（<https://github.com/vllm-project/vllm>）是原始第三方项目，
> 许可证为 Apache License 2.0；本仓库保留 `LICENSE`、源码版权头和
> 原始 README。
>
> gfx936/GQA6 attention 特化参考 AMD AITER
> `aiter/ops/triton/unified_attention.py`，验证版本为
> `0.1.dev1+g9daa788.d20260401`；GDN recurrent 源码包含
> flash-linear-attention 的 MIT 许可代码。来源、版本、许可证和改动范围见
> [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

# PRA2026-BH408：Qwen3.5-27B 单卡推理优化

这是面向 2026 智能计算创新设计赛的完整源码提交。已完成 full×3、accuracy
和 SLA 闭环的计分锚点为 **H11.5 + H10.8**；当前提交在其上加入 H10-only
ROCm TunableOp profile，以及 2026-07-15 合并验证通过的 page784 later-prefill、
低 live-range GQA6、row2 K=5120 GEMV 和连续 M-RoPE 传输修复。仓库交付完整
源码，不是仅包含差分或预编译 wheel。

## 本次实测最终结果

- 三次 full 综合分：`88.490349137758 / 88.578483694186 / 88.576533680101`
- 三轮均分：`88.5484555040153`
- 相对上一最佳 R24 的 20/50/30 加权吞吐提升：`+10.2361569769%`
- 固定 accuracy：`77.96 / 33.05 / 100.00 / 100.00`，`K=1.00`
- 三轮共 `450/450` 请求成功，TTFT/TPOT SLA 全部通过
- H11.5+H10.8 full 计分 wheel SHA256：
  `03568ba87ff64fd0a8aade299026d7ee78cbf40d9c1ed5884fb584250b2031f2`
- H10-only 本次 661 秒 all3：三档均为正，20/50/30 加权
  `+0.939469%`，27/27、输出长度和全文 hash exact
- H10-only fixed accuracy：`77.96 / 32.95 / 100 / 100`，`K=1.00`
- H10-only 独立重建 wheel SHA256：
  `fe8ceeec1634db072b179ba88f364e489640ea246eef5aab8a0487253511307a`
- 最新合并候选使用三条冻结最差样本各测试一次，相对当前最佳分别为
  `+4.697% / +9.522% / +13.611%`
- 最新合并候选 fixed accuracy 为
  `77.9597 / 32.8749 / 100 / 100`，`K=1.00`；本次提交后的完整
  `run_throughput.sh all` 尚待执行，不能提前计入权威综合分

详细测试口径见 [docs/cscc/RESULTS.md](docs/cscc/RESULTS.md)。

## 提交内容

- 完整 vLLM 源码、ROCm custom ops、CMake/Python 构建体系
- H11.5 wide-causal GQA6 prefill 源码
- H10.8 gfx936 strided LLMM1 源码
- H10-only fail-closed TunableOp loader、四行 profile 和 worker 集成
- page784 `768 + 16` 分解、官方 paged/contiguous attention producer 与
  FP32 LSE merge wrapper
- GQA6 逻辑 64-token tile 的 `2×32` 数值子块、row2 K=5120 GEMV 和
  连续 M-RoPE buffer copy
- [BUILD.md](BUILD.md)：统一评测容器内的源码编译与安装方法
- [ENVIRONMENT.md](ENVIRONMENT.md)：工具链和环境变量说明
- [scripts/cscc_gfx936_env.sh](scripts/cscc_gfx936_env.sh)：H10-only 启动环境
- [docs/cscc/OPTIMIZATION.md](docs/cscc/OPTIMIZATION.md)：技术路线与贡献
- [docs/cscc/COMPLIANCE.md](docs/cscc/COMPLIANCE.md)：赛题边界与提交审计
- [evidence/manifests](evidence/manifests)：源码/运行时/最终 wheel 哈希锚点
- [README_UPSTREAM.md](README_UPSTREAM.md)：vLLM 上游 README

模型权重、tokenizer、固定评测脚本、测试数据、原始 benchmark 输出、
build 目录和预编译 wheel 不在仓库中；它们由评测平台提供或在评测机生成。

## 快速编译

评测容器必须预装指定 DTK/PyTorch/Triton/AITER 环境。不要用公开 PyPI
requirements 覆盖平台定制包。

```bash
bash scripts/build_cscc_wheel.sh
python3 -m pip install --force-reinstall --no-deps dist/vllm-*.whl
```

构建脚本复现已验证的 `build_py --force` 和 `bdist_wheel` 流程。完整说明
和精确工具链版本见 [BUILD.md](BUILD.md)。

## 服务与评测

本提交不修改组委会固定的 `start_vllm.sh`、`run_throughput.sh` 或
`run_accuracy.sh`。安装 wheel 后应直接使用评测平台提供的固定脚本启动
服务和评测。H10-only 提交版需在启动前执行：

```bash
source scripts/cscc_gfx936_env.sh
bash /path/to/testdata/start_vllm.sh
```

变量作用、fail-closed 范围及固定默认值见 [ENVIRONMENT.md](ENVIRONMENT.md)。

## 源码完整性检查

以下命令分别校验当前提交的 15 个累计核心源码文件和 H10-only 增量文件：

```bash
sha256sum -c evidence/manifests/repo_source.sha256
sha256sum -c evidence/manifests/h10_only_submission.sha256
bash -n scripts/build_cscc_wheel.sh
source scripts/cscc_gfx936_env.sh
```

两个检查均应成功。最终候选刻意保留 `vllm/version.py` 中上游累计栈已有的
两处尾随空格，以维持已测试源码的逐字节哈希；这不影响编译或运行。

## 许可证

主项目沿用 Apache License 2.0，见 [LICENSE](LICENSE)。第三方来源及 MIT
许可文本见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
