# 提交完整性与合规说明

## 官方提交要求映射

| 要求 | 本仓库材料 |
| --- | --- |
| 完整源代码 | `vllm/`、`csrc/`、`cmake/`、`setup.py`、`pyproject.toml` 等 |
| 编译脚本 | `scripts/build_cscc_wheel.sh`、`BUILD.md` |
| 环境变量说明 | `ENVIRONMENT.md` |
| 优化方案与贡献 | `docs/cscc/OPTIMIZATION.md`、`docs/cscc/RESULTS.md` |
| 第三方声明 | README 顶部、`THIRD_PARTY_NOTICES.md`、`LICENSE` |

## 最终候选源码

以下 13 个文件构成当前提交相对上游基线的累计核心源码锚点；其中
`setup.py` 已包含 H10-only profile package-data：

1. `csrc/rocm/ops.h`
2. `csrc/rocm/skinny_gemms.cu`
3. `csrc/rocm/torch_bindings.cpp`
4. `setup.py`
5. `vllm/_custom_ops.py`
6. `vllm/model_executor/layers/fla/ops/fused_recurrent.py`
7. `vllm/model_executor/layers/utils.py`
8. `vllm/model_executor/models/qwen3_next.py`
9. `vllm/platforms/rocm.py`
10. `vllm/v1/attention/backends/gdn_attn.py`
11. `vllm/v1/attention/backends/rocm_aiter_unified_attn.py`
12. `vllm/v1/attention/ops/rocm_aiter_unified_attention_gqa6.py`
13. `vllm/version.py`

第 12 项是新增核心文件，已正式纳入本提交。运行：

```bash
sha256sum -c evidence/manifests/repo_source.sha256
```

应得到 13 项 `OK`。

H10-only 新增/修改 `.gitignore`、`setup.py`、
`vllm/v1/worker/gpu_worker.py`，并新增：

- `vllm/platforms/rocm_tunableop.py`
- `vllm/platforms/tunable_profiles/gfx936_qwen3_5_27b_bf16_tn_m4096.csv`
- `scripts/cscc_gfx936_env.sh`

增量哈希由下列命令校验：

```bash
sha256sum -c evidence/manifests/h10_only_submission.sha256
```

## 未修改的评测资产

本仓库不包含且没有修改：

- 官方 Qwen3.5-27B BF16 权重；
- 官方 tokenizer 和 chat template；
- 固定 `start_vllm.sh`；
- 固定 `run_throughput.sh`；
- 固定 `run_accuracy.sh`；
- 官方 throughput/accuracy 数据集；
- 固定服务参数和单请求调度边界。

对应固定脚本哈希记录在
`evidence/manifests/fixed_scripts_reference.txt`。它是评测平台外部脚本
的引用记录，不是可在本仓库执行 `sha256sum -c` 的清单。

## 禁止项审计

完整 vLLM 源码保留上游的通用能力；以下结论专指本次相对 OpenDAS 基线的
优化改动和最终评测配置：

- 最终候选改动不含 H10.10 K6144 pair-reduce gate 或 ABI marker；
- 最终候选改动不含 H10.9 强制 hipBLAS backend 切换；
- 本次改动未新增模型权重持久化量化、重排或跨样本结果缓存；
- 最终评测未启用 prefix cache；
- 本次改动未新增 prompt/数据集特判；
- 上游 scheduler 源码随完整项目提交，但本次优化未修改 scheduler/batch
  文件或评测边界；
- H10-only 环境变量全部公开记录在 `ENVIRONMENT.md` 和
  `scripts/cscc_gfx936_env.sh`；没有隐藏性能开关。

## 仓库排除项

提交中不应出现：

- `build/`、`dist/`、`.deps/`、`*.egg-info/`
- `*.whl`、`*.so`
- `*.safetensors`、`*.pt`、`*.pth`、`*.onnx`、`*.gguf`
- 模型权重、原始 benchmark/accuracy 输出或 goal_runs
- `.env`、`config.env`、credential、token、私钥

预编译产物不能替代完整源码。

## 已知格式项

`vllm/version.py` 第 15、16 行包含最高分冻结源码已有的两处尾随空格。
为保持与完整 full/accuracy 测试源码逐字节一致，本提交原样保留；它不影响
Python 语义、编译或运行。除这两处外，冻结源码差分没有其他
`git diff --check` 问题。

## 提交前验证

1. baseline 13 项源码和 H10-only 增量哈希全部通过；
2. 新增 GQA6 文件已被 Git 跟踪；
3. 构建脚本 `bash -n` 通过；
4. 无模型、wheel、native binary、测试数据或凭据；
5. README 第一屏包含第三方声明；
6. Git remote URL 不含用户名或密码；
7. 远端 commit 与本地远程容器 commit SHA 一致。
