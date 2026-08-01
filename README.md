# PRA2026-BH408：Qwen3.5-27B 单卡 ROCm 推理优化

本仓库是 vLLM 0.18.1 的完整源码提交。直接代码基线为 OpenDAS
`vllm_cscc` 的
`fa718036bdb9dfd80a872b86c8ac16c9d02bfd31`；第三方来源、版本和许可证见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

当前 `repro-minimal` 分支在不改变最高性能版本
`pra2026-bh408` 的前提下，删除了不可达和已否决实验，收紧了专用算子接口，
并重建了可从干净 checkout 复现的构建与校验链路。

## 当前结论

- 目标：gfx936、Qwen3.5-27B、BF16、单卡、TP/PP/DP=1。
- 固定服务参数：`max_num_batched_tokens=4096`、`max_num_seqs=128`。
- 不量化，不裁剪，不使用 prefix cache，不使用投机解码。
- 重构前权威全量结果：性能分 `91.66283719585402`，精度系数 `1.00`，
  110/110 精度请求完成。
- 重构后的三档最差样本门禁：22/22 成功，输入长度、输出长度和生成文本
  SHA-256 逐条一致；相对重构前加权请求速率 `-0.072%`、TPOT
  `-0.146%`，属于运行波动。
- 重构后 clean wheel 的全量复测：性能分 `91.67227258084725`，精度
  110/110、`K=1.00`；相对重构前仅 `+0.0103%`，按性能等价而非新增收益
  解释。

最终全量复测结果记录在 [docs/cscc/RESULTS.md](docs/cscc/RESULTS.md)。

## 一次复现

```bash
bash scripts/verify_cscc_repro.sh

export TMPDIR=/path/on/a-filesystem-with-free-space
DIST_DIR="$PWD/dist-repro" bash scripts/build_cscc_wheel.sh

WHEEL="$(find "$PWD/dist-repro" -maxdepth 1 -name 'vllm-*.whl' -print -quit)"
bash scripts/verify_cscc_repro.sh "$WHEEL"
python3 -m pip install --force-reinstall --no-deps "$WHEEL"
```

启动前加载唯一的冻结 TunableOp profile：

```bash
source scripts/cscc_gfx936_env.sh
HIP_VISIBLE_DEVICES=0 vllm serve /path/to/Qwen3.5-27B \
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

不要加入 `--speculative-config`、draft/MTP 模型或 prefix-caching 参数。

## 权威材料

- [docs/cscc/CLOSED_BOOK_REPRODUCTION.md](docs/cscc/CLOSED_BOOK_REPRODUCTION.md)：
  从零构建、启动、验收和故障定位。
- [BUILD.md](BUILD.md)：工具链和干净 wheel 构建。
- [ENVIRONMENT.md](ENVIRONMENT.md)：唯一必需环境变量和精确作用域。
- [docs/cscc/OPTIMIZATION.md](docs/cscc/OPTIMIZATION.md)：保留的优化簇与文件映射。
- [docs/cscc/RESULTS.md](docs/cscc/RESULTS.md)：性能、精度和重构门禁。
- [docs/cscc/COMPLIANCE.md](docs/cscc/COMPLIANCE.md)：禁止项与提交边界。

仓库不包含模型权重、评测数据、预编译 wheel、服务日志或结果 JSON。上游
README 保存在 [README_UPSTREAM.md](README_UPSTREAM.md)。
