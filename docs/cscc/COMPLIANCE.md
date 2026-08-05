# 提交边界与禁止项审计

## 未改变的评测资产

本仓库不包含、也不修改：

- Qwen3.5-27B 权重、tokenizer、chat template；
- throughput/accuracy 数据集；
- 组委会的 `start_vllm.sh`、`run_throughput.sh`、`run_accuracy.sh`；
- 单请求调度边界、temperature 或输出长度；
- 评分公式和 accuracy 后处理。

外部脚本哈希只作 provenance 记录，不把脚本复制进提交源码。

## 当前优化未使用

- speculative decoding、draft/MTP/辅助模型；
- prefix cache 或跨请求结果复用；
- 量化、权重重排持久化、模型权重修改；
- 剪枝、layer/head/channel/token 跳过、early exit；
- prompt 内容、数据集名称或样本答案特判；
- scheduler/batch 逻辑修改；
- 在线 autotune 或未记录的 hipBLASLt solution；
- 多卡、TP=2 或 DP=2 配置。

完整 vLLM 上游源码仍包含通用 `spec_decode` 模块；模块存在不表示正式服务
启用。运行时日志必须明确 `speculative_config=None`。

## 数值语义

GQA6、GDN 和 GEMV 会改变浮点归约排布，但仍计算完整 causal
attention、完整 token/head/layer 和相同权重。没有近似注意力、token 丢弃或
提前停止。精度以固定 110 条评测和生成文本回归门禁验证，不以 kernel 单测
替代端到端证据。

## Fail-closed 范围

- Attention：gfx936、BF16、head256、GQA6、causal prefill。
- GDN T4096：gfx936、BF16、Q/K/V/head layout、int32 metadata、单序列。
- GEMV：BF16、单 token、连续/对齐、6 个固定 `(M,K)` shape。
- TunableOp：Qwen3.5、BF16、4096 chunk、TP/PP/PCP/DP=1、冻结工具链。
- 静态编译：同上，且 `speculative_config is None`。

非目标 attention/GDN/GEMM 使用上游路径；TunableOp 配置不一致则拒绝启动，
防止带着未知 profile 进入正式评测。

## 仓库卫生

提交不应包含：

- `build/`、`dist/`、wheel、native binary；
- `.pyc`、`__pycache__`、旧实验模块；
- 模型权重、测试数据、prediction/result JSON、服务日志；
- `.env`、token、credential、私钥。

`scripts/build_cscc_wheel.sh` 在新空 build tree 中构建；
`scripts/verify_cscc_repro.sh <wheel>` 强制检查 wheel 没有从旧 build tree
继承残留。

## 第三方边界

项目沿用 Apache License 2.0。GQA6 路径参考 AMD AITER，GDN 代码包含
flash-linear-attention 来源。
精确版本、许可证和改动边界见根目录
`THIRD_PARTY_NOTICES.md`；不把第三方 kernel 表述为自研成果。
