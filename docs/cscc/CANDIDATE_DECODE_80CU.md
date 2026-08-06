# 80-CU Decode Attention 候选审计

状态：单请求服务门禁未通过，已冻结为研究候选；不得合并到当前最佳、运行
full 测试或计入性能得分。

## 实现与硬件对齐

目标文件：

- `vllm/v1/attention/ops/rocm_aiter_decode_attention_gqa6.py`
- `vllm/v1/attention/backends/rocm_aiter_unified_attn.py`

Qwen3.5-27B 的目标 full-attention 层为 24 个 query heads、4 个 KV heads。
AITER 原 decode 路径使用 16 个 context segments，只启动 `4 * 16 = 64` 个主
workgroups，不能填满 gfx936 的 80 个 CU。候选使用 20 个 active segments，
得到 `4 * 20 = 80` 个主 workgroups。最终归约仍在 FP32 中执行；由于 Triton
归约轴必须为 2 的幂，归约 kernel 使用 32 lanes 并屏蔽 20--31。

候选仅在以下条件全部满足时启用：gfx936、BF16、单请求单-token decode、
24/4 heads、head size 256、cache page 784、标准三维 Q/output 布局、无 ALiBi、
无 sliding window、无 softcap。未命中的调用回退现有 AITER 路径。条件只来自
设备与张量元数据，不读取 prompt、数据集名称或请求内容。

## 赛题限制逐项审计

该候选对应技术方案第 5 条允许的 Decode Attention kernel 优化，以及第 6 条
允许的 DCU 并行执行特征与资源组织适配。它保持：

- 官方 BF16 权重、tokenizer、chat template 和模型结构不变；
- Q/K/V 与 KV cache 均为 BF16，不量化权重、激活或 KV cache；
- causal mask、softmax 公式、全部 token/head/layer 计算和输出接口不变；
- scheduler、batch 参数、采样参数、服务接口和 benchmark/accuracy 脚本不变；
- 不使用剪枝、跳层、跳 token、early exit、投机解码、prefix cache、测试集缓存
  或请求内容特判；
- 仅分配当前 kernel 调用期内的 FP32 segment 临时工作区，不生成任何可复用
  权重、量化权重、压缩权重、模型文件或跨样本结果缓存。

20 段与现有 16 段的浮点结合顺序不同，因此 BF16 结果不保证逐位等同于旧
kernel；数学上仍计算同一完整 Attention。算子门禁已用独立 FP32 QK-softmax-PV
参考验证候选误差不劣于现有实现，但这不能替代固定服务输出和官方 accuracy
测试。若后续固定输出或 accuracy 门禁失败，候选必须回退，不能计入收益。

## 第三方代码与环境

主 kernel 来自评测容器预装 AMD AITER
`aiter/ops/triton/unified_attention.py`；masked reduction 由该文件的
`reduce_segments` 派生。版本、上游地址、许可证和自研边界已同步记录在仓库
根目录 `THIRD_PARTY_NOTICES.md`，源码保留 Apache-2.0 SPDX 与版权头。

服务候选没有新增自定义运行时环境变量。研究脚本为固定物理卡设置
`HIP_VISIBLE_DEVICES=0`、`ROCR_VISIBLE_DEVICES=0`、
`CUDA_VISIBLE_DEVICES=0`；独立 Triton 探针曾用
`LD_PRELOAD=/opt/dtk/lib/libamdhip64.so` 解决探针进程的符号可见性，该变量不
属于服务实现或性能开关，服务门禁不得依赖它。如最终服务确实需要新增环境
变量，必须先更新 `ENVIRONMENT.md` 和启动说明。

## 当前证据边界

算子级测试在物理 DCU 0 上完成；进程只见一个 gfx936 设备和 80 个 CU。
长度 7574/13962 的 kernel 时间分别由 75.373/127.866 微秒降至
63.082/104.963 微秒，集成模块与筛选模块逐位一致并可重复。该结果只证明
算子门禁通过；尚不能表述为端到端吞吐提升、SLA 通过或精度得分不扣分。

后续固定服务门禁中，4--8K 首条样本的 33-token 文本与当前最佳 exact，
8--16K 首条样本则发生 greedy 文本分岔。服务保持 `quantization=None`，两条
请求均成功且输出 33 tokens，但候选未满足预注册的双档 exact-text 条件。因此
停止在小样本阶段，不运行 full throughput/accuracy，也不把算子微基准外推为
端到端收益。完整证据位于研究仓库的
`H4d-attention-80cu-service-gate-dcu0-r1`。
