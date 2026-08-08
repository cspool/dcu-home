# 运行环境与开关

## 单卡正式配置

初赛与当前 R24 全量结论只对应以下服务契约：

```text
HIP_VISIBLE_DEVICES=0
dtype=bfloat16
tensor_parallel_size=1
pipeline_parallel_size=1
data_parallel_size=1
max_num_batched_tokens=4096
max_num_seqs=128
gpu_memory_utilization=0.95
enable_thinking=false
speculative_config=None
quantization=None
enable_prefix_caching=False
```

模型、tokenizer、chat template、最大输出、采样参数和 batch scheduler 参数以
组委会脚本为准，不通过环境变量覆盖。

## TunableOp 环境变量

正式启动前执行：

```bash
source scripts/cscc_gfx936_env.sh
```

脚本设置：

| 变量 | 固定值 | 作用 |
| --- | --- | --- |
| `VLLM_ROCM_TUNABLEOP_PROFILE` | `gfx936_qwen3_5_27b_bf16_tn_m4096` | 选择 wheel 内置 CSV |
| `VLLM_ROCM_TUNABLEOP_PROFILE_SHA256` | `169c7b11...52030` | 提交与人工核对用的兼容性记录 |
| `PYTORCH_TUNABLEOP_ENABLED` | `1` | 启用已加载结果 |
| `PYTORCH_TUNABLEOP_TUNING` | `0` | 禁止在线调优 |
| `PYTORCH_TUNABLEOP_RECORD_UNTUNED` | `0` | 禁止运行时记录未知 shape |
| `PYTORCH_TUNABLEOP_ROCBLAS_ENABLED` | `1` | 使用冻结 rocBLAS solution |
| `PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED` | `0` | 不切换到未验证 hipBLASLt solution |

CSV 含 5 个工具链 validators 和 5 个 M=4096 BF16 TN GEMM 结果。源码 loader
在 worker 完成 distributed device 选择后加载该文件，关闭 tuning/record，并要求
PyTorch 返回恰好 5 条结果；文件不存在或结果数不符时直接启动失败。

`VLLM_ROCM_TUNABLEOP_PROFILE_SHA256` 目前是提交侧核对值，不应误写成 loader
内部再次计算哈希；构建与安装验收应显式执行：

```bash
sha256sum vllm/platforms/tunable_profiles/gfx936_qwen3_5_27b_bf16_tn_m4096.csv
```

## 不使用的开关

GQA6/page784、GDN、固定 shape GEMV、连续 M-RoPE 和静态 4096 编译形状均由
device、dtype、model、shape、stride 与功能条件精确选择，不需要实验开关。
正式配置不得增加：

- 权重量化、持久化权重转换或压缩缓存；
- `--speculative-config`、draft/MTP 或其他投机解码；
- prefix cache；
- 修改 temperature、最大输出长度、输入长度或 scheduler 的变量。

## 单卡启动

```bash
source scripts/cscc_gfx936_env.sh
export HIP_VISIBLE_DEVICES=0
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"

vllm serve /path/to/Qwen3.5-27B \
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

启动日志至少核对架构、BF16、TP/PP/DP、`compile_sizes=[4096]`、
`speculative_config=None`、`quantization=None` 和 prefix cache 状态。只有
`/health` 成功且 Triton/Inductor/AOT 首次编译结束后才开始性能计时。

## 可选 DP=2

决赛多卡准备保留 `HIP_VISIBLE_DEVICES=0,1`、`TP=1`、`DP=2`、`backend=mp`
配置；它不是初赛单卡成绩的一部分。使用方法和双 rank 热重启检查见
[docs/cscc/DP2_MULTI_REQUEST.md](docs/cscc/DP2_MULTI_REQUEST.md)。
