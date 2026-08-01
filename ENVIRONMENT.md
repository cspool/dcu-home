# 运行环境与开关

## 唯一必需的性能环境

正式服务启动前执行：

```bash
source scripts/cscc_gfx936_env.sh
```

该脚本只启用 wheel 内的冻结 TunableOp profile：

| 变量 | 固定值 |
| --- | --- |
| `VLLM_ROCM_TUNABLEOP_PROFILE` | `gfx936_qwen3_5_27b_bf16_tn_m4096` |
| `VLLM_ROCM_TUNABLEOP_PROFILE_SHA256` | `169c7b11a0340d9e22405327b5e5667b2aa9e9e8d899bd59e10ca4fb7fb52030` |
| `PYTORCH_TUNABLEOP_ENABLED` | `1` |
| `PYTORCH_TUNABLEOP_TUNING` | `0` |
| `PYTORCH_TUNABLEOP_RECORD_UNTUNED` | `0` |
| `PYTORCH_TUNABLEOP_ROCBLAS_ENABLED` | `1` |
| `PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED` | `0` |

profile 包含 5 个 validators 和 5 个 `M=4096` BF16 TN GEMM 结果，覆盖
`K/N=5120/14336`、`5120/16384`、`5120/34816`、
`17408/5120`、`6144/5120`。这里的 `K=6144` 是冻结 rocBLAS prefill
solution，不是已删除的单 token K6144 pair-reduce 实验。

loader 在 device/distributed 初始化后加载，并在 graph capture 前再次检查
API 状态、文件哈希、validators 和结果集合。以下任一漂移都会 fail closed：

- 非 Qwen3.5-27B BF16；
- `max_num_batched_tokens != 4096`；
- TP/PP/PCP/DP 不是 1；
- profile 名称、SHA、工具链 validator 或 solution 集合变化；
- 在线 tuning、record-untuned 或 hipBLASLt 被开启。

## 无需设置的开关

page784、GQA6、GDN、固定形状 GEMV、连续 M-RoPE 和静态 4096 编译形状均由
源码按 device/dtype/model/shape 精确选择。不要设置历史实验变量；校验脚本会
拒绝源码中重新出现的 gate-up+SwiGLU、旧 `LLMM1Strided` 或 disable 开关。

## 固定服务契约

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
```

`HIP_VISIBLE_DEVICES` 和端口由运行者选择，不由仓库脚本写死。正式配置不增加
`--speculative-config`、draft/MTP、prefix caching、量化或多卡参数。

## 缓存与首次启动

冷 checkout 会生成 Triton、TorchInductor 和 vLLM compile cache。首次启动
比复用缓存慢是正常现象；正式性能测试必须等 `/health` 成功，并确认日志出现：

```text
Using the validated 4096-token static compile shape
VLLM_ROCM_TUNABLEOP_INIT status=ready
VLLM_ROCM_TUNABLEOP_PRE_CAPTURE status=ready
speculative_config=None
```

若宿主配置了 HTTP 代理，应设置 `NO_PROXY=127.0.0.1,localhost`，避免本地
benchmark 请求进入代理。构建时若 `/tmp` 空间不足，显式把 `TMPDIR` 指向
有空间的文件系统；该变量不改变推理算法。
