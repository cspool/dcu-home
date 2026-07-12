# 环境变量说明

## 结论

最终 H11.5 + H10.8 优化**不依赖新增的性能控制环境变量**。两项优化均由
源码中的设备、dtype 和精确 shape gate 自动命中。

尤其是：

- H10.8 不要求设置 `VLLM_ROCM_USE_SKINNY_GEMM`；
- 不需要通过环境变量强制切换 BLAS backend；
- 不使用 H10.10 K6144 或任何实验性开关；
- 不硬编码 `HIP_VISIBLE_DEVICES` 或 `CUDA_VISIBLE_DEVICES`。

## 编译变量

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `MAX_JOBS` | `16` | 控制本地源码编译并行度 |
| `VLLM_TARGET_DEVICE` | `rocm` | 明确选择 ROCm 构建目标 |
| `DIST_DIR` | `<repo>/dist` | wheel 输出目录，仅由提交构建脚本使用 |

这些变量只影响编译过程，不改变正式推理算法或计分参数。

## 评测平台运行变量

组委会固定启动脚本支持以下变量及默认值：

| 变量 | 固定默认值 | 用途 |
| --- | --- | --- |
| `MODEL_DIR` | `../Qwen3.5-27B` | 平台提供的官方模型目录 |
| `SERVED_MODEL_NAME` | `Qwen3.5-27B` | OpenAI API 模型名 |
| `VLLM_PORT` | `8001` | 服务端口 |
| `MAX_NUM_SEQS` | `128` | 固定服务参数 |
| `MAX_NUM_BATCHED_TOKENS` | `4096` | 固定服务参数 |
| `GPU_MEMORY_UTILIZATION` | `0.95` | 固定显存利用率参数 |

本提交不覆盖这些值，也不修改启动脚本。

## GPU 与动态库环境

`HIP_VISIBLE_DEVICES`、`CUDA_VISIBLE_DEVICES`、ROCm 动态库路径和设备分配
由评测平台管理，仓库脚本不写死这些值。

## localhost 与代理

评测服务运行在 `127.0.0.1:8001`。如果容器存在 HTTP(S) 代理，访问本地
服务时应由平台脚本清除代理，或设置：

```bash
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
```

这只避免 localhost 请求误入代理，不属于性能优化。
