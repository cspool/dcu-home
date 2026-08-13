# TP2 batch10 优化、构建与评测

## 结论

本分支提供的是双卡 `TP=2、DP=1`、服务端可容纳多 batch、客户端最大并发
10 的提交版本。现在已经补齐两层一键入口：

- `scripts/setup_cscc_tp2_batch10.sh`：构建、校验、无依赖解析安装、原生扩展导入检查；
- `scripts/run_cscc_tp2_batch10_e2e.sh`：可继续启动服务、等待就绪，并原样调用官方
  `run_throughput.sh` 和 `run_accuracy.sh`。

这两个脚本不会改写官方测试脚本。当前下载包中两个官方入口与原始压缩包逐字节一致：

```text
run_throughput.sh  209dcdcc9b1ba8a606afb9aec0dbff74b873170ba43b6685ebc7f2112011fdd7
run_accuracy.sh    d03b3525fb5387a5dda1bfeda1e3f60c604f4cc51f08c11fb227eef516fc0f89
```

## 一键构建和安装

在评测机已有官方镜像依赖和源码的前提下执行：

```bash
cd /path/to/pra2026-bh408-gqa-page784-k5120-tp2-batch10
OFFICIAL_TEST_DIR=/path/to/unpacked-testdata \
MAX_JOBS=16 \
bash scripts/setup_cscc_tp2_batch10.sh
```

脚本依次完成以下工作：

1. 用本仓库的 `setup.py` 构建干净 wheel；
2. 校验运行时源码清单、优化文件、TunableOp 配置和 wheel 内容；
3. 用 `pip --no-deps --force-reinstall` 安装，既不联网，也不解析或安装额外依赖；
4. 从安装目录导入 `vllm._C`、`vllm._rocm_C`，并核对安装后的调度器与源码
   SHA-256；
5. 对官方测试入口只执行语法和哈希检查，不执行、不修改测试。

若已有已验证 wheel，可省去重新编译：

```bash
WHEEL=/path/to/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl \
OFFICIAL_TEST_DIR=/path/to/unpacked-testdata \
bash scripts/setup_cscc_tp2_batch10.sh
```

构建脚本会恢复 `setup.py` 构建期间自动刷新的 `vllm/version.py`，因此成功或失败的
构建都不会因版本生成步骤污染提交源码。

## 启动服务

安装完成后可直接启动，不需要再设置隐藏的优化开关：

```bash
MODEL_DIR=/path/to/Qwen3.5-27B \
bash scripts/serve_cscc_tp2_batch10.sh
```

启动参数固定为：

```text
served model       Qwen3.5-27B
port               8001
tensor parallel    2
data parallel      1
max model length   32768
max sequences      128
batched tokens     4096
GPU memory util.   0.95
```

`HIP_VISIBLE_DEVICES` 默认是 `0,1`，也可在启动前明确设置为两个不同设备。服务脚本
会拒绝单卡、重复设备、错误端口或缺少模型配置的启动方式。启动前还会切换到源码树
之外的中立运行目录（默认 `/tmp`），避免当前工作目录中的 `vllm/` 遮蔽已安装 wheel
及其原生扩展；模型路径和 `vllm` 可执行文件都会先转换为绝对路径。

## 一键启动并调用官方脚本

完整评测时，只需提供模型和解压后的官方测试目录：

```bash
MODEL_DIR=/path/to/Qwen3.5-27B \
OFFICIAL_TEST_DIR=/path/to/unpacked-testdata \
TEST_TARGET=all \
bash scripts/run_cscc_tp2_batch10_e2e.sh
```

默认 `RUN_SETUP=1`，所以上述单条命令会执行“构建、校验、安装、启动、等待就绪、
吞吐、精度、关闭服务”的完整过程。可选择：

```text
TEST_TARGET=throughput  只调用 bash ./run_throughput.sh all
TEST_TARGET=accuracy    只调用 bash ./run_accuracy.sh all
TEST_TARGET=all         依次调用两个官方入口
TEST_TARGET=health      只启动并验证 /v1/models，不运行评测
RUN_SETUP=0             使用当前已经安装的 wheel，跳过构建安装
```

官方脚本在其原目录、用其原参数形式执行。封装层只传入官方已经支持的
`MODEL_DIR`、`MODEL_PATH`、`CONCURRENCY=10`、`SERVICE_PORT`、`RUNTIME_BASE` 和
`EVAL_WORK_DIR` 环境变量，不复制、不打补丁，也不改调用方式。因为当前官方入口固定
访问 8001 端口，运行正式测试时封装层也强制端口为 8001。

当前下载包中的精度调试入口会在新的 `EVAL_WORK_DIR` 中调用
`python run.py bench.py`，但压缩包本身没有包含 `run.py`。一键封装会将仓库内经过
验证的标准 OpenCompass CLI 入口 `scripts/opencompass_accuracy_entry.py` 复制到本次
运行目录，并通过官方已经支持的 `EVAL_WORK_DIR`、`RUNTIME_BASE` 环境变量传入。
这只补齐运行环境，不修改官方 `run_accuracy.sh` 的内容或命令行接口。

## 官方入口直跑补测

最终提交安装后重新启动同一 TP2 服务，并从原始压缩包解压出全新测试目录进行了
小样本直跑：

| 官方入口 | 参数 | 结果 |
|---|---|---|
| `run_throughput.sh` | `4-8K 1` | 成功 1、失败 0，总吞吐 1255.96 token/s |
| `run_accuracy.sh` | `hotpotqa 1` | rc=0，OpenCompass 分数 20.00 |

两份脚本在执行前后 SHA-256 均与原始压缩包一致。服务在两项测试后仍可访问，日志中
没有 OOM、HTTP 5xx、Traceback、HIP/kernel error，随后正常关闭并释放双卡显存。

## 当前优化实现

相对已有 TP2 batch10 版本，本轮保留的新增优化只改变匹配 Qwen3.5-27B、
`max_num_batched_tokens=4096`、`max_num_seqs=128`、图捕获上限 16 的调度路径：

- TP2 遇到超过 16K 的 prompt 时，prefill 调度预算由保守的 512 提升到 4096；
- 8K–16K 和 8K 以下仍使用 2048；
- 非 TP2 拓扑继续使用原有的 512/1024/2048 保守回退；
- 不匹配模型形状或服务配置时，继续走 vLLM 原调度路径。

TP2 将中间激活按 rank 分片，双卡实测有足够余量；上述限制同时避免把更激进预算
错误应用到 DP 或未知拓扑。已有的 kernel 形状检查、图捕获上限和普通 PyTorch/
vLLM fallback 仍然保留。

## 性能和正确性证据

最终 4096 调度候选完成了三档全量吞吐数据；该次数据由仓库封装器执行，其核心
`vllm bench serve` 参数与当前下载的官方吞吐入口一致，但不是再次执行官方文件本身。
用户要求停止重复原版测试后，没有再重跑原版吞吐或精度。

| 输入长度 | 完成/失败 | 最终耗时（秒） | 总 token/s | 相对上一版耗时 | 相对初始基线耗时 |
|---|---:|---:|---:|---:|---:|
| 4–8K | 80/0 | 280.88 | 1966.69 | 快 0.81% | 快 20.31% |
| 8–16K | 70/0 | 376.91 | 2462.97 | 快 1.47% | 快 30.70% |
| 16–32K | 60/0 | 752.55 | 2080.55 | 快 13.77% | 快 43.24% |

三档按“总 token 数之和 / 总耗时之和”合并，上一版为 1979.936 token/s，最终版为
2160.073 token/s，提升 **9.10%**。这接近但没有虚报为 10%；长输入档总 token/s
单独提升约 16.0%。

在最终实现上，官方原版精度入口此前完整通过，结果为：

| 数据集 | 最终分数 | 运行状态 |
|---|---:|---:|
| HotpotQA | 67.95 | rc=0 |
| GovReport | 33.07 | rc=0 |
| Retrieval multi-point | 100.00（20/20） | rc=0 |
| Aggregation keyword | 75.00（15/20） | rc=0 |

与对照结果相比没有精度退化。Aggregation 的 5 个未命中请求都在官方 1024 输出
token 上限处以 `finish_reason=length` 截断，不是 TP2 数值错误、OOM 或接口错误。

## 混合长度与 OOM 回退

最终实现还补测了三种短、中、长输入混排顺序。每组 10 个请求，由 4 个短请求、
3 个中请求和 3 个长请求组成，并发为 10：

| 顺序 | 完成/失败 | 总 token/s |
|---|---:|---:|
| 交替混排 | 10/0 | 2578.156 |
| 长请求优先 | 10/0 | 2586.267 |
| 短请求优先 | 10/0 | 2584.083 |

三组均无 OOM、HTTP 5xx、Traceback、HIP/kernel error 或服务重启。最终全量服务启动
时 KV cache 为 268,128 tokens，32K 理论并发约 30.4，运行中观测到的最高 KV 使用率
约 27%。因此当前并发 10 留有余量。

面对未知的决赛混合组合，不能数学上保证任何输入都永不 OOM，但当前实现有以下保险：

- 最大上下文 32768、并发槽 128、单步 token 预算 4096 的硬上限；
- 仅 TP2 长 prompt 使用 4096，其他拓扑自动退回保守预算；
- 图捕获只覆盖已验证的小 batch，其他形状走安全回退；
- 自定义 kernel 均有模型、dtype、shape 和拓扑门控，不匹配即走官方实现；
- 一键脚本在正式调用测试前检查服务存活，服务异常退出或超时会打印日志并返回失败。

## 构建产物和证据路径

最终已验证 wheel：

```text
/public/home/tangyu408/Qwen_DCU_Worker_0/tp2_batch10_optimization_20260813/full-final/wheel-final4096/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl
SHA-256 af15f45a82356d95c8d2aea34008e85d93c2690392720e1f949132d9757ad2c0
```

主要运行证据：

```text
/public/home/tangyu408/Qwen_DCU_Worker_0/tp2_batch10_optimization_20260813/full-final/
  throughput/c10-rinf/
  mixed-safety/
  accuracy-client.log
  wheel-final4096/
```

模型权重、测试数据、wheel 和大体积运行日志不纳入源码提交。
