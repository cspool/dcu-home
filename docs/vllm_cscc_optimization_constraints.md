# vllm_cscc 优化执行和测试约束

本文档集中保存执行、测试、正确性和合规边界。所有阶段性结论必须同时满足这里的约束，以及固定脚本产物中的证据。

## 索引

| ID | 主题 |
| --- | --- |
| C0 | 赛题口径 |
| C1 | 固定实验契约 |
| C2 | 允许与禁止 |
| C3 | 测量协议 |
| C4 | 正确性门槛 |
| C5 | Profiler 规范 |
| C6 | 源码边界 |
| C7 | Evidence Card 模板 |

## C0 赛题口径

官方目标是在单卡、并发数固定为 1 的在线服务负载下，在 TTFT P99 与 TPOT P99 满足 SLA 的硬约束前提下，最大化输出吞吐量，并改善显存使用效率与服务稳定性。

主指标：

- `output_throughput`，即 Output Tokens / Second。
- 输入 Prompt Token 不计入吞吐分子。
- `total_token_throughput` 只能作为诊断参考，不能作为优化主结论。

SLA 硬约束：

- 每个输入长度档位分别判断 `TTFT P99 <= baseline TTFT P99 * 1.5`。
- `TPOT P99 <= baseline TPOT P99 * 1.5`，TPOT P99 按全部请求汇总后的全局池计算。
- 服务完成率下降超过 1% 时，该档位吞吐得分清零；本地验收采用更严格的 `failed=0`。

评分权重：

- `4K-8K`：20%。
- `8K-16K`：50%。
- `16K-32K`：30%。
- 单档相对提升率：`(candidate output_throughput - baseline output_throughput) / baseline output_throughput`。
- 单档得分按赛题公式计算：`该档满分 * (60% + 40% * (1 - e^(-1.3 * 单档相对提升率)))`。

精度约束：

- 温度固定为 `0.0`。
- 官方精度评估使用 OpenCompass，对问答、摘要、检索、聚合四类任务分别计算相对 baseline 精度下降。
- 本地 token/text 哈希一致只能作为快速正确性门槛，不能替代最终 OpenCompass 精度系数。

精度系数按官方技术方案计算。对每类任务先计算
`Δ = (baseline 精度 - candidate 精度) / baseline 精度 × 100%`，再查表得到
该类任务的 `k_i`：

| 单类精度下降 `Δ` | `k_i` |
| --- | ---: |
| `Δ <= 1.0%` | `1.00` |
| `1.0% < Δ <= 2.0%` | `0.97` |
| `2.0% < Δ <= 3.0%` | `0.94` |
| `3.0% < Δ <= 5.0%` | `0.90` |
| `5.0% < Δ <= 10.0%` | `0.85` |
| `Δ > 10.0%` | `0` |

四类任务的最终精度系数为 `K = (k_1 + k_2 + k_3 + k_4) / 4`，最终得分为
`吞吐量公式分 × K`。权威来源为
[赛题技术方案](../../docs/2026年全国大学生计算机系统能力大赛-智能计算创新设计赛-基于国产加速卡的千问大模型推理服务优化-技术方案/2026年全国大学生计算机系统能力大赛-智能计算创新设计赛-基于国产加速卡的千问大模型推理服务优化-技术方案.md)。

## C1 固定实验契约

固定测试集：

- `remote-home/testdata/4-8K_throughput.jsonl`
- `remote-home/testdata/8-16K_throughput.jsonl`
- `remote-home/testdata/16-32K_throughput.jsonl`

固定服务边界：

- 统一模型权重、官方 tokenizer、官方 chat template、官方 bf16 原始权重、vLLM 0.18.1、统一 OpenAI 兼容服务接口。
- 不修改模型结构、权重文件、tokenizer、chat template、请求/响应格式、OpenAI API 路径或评测结果解析口径。
- 评测运行不得依赖宿主机额外库、外网下载或未随提交说明的第三方组件。
- `start_vllm.sh` 可以做最小改动，但只允许打开新 wheel 中新增的源码特性或诊断日志；host/port/route/served model 等服务接口字段以评测平台为准。
- 官方锁定参数不得改动：`max_tokens`、`temperature=0`、`max-model-len`、`max-num-seqs`、`max-num-batched-tokens` 以及其它影响任务定义、上下文范围、输出行为或 batch scheduler 的参数。
- 严禁通过修改 `vllm/config/scheduler.py`、`vllm/v1/core/sched/` 等 batch scheduler 相关源码获取收益。这些文件只允许只读审计和非性能验收插桩。

固定晋级、baseline 与最终计分吞吐测试命令：

```bash
cd /data3/Projects/scnet_ssh/remote-home/testdata
env -u http_proxy \
    -u https_proxy \
    -u HTTP_PROXY \
    -u HTTPS_PROXY \
    -u MODEL_DIR \
    -u SERVED_MODEL_NAME \
    -u VLLM_HOST \
    -u VLLM_PORT \
    -u MAX_CONCURRENCY \
    -u REQUEST_RATE \
    -u CUSTOM_OUTPUT_LEN \
    -u NUM_WARMUPS \
    NO_PROXY=127.0.0.1,localhost \
    no_proxy=127.0.0.1,localhost \
    ./run_throughput.sh all
```

吞吐规则：

- `run_throughput.sh` 不允许修改。
- baseline、晋级验证和最终计分证据禁止传第二参数限制 `NUM_PROMPTS`，即禁止
  `./run_throughput.sh all 1` 这类缩短测试。源码候选的非计分小样本筛选可固定
  使用 `./run_throughput.sh all 3`，但除报错外其外层计时必须自然运行或等待至
  至少 600 秒，且该结果只能决定是否晋级，不能计入性能结论；晋级候选仍须按
  C3 使用无第二参数的完整 `all` 重跑三次。
- R27 起，standalone kernel/backend 的任何性能比较也必须在同一个随机交错
  窗口累计至少 600 秒；“短诊断”“autotune screen”或少量 event repeats 只要
  被用于比较快慢，就属于性能测试。少于 600 秒的百分比一律标为
  `protocol-invalid`，不能用于选 config、晋级或性能结论。parse/compile、
  code-object 资源、数值/repeat、静态 ISA、一次性路径/PMC 捕获属于前置门，
  可在明确失败时提前停止，但不得夹带短时性能排序。
- `MAX_CONCURRENCY`、`REQUEST_RATE`、`CUSTOM_OUTPUT_LEN`、`NUM_WARMUPS` 必须使用脚本默认值：`1`、`1`、`1024`、`2`。
- `RESULT_ROOT` 只允许用于区分结果落盘目录，不允许影响 workload 或筛选样本。
- 远程容器可能预置 `http_proxy`/`https_proxy`。访问 `127.0.0.1:8001` 时必须绕过代理，否则 `curl` 或 `vllm bench serve` 可能请求到 Squid 代理并返回 HTML/503，而不是 vLLM API。`NO_PROXY/no_proxy=127.0.0.1,localhost` 只用于修正本机回环 API 路由，不改变模型、请求、并发、输出长度、采样或评分口径。

API 连通性检查必须使用绕过代理的方式：

```bash
curl --noproxy 127.0.0.1,localhost \
    http://127.0.0.1:8001/v1/models

curl --noproxy 127.0.0.1,localhost \
    http://127.0.0.1:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen3.5-27B","messages":[{"role":"user","content":"你好,简单回复一句话。"}],"temperature":0.0,"max_tokens":64}'
```

若未绕过代理时出现 Squid HTML 错误页或 `Service Unavailable`，该轮 API/吞吐结果判为环境访问无效，不能写入 baseline 或优化结论。

Accuracy 规则：

- 固定使用 `run_accuracy.sh all`。
- 禁止传第二参数缩短行数。
- 需要 DTK 环境时，只允许用 `source /opt/dtk-26.04-DCC2602-0317/env.sh` 修正动态库路径。
- 访问本机 vLLM API 时保持 no-proxy。
- 最终表以 `run_accuracy.sh` 输出为准；OpenCompass 原始 summary 中 RULER 聚合任务可能未按脚本重算，不能替代最终表。

## C2 允许与禁止

允许的优化方向：

- 修改 `remote-home/vllm_cscc` 源码并重新编译 wheel。
- 编译 custom kernel，前提是源码、构建脚本和依赖说明完整，能在统一容器内复现。
- KV Cache 分配机制、显存预算内部实现、块管理策略和 cache layout 的源码级优化，但不能改变锁定的上下文、batch scheduler 参数或请求语义。
- Attention、Linear、GDN/linear attention、KV write/update 等执行路径与算子优化。
- 非持久化、推理过程内的算子级低精度计算优化，例如激活动态量化、KV Cache 量化、kernel 内部临时类型转换、低精度矩阵乘法；必须通过精度门槛，且不能生成可复用量化权重或压缩缓存。

禁止的行为：

- 调低 max tokens、截断输入、跳过长样本、过滤困难样本、跳过层、跳过 head、token pruning、early-exit。
- 预缓存测试集、答案、prompt 特征或预生成中间结果。
- 任何形式的 speculative decoding、draft model、MTP、多头预测、外挂小模型、自训练预测器或预生成 token 缓存。
- 不启用 prefix cache 或跨请求 prefix/中间状态复用；源码图中的 `prefix_prefill.py` 只作为可能执行路径审计对象。
- 后训练、蒸馏、微调、替换权重、结构化/非结构化剪枝、动态通道跳过。
- 权重加载前后、服务初始化阶段或正式推理前的持久化量化、权重重排压缩、模型格式转换、生成可复用量化权重缓存或压缩权重缓存。
- 通过调参、修改评测脚本、修改服务接口或绕开官方资源统计路径获得收益。

## C3 测量协议

每个晋级候选或进入性能结论的候选必须完成同一闭环：

1. 记录 baseline wheel 文件名、mtime、`pip show vllm`、`vllm --version` 或等价来源信息。
2. 记录源码 diff，只允许 `remote-home/vllm_cscc` 内源码和必要的 `start_vllm.sh` 特性开关改动。
3. 在目标远程容器中重新编译 wheel，保存构建命令、构建日志、新 wheel 文件名和 mtime。修改 Python 源码后必须刷新 `build/lib` 产物，例如先执行 `python3 setup.py build_py --force` 再执行 `python3 setup.py bdist_wheel`；否则 `bdist_wheel` 可能复用旧 Python 文件，导致 wheel 未包含源码 diff。
4. 安装或加载新 wheel 后，确认 `vllm` 命令实际来自新 wheel，并从非源码目录（例如 `/tmp`）导入 `vllm` 检查 site-packages 文件路径和源码 marker。若 installed marker 与源码 diff 不一致，该 wheel/test 视为无效，必须重新强制构建和安装。
5. 使用受控 `start_vllm.sh` 启动服务；若脚本有 diff，记录每个新增参数命中的源码路径与环境变量说明。
6. 启动后先用 `curl --noproxy 127.0.0.1,localhost` 检查 `/v1/models` 和 `/v1/chat/completions`，确认返回 vLLM JSON 且模型名为 `Qwen3.5-27B`。
7. 使用固定命令运行 `run_throughput.sh all`，三档各完整 50 条；运行环境必须保留 `NO_PROXY/no_proxy=127.0.0.1,localhost` 或等价地 unset `http_proxy/https_proxy`，避免 benchmark 客户端走代理。
8. 最终候选至少重复 3 次，做逐请求 paired comparison。
9. 汇总 `output_throughput`、`request_throughput`、`p99_ttft_ms`、全局 `p99_tpot_ms`、`mean_ttft_ms`、`mean_tpot_ms`、`mean_e2el_ms`、完成率和方差/置信区间。
10. 用每档 `output_throughput` 计算相对 baseline 提升，并按 20/50/30 权重汇总。
11. 对比 completed/failed、输出 token 数、finish reason、stop reason、输出 token ids 或文本哈希。
12. 候选筛选可使用等价精度审计；所有进入最终结论的候选必须执行固定 `run_accuracy.sh`/OpenCompass 精度评估，并报告四类任务相对 baseline 精度下降与官方精度系数。

## C4 正确性门槛

- 本地吞吐验收 `failed=0`。
- deterministic 输出不得发生非预期变化。
- 若输出 token 数、finish reason、stop reason 或输出哈希不一致，该请求的性能数据不能直接合并进收益结论，除非 OpenCompass 证明精度仍满足赛题系数要求。
- 若 benchmark 产物没有 token ids，可用独立只读分析脚本或固定 API audit 采集哈希；不得修改 `run_throughput.sh`。

## C5 Profiler 规范

- 必须捕获 EngineCore worker 的 HIP/HIPTX kernel timeline。
- 同步插桩 latency 只用于归因，不作为最终无扰动吞吐证据。
- kernel 证据至少包含 kernel name、调用次数、总耗时、平均耗时、shape 或 token/chunk 维度。
- DCU hardware counter 证据优先包含达成带宽、`MemUnitBusy`、`MemUnitStalled`、`VALUBusy`、VGPR/LDS 或 occupancy；至少要能把热点归入 memory-bound、compute-bound、latency-bound 或 launch-bound。
- 先做路径确认，再做源码优化；不允许只凭配置名推断 backend。

## C6 源码边界

只读锁定与审计：

- `remote-home/testdata/run_throughput.sh`
- `remote-home/testdata/run_accuracy.sh`
- `remote-home/testdata/start_vllm.sh`
- `vllm/config/scheduler.py`
- `vllm/v1/core/sched/`
- 模型权重、tokenizer、chat template、评测结果解析脚本

配置与 backend 选择：

- `vllm/envs.py`
- `vllm/config/attention.py`
- `vllm/config/cache.py`
- `vllm/config/compilation.py`
- `vllm/platforms/rocm.py`
- `vllm/_aiter_ops.py`

Attention、KV 与显存管理：

- `vllm/v1/attention/selector.py`
- `vllm/v1/attention/backends/registry.py`
- `vllm/v1/attention/backends/flash_attn.py`
- `vllm/v1/attention/backends/rocm_attn.py`
- `vllm/v1/attention/backends/rocm_aiter_fa.py`
- `vllm/v1/attention/backends/rocm_aiter_unified_attn.py`
- `vllm/v1/attention/backends/gdn_attn.py`
- `vllm/v1/attention/ops/prefix_prefill.py`
- `vllm/v1/attention/ops/chunked_prefill_paged_decode.py`
- `vllm/v1/attention/ops/paged_attn.py`
- `vllm/v1/attention/ops/triton_prefill_attention.py`
- `vllm/v1/attention/ops/triton_decode_attention.py`
- `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/single_type_kv_cache_manager.py`
- `vllm/v1/core/block_pool.py`
- `vllm/v1/core/kv_cache_utils.py`
- `vllm/v1/kv_cache_interface.py`
- `vllm/v1/worker/block_table.py`
- `vllm/v1/worker/gpu/block_table.py`
- `csrc/rocm/attention.cu`
- `csrc/cache_kernels.cu`
- `csrc/cache_kernels_fused.cu`

Fusion、GDN、linear 与运行时量化：

- `vllm/compilation/passes/fusion/qk_norm_rope_fusion.py`
- `vllm/compilation/passes/pass_manager.py`
- `vllm/compilation/passes/utility/fix_functionalization.py`
- `csrc/fused_qknorm_rope_kernel.cu`
- `csrc/torch_bindings.cpp`
- `csrc/ops.h`
- `vllm/model_executor/models/qwen3_5.py`
- `vllm/model_executor/models/qwen3_next.py`
- `vllm/model_executor/layers/fla/ops/`
- `vllm/model_executor/layers/linear.py`
- `vllm/model_executor/kernels/linear/`
- `vllm/model_executor/layers/quantization/kv_cache.py`
- `vllm/utils/torch_utils.py`

非 batch-scheduler runtime：

- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/gpu/model_runner.py`
- `vllm/v1/worker/gpu_input_batch.py`
- `vllm/v1/worker/workspace.py`

## C7 Evidence Card 模板

每个候选结论必须单独填写：

- 候选 ID：例如 H1.1。
- 变更摘要：源码文件、启动脚本 diff，如有。
- 构建摘要：baseline wheel、新 wheel、构建命令、安装来源。
- 环境变量说明：新增变量名、取值、作用、必要性；无新增则写明无。
- 固定实验命令：启动命令、吞吐命令、环境变量。
- 路径证据：日志、profiler 或插桩证明命中目标 backend。
- 瓶颈归因：memory-bound、compute-bound、launch-bound 或 Python overhead-bound。
- 性能表：三档 `output_throughput`、`p99_ttft_ms`、全局 `p99_tpot_ms`、`mean_ttft_ms`、`mean_tpot_ms`、`mean_e2el_ms`、完成率、3 次重复统计。
- 评分表：三档相对 baseline 提升、20/50/30 加权结果、SLA 是否熔断。
- 精度表：筛选阶段可记录等价精度审计；最终结论必须记录固定 `run_accuracy.sh`/OpenCompass 结果、四类任务精度下降与官方精度系数。
- 合规说明：排除调参、脚本改动、batch scheduler 改动、持久化量化、模型语义改变和数据集过拟合。
- 结论：保留、回滚或继续深入。

只有满足上述证据链的结果才能写入最终优化结论。
