# Qwen 初赛与决赛题目简述

## 共同目标

在国产 DCU 上使用 vLLM 0.18.1 部署 Qwen3.5-27B（BF16），通过统一的 OpenAI 兼容接口完成长上下文在线推理。在保证模型精度、请求完成率以及 TTFT/TPOT 长尾时延约束的前提下，尽可能提高输出吞吐量（output token/s）。

性能评测包含 4–8K、8–16K 和 16–32K 三个输入长度区间。**三个区间是三个相互独立的 benchmark，而不是将不同长度请求混合在同一条请求流中测试。** 每档分别配置吞吐测试参数、运行压测、保存结果并统计吞吐与时延。精度测试覆盖问答、摘要、检索和聚合四类任务。模型权重、tokenizer、chat template、请求内容、输出长度和采样语义等任务定义保持不变。

## 初赛题目

初赛采用单卡 DCU。三个长度区间分别独立评测，但各档并发数均固定为 1。参赛者主要从 vLLM 源码、执行路径和 DCU 算子层面优化，包括 KV Cache 与显存管理、PagedAttention、Attention/Linear 内核、算子融合、kernel launch 及数据搬运等。

初赛的核心特点是：服务端与压测端关键参数基本锁定，重点考察在固定负载下的实现优化。三档吞吐得分权重分别为 20%、50% 和 30%；任一档 TTFT P99 或全局 TPOT P99 超过对应 baseline 的 1.5 倍时，该档吞吐得分清零，最终成绩还会乘以四类任务的精度系数。

## 决赛题目

决赛改为双卡 DCU，并在初赛优化范围上增加“并行拓扑与负载配置联合搜索”。当前 baseline 位于 `/public/home/tangyu408/cscc-testdata-20260812`，默认配置为：

- TP=2、DP=1，`max-model-len=32768`；
- `max-num-seqs=128`、`max-num-batched-tokens=4096`、`gpu-memory-utilization=0.95`；
- baseline 分别运行三个长度 benchmark，但三档默认值恰好都设为并发 10、`request-rate=inf`、最大输出长度 1024；
- 吞吐数据量依次为 80、70 和 60 条请求。

决赛扩大后的搜索空间包括：

1. **TP/DP 拓扑可通过环境变量选择**：重点比较双卡下的 TP=2、DP=1 与 TP=1、DP=2。三个长度档是独立 benchmark，因此可为每档重新启动服务并选择该档拓扑。当前原始 baseline 仍硬编码 TP=2；环境变量化属于决赛扩大搜索空间后的启动接口。
2. **服务端并发容量参数解锁**：`max-num-seqs`、`max-num-batched-tokens` 和 `gpu-memory-utilization` 改为环境变量，可针对每个独立长度 benchmark 分别调优。baseline 的 128、4096 和 0.95 是默认起点，不是决赛必须保持的值。`max-model-len=32768` 仍固定，不能借此缩短输入或减少实际工作量。
3. **客户端并发按长度独立设置**：4–8K、8–16K、16–32K 各自拥有独立的 `max-concurrency`，不再共享决赛原始脚本中的单个全局 `CONCURRENCY`。例如分别使用 `CONCURRENCY_4_8K`、`CONCURRENCY_8_16K` 和 `CONCURRENCY_16_32K`。`request-rate=inf` 作为决赛到达模型保持固定。
4. **按档独立执行与统计**：每档使用自己的数据集、服务端并发配置、客户端并发配置和结果目录，依次完成“启动服务—运行该档压测—保存该档结果”。三档请求不混跑，各自生成吞吐和 TTFT/TPOT 结果；若计分规则要求汇总某项统计量，只在计分阶段处理独立结果，不改变三档独立 benchmark 的性质。
5. **精度和任务语义参数固定**：模型权重与结构、tokenizer、BF16 dtype、chat template、temperature=0、输出长度 1024、请求内容、停止条件、服务接口、精度数据集、OpenCompass 配置、后处理和精度系数计算均不得修改。

因此，决赛的核心问题可概括为：对每个长度 benchmark，分别在两张 DCU 的资源约束下选择 **并行拓扑（TP/DP）× 服务端并发容量 × 客户端并发负载**，并结合合法的源码、KV Cache 与算子优化，在固定精度语义、服务稳定性和时延达标的条件下最大化该档输出吞吐量。三个 benchmark 分别调优和评测，最后再按比赛规则汇总成绩。

### 初赛参数限制与决赛脚本变化

初赛技术方案明确锁定模型与分词、上下文和 batch scheduler、生成参数、服务接口以及 benchmark 统计口径。下面把初赛脚本快照与决赛 baseline 脚本逐项对照：

| 类别 | 参数 | 初赛 | 决赛 baseline | 判断 |
|---|---|---|---|---|
| 硬件/拓扑 | TP、DP | 单卡，TP=1、DP=1 | 双卡，TP=2、DP=1 | **已改变**；决赛新增的服务端搜索重点是 TP/DP 拓扑 |
| 模型与接口 | model、tokenizer、served model、API 路由、BF16、chat template | 固定 | 与初赛相同 | **继续锁定** |
| 上下文 | `max-model-len` | 规则固定为 32768 | 脚本显式设置 32768 | **继续锁定** |
| 调度容量 | `max-num-seqs` | 固定为 128，规则明确禁止修改 | baseline 默认值仍为 128 | **决赛改变：作为并发容量参数解锁，可通过环境变量搜索** |
| 调度容量 | `max-num-batched-tokens` | 固定为 4096，规则明确禁止修改 | baseline 默认值仍为 4096 | **决赛改变：作为批处理容量参数解锁，可通过环境变量搜索** |
| 其他服务配置 | `gpu-memory-utilization` | 脚本为 0.95 | baseline 默认值仍为 0.95 | **决赛改变：作为显存/并发容量参数解锁** |
| 客户端负载 | `max-concurrency` | 固定为 1 | 环境变量 `CONCURRENCY`，默认 10 | **已改变**；扩大为多请求负载，增强版再按长度拆成三个变量 |
| 到达模型 | `request-rate` | 固定为 1 | 固定为 `inf` | **已改变为新的固定负载口径**，不等于允许任意搜索 |
| 生成与精度语义 | dtype、`temperature`、`max_tokens`/输出长度 | BF16、temperature=0，输出长度固定 | BF16、temperature=0，`custom-output-len=1024` | **继续锁定** |
| 数据顺序 | dataset、完整请求数、`no-oversample`、`disable-shuffle` | 固定 | 保持同类设置，三档分别运行 | **继续锁定**；减少请求数仅用于本地调试，不用于正式计分 |
| 统计口径 | warmup、percentile、结果解析 | 固定 | 仍为 2 个 warmup，统计 P50/P95/P99 等 | **继续锁定** |

需要特别区分“baseline 默认值”和“决赛搜索权限”：决赛 baseline 仍使用 `max-num-seqs=128`、`max-num-batched-tokens=4096` 和 `gpu-memory-utilization=0.95`，这只表示默认起点没有变化。决赛扩大搜索空间后，并发和资源容量相关参数改为环境变量，因此上述参数与 TP/DP、客户端并发一起解锁；模型精度和任务语义相关参数则继续固定。

`max-num-seqs` 和 `max-num-batched-tokens` 分别限制单次调度的序列数和 token 预算，并不直接规定单请求输出长度。输出长度仍由固定的请求输出上限、EOS/停止条件及 `max-model-len` 共同决定。因此，决赛可以开放这两个并发调度参数，同时继续锁定精度与生成语义；但因错误配置造成请求失败、超时或输出变化，仍须接受完成率、SLA 和精度门禁。

## 计分公式

对长度档位 $i\in\{\mathrm{4\text{-}8K},\mathrm{8\text{-}16K},\mathrm{16\text{-}32K}\}$，令 $T_i$、$B_i$ 分别为参赛方案与 baseline 的输出吞吐量，三档满分 $M_i$ 依次为 20、50、30。

$$
r_i = \frac{T_i - B_i}{B_i}
$$

$$
S_i = M_i\left[0.60 + 0.40\left(1-e^{-1.3r_i}\right)\right]
$$

$r_i$ 按小数代入，例如提升 20% 时 $r_i=0.20$。仅在该档通过 SLA 时使用上述得分，否则 $S_i=0$。

$$
\mathrm{TTFT}_{P99,i} \le 1.5 \times \mathrm{TTFT}^{\mathrm{baseline}}_{P99,i}
\qquad
\mathrm{TPOT}_{P99,\mathrm{global}} \le 1.5 \times \mathrm{TPOT}^{\mathrm{baseline}}_{P99,\mathrm{global}}
$$

TTFT 按档判断，TPOT 汇总后取全局 P99；服务完成率下降超过 1% 时，相应档位同样清零。四类精度任务中，第 $j$ 类的相对精度下降为：

$$
\Delta_j = \frac{A^{\mathrm{baseline}}_j-A_j}{A^{\mathrm{baseline}}_j}\times 100\%
$$

| 精度下降 $\Delta_j$ | 系数 $k_j$ |
|---|---:|
| $\Delta_j\le 1\%$ | 1.00 |
| $1\%<\Delta_j\le 2\%$ | 0.97 |
| $2\%<\Delta_j\le 3\%$ | 0.94 |
| $3\%<\Delta_j\le 5\%$ | 0.90 |
| $5\%<\Delta_j\le 10\%$ | 0.85 |
| $\Delta_j>10\%$ | 0 |

$$
S_{\mathrm{final}}
= \left(\sum_i S_i\right)\left(\frac{1}{4}\sum_{j=1}^{4}k_j\right)
$$

若决赛未公布新公式，则沿用上述口径；否则以决赛规则与实际评测脚本为准。

## 主要区别

| 项目 | 初赛 | 决赛 |
|---|---|---|
| 硬件 | 单卡 DCU | 双卡 DCU |
| Benchmark 组织 | 三个长度档独立运行 | 三个长度档仍独立运行，禁止混成一个请求流 |
| 客户端负载 | 各档并发均固定为 1，`request-rate=1` | 各档分别配置并发，`request-rate=inf` |
| 并行拓扑 | TP=1、DP=1 | 重点搜索 TP=2/DP=1 与 TP=1/DP=2 |
| 服务端并发容量 | `max-num-seqs=128`、`max-num-batched-tokens=4096` 固定 | 默认值不变，但参数解锁并可按档搜索 |
| 精度相关参数 | 模型、BF16、tokenizer、chat template、生成语义和精度脚本固定 | 全部继续固定 |
| 其他测试参数 | 数据、输出长度、到达模型、接口和统计口径固定 | 除分档并发外继续固定 |
| 优化重点 | 源码、调度、KV Cache 与算子优化 | 初赛优化 + 双卡拓扑和负载联合调优 |
| 最终目标 | 分别提高三个固定负载的合规吞吐 | 分别寻找三个 benchmark 的最优双卡配置和合规吞吐 |

无论初赛还是决赛，均不得通过修改模型权重或结构、降低精度、截断输入、跳过样本、减少规定输出、缓存测试答案或更改评测统计逻辑来获得成绩。决赛环境变量搜索用于并行拓扑、服务端并发容量和客户端并发负载，不用于改变精度或任务定义。

对比依据：初赛技术方案与脚本快照；决赛 `start_vllm.sh` 和 `run_throughput.sh`。
