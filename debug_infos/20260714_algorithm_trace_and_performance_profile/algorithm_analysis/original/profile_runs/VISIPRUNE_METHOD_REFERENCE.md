# VisiPrune Workload Analysis

## `torch.profiler` 与当前 `workload_analysis` 的区别

`torch.profiler` 和当前 `workload_analysis` 都可以观察一次真实运行，但它们的目标不同。

- `torch.profiler` 面向性能分析。
- `workload_analysis` 面向算法执行理解、动态 workload 建模和 layer process 证据重建。

因此，两者记录的“op flow”不能按同一种语义解释。

## `torch.profiler`: 性能事件 profiler

`torch.profiler` 的主要目标是回答：

```text
一次真实运行中，哪些 PyTorch op / CUDA kernel / CPU activity 消耗了时间和显存？
```

它关注的数据通常包括：

- PyTorch op 名称
- CPU time / CUDA time
- CUDA kernel timeline
- op 调用次数
- memory allocation
- tensor shape
- Python stack
- Chrome trace

启用示例：

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    model.generate(...)
```

`torch.profiler` 的 shape、时间和显存数据对性能分析重要，但它不天然理解
VisiPrune 的算法语义。例如，它不会自动提供：

- `forward_id`
- prefill / decode phase
- 每层 `q_len / kv_len / past_len`
- VisiPrune token selection 事件
- deep exit 事件
- layer 在 pruning schedule 中的角色
- token 数变化边界，例如 `624 -> 58 -> 48`

`torch.profiler` 也不是严格的数据依赖图工具。它可以显示真实运行中发生过哪些
op，并提供时间线和统计表，但它的核心目标不是把这些 op 重建成一个可审计的
layer tensor process。

`with_flops=True` 只能对部分算子，例如 matmul / conv，给出有限 FLOPs 估计。
它不会自动推导 VisiPrune 这种动态 token pruning 算法的理论复杂度。

### `with_stack=True` / `with_modules=True` 的 eager 实测含义

在当前 eager VisiPrune 路径中，已经运行过一次真实 GPU `torch.profiler`
实验，并同时开启：

```text
record_shapes=True
with_stack=True
with_modules=True
```

该次运行捕获到：

```text
真实 profiler events: 58269
FunctionEvent.stack 非空行数: 0
module_hierarchy 非空行数: 0
```

这说明两点：

- profiler 确实观察到了真实执行的 CPU / CUDA / ATen event flow。
- 但本次 eager 路径下，自动 Python stack 和 module hierarchy 元数据没有被有效填充。

因此，`with_stack=True` 和 `with_modules=True` 在这里不能被理解为“已经获得了
源码级或模块级算法 trace”。它们只是请求 profiler 尝试记录这些归因信息；对于
普通 eager model，尤其是复杂 `generate()` 路径，这些字段可能为空或不稳定。

这也是 `torch.profiler` 不适合作为依赖分析主工具的直接原因之一。依赖分析需要的
不是简单的事件列表，而是可重建数据流的元数据，例如：

- op 的输入 tensor id
- op 的输出 tensor id
- producer-consumer edge
- view / alias / storage 关系
- inplace mutation 证据
- layer / phase / schedule 语义标签

`torch.profiler` 主要回答：

```text
哪些事件发生了，什么时候发生，用了多久，输入 shape 大概是什么？
```

它通常不直接回答：

```text
哪个 op 产出的 tensor 被后续哪个 op 消费？
这个 view / reshape / slice 是否共享 storage？
这个 inplace op 修改了哪条后续数据路径？
这个 op 属于 VisiPrune 的哪个 layer、phase 或 selection process？
```

即使某些环境下 `with_stack` 或 `with_modules` 能产生非空结果，它们也主要提供
源码归因和模块归因，不等价于 tensor 级 producer-consumer 依赖。因此它可以辅助
定位热点和粗粒度执行区域，但不能直接替代 filtered dispatch profile 或 FX /
reconstruction 流程做依赖重建。

因此，`torch.profiler` 适合做：

- 新算法早期的通用热点初筛
- CPU/CUDA 时间线观察
- shape / memory / stack 辅助定位
- 判断哪些区域值得进一步做 nsys/ncu 或 dispatch 取证

但它不适合作为当前项目的算法 trace 或 layer process 重建主证据。

## `workload_analysis`: 算法行为与证据重建

当前 `workload_analysis` 的产物按工作顺序分为四类：

1. `torch_profile`: 最早的通用 profiler 初筛产物，用于观察真实运行的事件、shape、时间线和 profiler 元数据质量。
2. `algorithmic_trace`: 先做的算法 schedule / 理论 workload 产物，用于确定 VisiPrune 真实动态路径。
3. `dispatch`: 后续 layer process 取证的一种方法，使用真实 eager TorchDispatch op log 和 tensor id 做运行时证据重建。
4. `fx`: 后续 layer process 取证的另一种方法，使用同一批目标 layer 的固定输入 replay 生成 FX GraphModule / DAG。

`dispatch` 和 `fx` 是同一步目标的两条方法线：都从 `algorithmic_trace` 选出的重要 layer/forward 事件出发，但证据形态不同。

### `algorithmic_trace`

`algorithmic_trace` 的目标是回答：

```text
VisiPrune 在一次真实 generate() 中如何改变 token schedule 和理论 workload？
```

它通过 wrapper / hook 记录上层动态执行 flow，包括：

- `forward_id`
- prefill / decode phase
- 每层 `q_len`
- 每层 `kv_len`
- 每层 `past_len`
- hidden state shape
- VisiPrune selection 事件
- deep exit 事件
- 每层动态 token schedule
- 理论 FLOPs

重要输出包括：

```text
algorithmic_trace.json
layer_trace.csv
selection_trace.csv
operator_flops.csv
```

这里的重点不是 wall-clock latency，也不是 CUDA kernel timeline，而是：

```text
算法在真实请求中走了哪条动态路径，以及这条路径对应什么理论 workload。
```

因此，`algorithmic_trace` 是 VisiPrune 动态 schedule 的权威来源。

### Filtered Dispatch Profile

filtered dispatch profile 的目标是回答：

```text
在选中的 layer / forward 事件内，真实 eager 执行时发生了哪些 ATen op？
这些 op 的 tensor shape、数据依赖、alias、inplace 行为是什么？
```

它使用：

```text
torch.utils._python_dispatch.TorchDispatchMode
__torch_dispatch__
```

这不是编译时 trace，也不是 `torch.compile` / FX / export IR。它是运行时 eager
执行过程中观察到的 ATen dispatch op 流。

它记录的数据包括：

- selected `event_id`
- `forward_id`
- `layer_id`
- prefill / decode phase
- `q_len / kv_len / past_len`
- op schema
- input tensor ids
- output tensor ids
- tensor shape / dtype / device
- alias / storage / inplace mutation 信息
- sampled module stack
- op count summary

重要输出包括：

```text
dispatch_manifest.csv
dispatch_ops.csv
dispatch_op_summary.csv
observed_layer_events.csv
run_metadata.json
```

其中：

- `dispatch_manifest.csv` 说明为什么选这些 layer / forward 事件。
- `dispatch_ops.csv` 是选中 layer 内真实 ATen dispatch op 的主要证据。
- `observed_layer_events.csv` 用于校验全局 layer 事件编号，不代表全量 dispatch profile。

filtered dispatch profile 的重点不是执行时间，而是：

```text
用真实运行时 op、tensor id、shape、alias 和 inplace 证据，反推出 layer 的实际 tensor process。
```

这些数据后续可以被 `dispatch-layer-reconstruct-onnx` 等流程消费，用来生成更可读的
process 表达、small-shape Torch flow 或 ONNX stage。

### FX Trace 与 Process Reconstruction

FX 路线的目标是回答：

```text
对同一批选中 layer / forward 事件，能否得到固定输入下的 ATen GraphModule DAG，
并把该 DAG 重建成可读 process？
```

它先在真实 eager `generate()` 中采样目标 layer 的输入，然后在请求结束后离线 replay
这些输入并运行 `make_fx(...)`。因此 FX 产物不是运行时 op log，也不是 CUDA/kernel
profile；它是一个固定输入路径上的低层 ATen DAG。

当前主要产物位于：

```text
fx/traces/fx_filtered_dispatch_layers_specialized/
```

每个目标 event 目录包含：

```text
fx_graph.py
fx_graph.txt
fx_graph_module.pt
fx_nodes.json
fx_process_nodes.csv
fx_process_reconstruction.json
fx_process_reconstruction.md
fx_trace_metadata.json
```

其中 `fx_process_reconstruction.*` 是基于 FX DAG 规则重建出的 readable process
标签，不是 PyTorch FX 官方语义，也不是运行时模块归属证明。

## 关键区别

| 维度 | `torch.profiler` | `workload_analysis` |
| --- | --- | --- |
| 主要目标 | 性能分析 | 算法理解与执行证据重建 |
| 运行方式 | profile 真实运行的 op/kernel/activity | wrapper 记录算法 schedule，dispatch mode 记录选中 layer ATen op |
| 时间数据 | 核心数据 | 非核心数据 |
| shape 数据 | 性能辅助信息 | process 重建证据 |
| FLOPs | 有限 op 级估计 | 基于算法 schedule 的理论 FLOPs |
| layer 语义 | 不天然提供 | 显式记录 `forward_id/layer_id/phase/q_len/kv_len` |
| VisiPrune selection | 不天然提供 | 显式记录 |
| 数据依赖 | 不是主要目标 | 通过 tensor ids / input-output ids / alias 信息重建 |
| 归因目标 | 找热点 | 解释 selected layer 实际执行过程 |
| 适合作为性能结论吗 | 可辅助，但本项目正式性能仍用 nsys/ncu | 不适合，主要不是性能工具 |

## 当前产物与 skill 使用记录

当前目录的使用顺序是：

```text
torch_profile 通用初筛
  -> algorithmic_trace 算法 schedule
  -> 从 layer_trace + selection_trace 选择 35 个目标 event
  -> dispatch 方法线 或 FX 方法线
```

### 0. `torch_profile`: 通用 profiler 初筛

对应步骤/skill：通用 `torch.profiler` 初筛；用 `$trace-patch-target-discovery`
确定需要显式记录的 request、forward、layer、attention、MLP、selection
边界和 join keys。

用来做什么：

- 先观察一次真实 `generate()` 的 CPU/CUDA/PyTorch profiler event flow。
- 验证 `with_stack=True`、`with_modules=True` 在当前 eager VisiPrune/LLaVA 路径下是否能提供有效 stack/module metadata。
- 生成 Chrome trace、key averages、record_function scopes 和 profiler-derived process sketch。
- 只作为早期通用分析和热点/shape/timeline 初筛，不作为算法 schedule 或 tensor 依赖主证据。

当前产物：

```text
torch_profile/traces/visipruner_full_1tok_stack_modules/
```

复现当前同类产物：

```bash
GPU=1 TOKENS=1 \
/workspace/VisiPrune/workload_analysis/torch_profile/runners/run_visipruner_full_profile.sh
```

关键输出包括 `metadata.json`、`chrome_trace.json`、`profiler_events.csv`、
`profiler_key_averages.csv`、`record_function_scopes.csv`、`layer_events.csv`、
`selection_events.csv` 和 `process_view.md`。

### 1. `algorithmic_trace`: 算法 schedule 与理论 workload

对应 skill：`$trace-patch-target-discovery` 先决定 wrapper/hook 边界；
`$visipruner-trace-dispatch-profile` 的 algorithmic trace 部分负责运行真实
`generate()` 并生成 fresh-forward trace。

用来做什么：

- 记录真实请求中的 `forward_id`、`phase`、`layer_id`、`q_len`、`past_len`、`kv_len`。
- 记录 VisiPrune middle selection、deep exit 和 token schedule 变化。
- 计算理论 FLOPs，并生成 dense-eager 对照。
- 为后续 dispatch / FX 两条方法线提供目标 event 选择依据。

当前产物：

```text
algorithmic_trace/traces/fresh_forward_visipruner_full_32tok/
algorithmic_trace/traces/fresh_forward_dense_eager_32tok/
algorithmic_trace/comparisons/fresh_visipruner_vs_dense_32tok.*
open_tool_dense_baseline/dense_baseline/
```

复现当前同类产物：

```bash
GPU=1 TOKENS=32 \
/workspace/VisiPrune/workload_analysis/algorithmic_trace/runners/run_full_forward.sh
```

关键输出包括 `algorithmic_trace.json`、`layer_trace.csv`、`selection_trace.csv`
和 `operator_flops.csv`。当前 VisiPrune trace 有 `1024` 个 layer events 和
`21` 个 selection events。

### 2. 目标 event 选择

对应 skill：`$visipruner-trace-dispatch-profile` 的 layer selection / filtered
dispatch 准备部分。

用来做什么：

- 从 `selection_trace.csv + layer_trace.csv` 选择与 VisiPrune 动态剪枝最相关的 layer。
- 优先保留 middle selection、deep exit、token 数变化边界、shallow 代表层和 decode cache regime 代表层。
- 为 dispatch 和 FX 两条后续方法线提供同一批目标 event，避免两条线分析对象不一致。

当前规则记录在：

```text
DISPATCH_FILTER_RULES.md
dispatch/profiles/filtered_dispatch_visipruner_full_32tok/dispatch_manifest.csv
fx/traces/fx_filtered_dispatch_layers_specialized/run_metadata.json
```

当前目标集是 `35` 个 event，包括 prefill 的 `input1_layer0`、`input1_layer5`、
`input1_layer6`、`input1_layer7` 到 `input1_layer28`，以及 decode 代表事件
`input2_layer18/19/27/28/31` 和 `input32_layer18/19/27/28/31`。

### 3A. `dispatch`: 真实 eager ATen/tensor-id 方法线

对应 skill 组：

- `$visipruner-trace-dispatch-profile`: 生成 filtered TorchDispatch profile。
- `$dispatch-layer-reconstruct-onnx`: 从 dispatch CSV 重建 per-layer process、small-shape Torch flow 和 ONNX stage。

用来做什么：

- 对 35 个目标 event 进入 `TorchDispatchMode`，捕获真实 eager ATen op log。
- 记录 op schema、shape、dtype、device、input/output tensor ids、alias、inplace 和 module stack。
- 用每个 layer 自己的 `dispatch_ops.csv` 生成 `dispatch_review/`、`torch_flow/` 和 `onnx/`。
- 作为“真实运行中发生了哪些 ATen op，以及 tensor-id 数据流如何连接”的主证据。

生成 filtered dispatch profile：

```bash
/workspace/VisiPrune/workload_analysis/env/run_with_analysis_env.sh \
  /workspace/VisiPrune/workload_analysis/dispatch/tools/visipruner_filtered_dispatch_profile.py \
  --gpu 1 \
  --tag filtered_dispatch_visipruner_full_32tok
```

当前产物：

```text
dispatch/profiles/filtered_dispatch_visipruner_full_32tok/
```

其中 `dispatch_manifest.csv` 有 `35` 个目标 event，`dispatch_ops.csv` 有
`3163` 条 ATen dispatch op 记录。

生成 dispatch reconstruction / ONNX：

```bash
python /workspace/VisiPrune/workload_analysis/dispatch/layer_pipeline/run.py \
  --source-csv /workspace/VisiPrune/workload_analysis/dispatch/profiles/filtered_dispatch_visipruner_full_32tok/dispatch_ops.csv \
  --out-dir /workspace/VisiPrune/workload_analysis/dispatch/visualize \
  --layers <dispatch_manifest.csv 中的 35 个 event_id>
```

当前产物：

```text
dispatch/visualize/<event_id>/dispatch_review/
dispatch/visualize/<event_id>/torch_flow/
dispatch/visualize/<event_id>/onnx/
dispatch/visualize/<event_id>/layer_manifest.json
```

完成后用下面脚本做审计：

```bash
python /workspace/VisiPrune/workload_analysis/dispatch/layer_pipeline/review_reconstruction.py
python /workspace/VisiPrune/workload_analysis/dispatch/layer_pipeline/audit_layer_reconstruction.py
```

### 3B. `fx`: 固定输入 FX DAG 方法线

对应 skill 组：

- `$visipruner-fx-trace-workflow`: 对同一批目标 event 做 selected-layer FX trace。
- `$visipruner-fx-process-visualization`: 解释和手工可视化 `fx_process_reconstruction.*`，不负责重新采集或导出 ONNX。

用来做什么：

- 在真实 eager `generate()` 中采样同一批 35 个目标 layer 输入。
- 请求结束后离线 replay 采样输入，用 `make_fx(...)` 生成固定输入 `GraphModule`。
- 用 `fx_layer_process_reconstruct.py` 从 `GraphModule.graph.nodes` 重建 readable process。
- 作为“固定输入下低层 ATen DAG 和 node dependency”的证据，和 dispatch 的运行时 op log 互补。

生成 FX trace：

```bash
/workspace/VisiPrune/workload_analysis/env/run_with_analysis_env.sh \
  /workspace/VisiPrune/workload_analysis/fx/fx_dynamic_trace.py \
  --model-layer-trace \
  --trace /workspace/VisiPrune/workload_analysis/algorithmic_trace/traces/fresh_forward_visipruner_full_32tok/algorithmic_trace.json \
  --layers <同一批 35 个 event_id，逗号分隔> \
  --gpu 1 \
  --tag fx_filtered_dispatch_layers_specialized
```

当前产物：

```text
fx/traces/fx_filtered_dispatch_layers_specialized/
```

当前 `run_metadata.json` 记录 `fx_sample_count=35`、`fx_trace_count=35`、
`fx_trace_error_count=0`。

生成 FX process reconstruction：

```bash
/workspace/VisiPrune/workload_analysis/env/run_with_analysis_env.sh \
  /workspace/VisiPrune/workload_analysis/fx/fx_layer_process_reconstruct.py \
  --trace-dir /workspace/VisiPrune/workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized \
  --recursive
```

当前产物：

```text
fx/traces/fx_filtered_dispatch_layers_specialized/<event_id>/fx_process_reconstruction.md
fx/traces/fx_filtered_dispatch_layers_specialized/<event_id>/fx_process_reconstruction.json
fx/traces/fx_filtered_dispatch_layers_specialized/<event_id>/fx_process_nodes.csv
fx/traces/fx_filtered_dispatch_layers_specialized/fx_process_reconstruction_manifest.csv
fx/traces/fx_filtered_dispatch_layers_specialized/fx_process_reconstruction_manifest.json
```

解读或补充人工可视化时使用 `$visipruner-fx-process-visualization`。该 skill 的边界是解释
FX reconstruction 文件，不能把 FX process label 当成真实运行时模块归属。

### Skill 覆盖边界

按已安装 skill 的描述和 `SKILL.md` 正文检查，直接用于生成或解释当前
`/workspace/VisiPrune/workload_analysis` 产物的 skill 已覆盖：

- `$trace-patch-target-discovery`: 前置设计 wrapper/hook/record_function 边界和 join keys。
- `$visipruner-trace-dispatch-profile`: 生成 `algorithmic_trace/traces/*` 和 `dispatch/profiles/*`，并选择 VisiPrune 相关目标 event。
- `$dispatch-layer-reconstruct-onnx`: 从 `dispatch/profiles/*/dispatch_ops.csv` 生成 `dispatch/visualize/*`、`torch_flow/`、`onnx/` 和审计产物。
- `$visipruner-fx-trace-workflow`: 生成 `fx/traces/*` 下的 selected-layer FX trace、`fx_graph_module.pt`、`fx_layer_trace_manifest.csv` 等。
- `$visipruner-fx-process-visualization`: 解释和手工可视化 `fx_process_reconstruction.*`；它不生成 FX trace，也不导出 ONNX。

当前没有单独的已安装 `torch_profile` 专用 skill。`torch_profile/` 是最早的通用
`torch.profiler` 初筛实验；它的 patch/记录边界由 `$trace-patch-target-discovery`
约束，具体脚本和解释记录在 `torch_profile/README.md`。

相关但不属于本目录产物生成链路的 skill 不应混入这里：

- `$visipruner-process-performance-breakdown`: 会消费
  `workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized/*`，但产物是
  `autoresearch/experiments/e2_single_request_latency/SAME_INPUT_*_PROCESS_WISE_PERFORMANCE_REPORT.md`。
  它是 `workload_analysis` 的下游消费者，不是生成当前目录产物的步骤。
- `$visipruner-same-input-workflow`、`$visipruner-sampled-latency-attribution`、
  `$visipruner-same-input-evidence`: 属于 `autoresearch` / E2 SAME_INPUT 性能实验链路，
  处理 Nsight/NVTX/CUPTI layer latency，不生成 `workload_analysis` 下的 trace、dispatch、FX 或 ONNX 产物。
- `project-docker-runner`、`find-skills`、`humanizer-zh`、`nature-*` 等安装 skill
  与当前 `workload_analysis` 产物链路无关。

## 结论

更准确的表述是：

```text
torch.profiler 是性能 profiler：
它记录真实运行中的 op/kernel 时间线，重点是时间、次数、shape、memory 和热点。

workload_analysis 是算法执行与证据重建工具：
torch_profile 提供最早的通用 profiler 初筛；
algorithmic_trace 记录上层动态 schedule；
filtered dispatch profile 记录选中 layer 的运行时 ATen op、shape、tensor ids、
alias 和 inplace 关系；
FX trace 记录同一批目标 layer 的固定输入 GraphModule / ATen DAG；
这些数据用于理解 VisiPrune 实际执行过程，而不是做 wall-clock 性能结论。
```

需要特别注意：

```text
filtered dispatch profile 不是编译时 trace。
它是运行时 eager dispatch trace。

FX process reconstruction 不是运行时模块归属证明。
它是固定输入 FX DAG 上的 readable process grouping。
```
