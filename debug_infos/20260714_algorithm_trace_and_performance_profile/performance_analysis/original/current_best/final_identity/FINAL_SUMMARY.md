# H11.5 + H10.8 R25 final evidence

## Final status

H11.5 wide-causal GQA6 prefill 与 H10.8 gfx936 strided LLMM1 已完成最终
源码/运行时冻结、同一服务上的三次固定 full `run_throughput.sh all`、固定
`run_accuracy.sh all`、SLA、输出重复性、服务健康与正常停服闭环。

最终结果：

- 三轮吞吐公式分（accuracy 前）：
  `88.490349137758 / 88.578483694186 / 88.576533680101`；
- accuracy 系数：`K=1.00`；
- 三轮最终综合分不变，均值：`88.5484555040153`；
- 相对 R24 三轮均值的 20/50/30 加权吞吐提升：
  `+10.2361569769%`；
- 三轮共 `450/450` 请求成功、`failed=0`，SLA 全部通过；
- 三轮的逐请求 `output_lens` 和 `generated_texts` 完全一致；
- 服务最终 health/model 均为 HTTP 200，停止后 health 为 `000`，端口 8001
  无监听且没有 vLLM、accuracy 或 OpenCompass 残留进程。

因此该候选完成了最终测试闭环，但**未达到综合分 90，也未达到相对 R24
+20% 的性能终止条件**。本文件不把这两个条件写成通过；是否按独立的五小时
条件结束目标，应由目标计时记录决定。

此前的 `PROVISIONAL_SUMMARY.md` 只记录 accuracy 运行中的快照；其 pending
状态已由本文件和 `final_accuracy/` 中的完整证据取代，但仍保留在 manifest
中作为可审计历史。

## Candidate and immutable runtime evidence

- Final wheel:
  `vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl`。
- Wheel SHA256:
  `03568ba87ff64fd0a8aade299026d7ee78cbf40d9c1ed5884fb584250b2031f2`。
- Installed `_C.abi3.so` SHA256:
  `209fa0909af1690a8b37976c76f9dd2594da6074f43d9ae31a2c98d1406f52a6`。
- Installed `_rocm_C.abi3.so` SHA256:
  `51e4839b564355279fcca4bc426ccd1da0a5f03d0e39006210960e99fd124ab1`；
  它与最终 wheel 内同名 member 逐字节一致，包含 H10.8
  `LLGemm1_strided_kernel` ABI marker，且不含已拒绝 H10.10 的 K6144
  pair-reduce marker。详见 `runtime/rocm_native_extension_attestation.txt`。
- 所有在 site-packages 中存在的修改/新增 Python 文件均与仓库对应文件逐字节
  一致，见 `runtime/repo_site_hash_pairs.txt`。
- untracked GQA6 文件已单独复制到 `source/untracked/`；tracked diff、git
  status、完整变更文件列表和 repo SHA256 均在 `source/`。
- 构建命令、build/install 日志与状态、wheel 副本和 hash 在 `wheel/`。

固定脚本在服务启动前后 hash 一致：

| Script | SHA256 |
| --- | --- |
| `run_throughput.sh` | `adf0cf91266745b37df916926c7d495ec79f00a11be653c219d1d5df4d93c681` |
| `run_accuracy.sh` | `2e641672a45ac96318c2118df8df4dae2babf87c16afd49cbe4b037ff9beed4e` |
| `start_vllm.sh` | `7c3e8c5ecdf02109e02af8c3b5ba05050b26339c7f50869b5288eea359364fad` |

三个 throughput 与四个 accuracy 数据集的冻结 hash 见
`fixed/fixed_datasets.sha256`。

## Three fixed full runs

三轮均使用未修改的 `run_throughput.sh all`，不传第二参数，默认
`MAX_CONCURRENCY=1`、`REQUEST_RATE=1`、`CUSTOM_OUTPUT_LEN=1024`、
`NUM_WARMUPS=2`，并只清除 localhost proxy 影响。每档各 50 条、benchmark
主体均超过 600 秒。

| Run | Window epoch / wrapper | 4-8K tok/s | 8-16K tok/s | 16-32K tok/s | Final score, K=1 | Relative to R24 | Global TPOT P99 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `1783833368-1783835766` / `2398 s` | `19.589185273966` | `17.025544511643` | `13.003919636950` | `88.490349137758` | `+10.021653961%` | `47.194475 ms` |
| 2 | `1783835836-1783838224` / `2388 s` | `19.587005633034` | `17.127750783436` | `13.003706763767` | `88.578483694186` | `+10.346212210%` | `47.184350 ms` |
| 3 | `1783838265-1783840653` / `2388 s` | `19.584774728543` | `17.126009556001` | `13.004622851957` | `88.576533680101` | `+10.340604760%` | `47.189230 ms` |

三轮均值：

| Band | R24 full mean | Candidate full mean | Relative change |
| --- | ---: | ---: | ---: |
| 4-8K | `18.349460025353` | `19.586988545181` | `+6.744223%` |
| 8-16K | `15.604371326753` | `17.093101617027` | `+9.540469%` |
| 16-32K | `11.434815640267` | `13.004083084225` | `+13.723592%` |

20/50/30 加权后相对 R24 为 `+10.2361569769%`。完整原始 JSON、日志、
status、epoch 和每轮后 health 证据在 `full_runs/run1..run3/`。

## SLA and deterministic output

| Metric | Official 1.5x limit | Candidate maximum | Result |
| --- | ---: | ---: | --- |
| 4-8K TTFT P99 | `7188.718023 ms` | `1956.991400 ms` | pass |
| 8-16K TTFT P99 | `37329.278019 ms` | `6129.860216 ms` | pass |
| 16-32K TTFT P99 | `43111.256069 ms` | `6549.847646 ms` | pass |
| Global TPOT P99 | `107.705132 ms` | `47.194475 ms` | pass |

全局 TPOT P99 按每请求 `sum(itls) / (output_len - 1)` 重建 150 个请求后
计算，不是三档 JSON 的 `p99_tpot_ms` 最大值。三轮每档均
`completed=50, failed=0`，每轮结束 `/v1/models` 均为 HTTP 200。

三轮全部 150 条请求的 output length 和 generated text 均逐请求完全一致；
因此三轮性能可以直接做重复性统计。相对 R24 的历史输出漂移仍由本轮固定
accuracy 的 `K=1` 完成最终计分验收。

## Accuracy and K

固定 `run_accuracy.sh all` 未传第二参数，status `0`，运行窗口
`1783840674-1783841621`，共 `947 s`。测试后 `/v1/models` 仍为 HTTP
200。最终固定脚本表为：

| Dataset | Official baseline | Candidate | Relative decrease Delta | k |
| --- | ---: | ---: | ---: | ---: |
| hotpotqa | `77.959706960` | `77.959706960` (`77.96`) | `0.000000%` | `1.00` |
| gov_report | `32.961006236` | `33.054713499` (`33.05`) | `-0.284297%` | `1.00` |
| retrieval_multi_point | `100.00` | `100.00` | `0.000000%` | `1.00` |
| aggregation_keyword_aggregation | `100.00` | `100.00` | `0.000000%` | `1.00` |

四类均为 `k_i=1.00`，故 `K=1.00`。OpenCompass 原生中间 summary 仍把
aggregation 记为 `0.0`；按固定脚本契约，权威结果是基于预测列表与 gold
多重集合等价重算后的 `100.00`，不能用原生中间值替代。

完整 accuracy run 证据位于 `final_accuracy/run_evidence/`；OpenCompass
配置、四个 prediction JSON、四个 result JSON 和 summary txt/csv/md 位于
`final_accuracy/opencompass_output/`，各子集另有 SHA256 清单。

## Service lifecycle and shutdown

- start epoch: `1783833164`；
- ready epoch: `1783833333`，models/health 均为 HTTP `200`；
- accuracy 后最终 models/health 均为 HTTP `200`；
- stop epoch: `1783841652-1783841659`；
- stop 后 health: `000`；
- `serve_final/residual_check_summary.txt` 记录匹配进程数 `0`、端口 8001
  listener 数 `0`；
- final server log SHA256:
  `0017f838c957e4ee673bd6c35a91b85c91a04463fbb6eb3b039e43c81453f18b`。

服务最终快照和停止证据位于 `serve_final/`。`serve_snapshot/` 是 accuracy
运行中的早期快照，只作为历史证据保留。

## Baseline anchors and manifests

- `anchors/official/` 冻结 official throughput/accuracy baseline；
- `anchors/r24/` 冻结 R24 三次 full、accuracy 和 final build manifest；
- `final_evidence.sha256` 覆盖 final summary、源码/runtime/wheel/fixed、三次
  full、完整 accuracy、最终 serve、official/R24 anchors 以及 provisional
  历史文件；
- `final_manifest_check.txt` 与 `final_manifest_check_status.txt` 记录
  `sha256sum -c`；
- `final_manifest_identity.sha256` 固定主 manifest、校验日志和 status。

最终结论仅为：H11.5 + H10.8 是已完整验证、SLA 与 accuracy 均通过的
R25 候选，最终综合分均值 `88.5484555040153`，相对 R24 加权吞吐提升
`+10.2361569769%`；它没有达到 90 分或相对 R24 +20%。
