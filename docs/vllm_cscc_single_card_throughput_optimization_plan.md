# vllm_cscc 单卡单请求吞吐优化计划与阶段性结论

本文档是优化工作的入口文档，只保留优化路线、当前阶段性结论和后续执行计划。完整实验数据与执行约束拆分到独立文档，通过索引项引用。

配套文档：

- [阶段性实验完整结果](./vllm_cscc_stage_experiment_results.md)
- [优化执行和测试约束](./vllm_cscc_optimization_constraints.md)

## 阶段性结论

### R28 P10a 最终结论：R28-A 性能 NO-GO

P10a 的 R28-A compile/correctness/resource 前门通过：两个目标 shape、四个 config
均使用 MMAC 且 `0 spill`。正式单一随机交错窗口自然完成
`600.042803 s / 8664 sweeps / 103968 samples`。但 best Triton 在 gate/up
相对 auto/H10 慢 `44.5196%/54.0498%`，在 down 慢
`24.2855%/32.3697%`；双 shape 层加权相对 auto 为 `-37.2634%`，所以
**performance NO-GO**。不接 production、不 build/install、不启动服务。

### R28 P10g 最终结论：BQ16/BQ32 group3 GPU 前静态 NO-GO

R28-G 复用冻结 Hg3 QK-hoist 源码，原计划只对
`BQ16/BQ32-BK32-BV64-w2/s2-GROUP_SIZE3` 做无计时 compile/bitwise/resource
前门。源码循环复核在 GPU 初始化前发现 K/H/V 位于 `i_qtile` 内：BQ16 四个
qtiles 的逻辑 major I/O 为 `791.5 MiB`，相对 explicit raw baseline
`385.5 MiB` 增加 `105.3178%`；BQ32 两个 qtiles 也为 `436.5 MiB`
（`+13.2296%`）。两者虽减少 `26.6667%` FLOPs，但从 R28-E 回归恢复到
`+6.01%` 至少需 `30.1443%` latency recovery，BQ32 相对 R28-E 的极乐观
I/O+FLOPs 相加上界仅 `18.7069%`。因此静态 NO-GO；未启动 GPU、未编译/
launch/bitwise/resource、未计时，也不申请正式窗。未执行 probe 草稿已删除。

### R28 P10h 最终结论：C110 S32 correctness NO-GO

此前 P9 只执行了 C000/C101/C100，未执行
`C110=H10 confirmed + S32 on + Hg3 off`，所以 C101 只能证明 Hg3 足以导致
漂移，不能确认或排除 S32。R28-H 使用现有 frozen candidate wheel `f877d08f...`
和 fresh C110 service 补做相同两条 exact-replay。payload SHA 和服务端
`usage.prompt_tokens=10207/22305` 均与合同完全一致；route 也通过 H10 四 key、
S32 hit/no-fallback 和 Hg3 fallback/no-hit。

但两条输出分别变为 `572` tokens / `4655f859...` 与
`289` tokens / `6cb59007...`，均不是 baseline 的 `498/63f45927...` 和
`259/729dd61d...`。`output_canary.json` 顶层 `status=pass` 只表示旧 P9 harness
成功执行请求；该脚本仅对 C000 强制 baseline control，并不表示 C110 exact。
权威字段是两条 `frozen_baseline_exact_match=false`。因此 S32 在 Hg3 关闭时也
足以造成输出漂移，correctness 硬门失败。补充 C010（H10/S32/Hg3=`0/1/0`）
还逐字节复现 C110 的 `572/4655f859...` 与 `289/6cb59007...`，且 H10 disabled、
S32 hit、Hg3 fallback route 全部通过，进一步排除 H10×S32 交互解释并证明 S32
单独充分；不运行任何性能窗或 full/accuracy。

### R28 P10i 最终结论：S32 bitwise-exact salvage 静态 NO-GO

只读审计纠正了旧 post-install validator 的表述：“zero tolerance mismatches”
实际指 `delta > 0.015 + 0.01*|reference|` 的元素数为零，并另设
`max_abs<=4.8828125e-4`；不是零差异。九条 S32 记录全部
`numeric.exact=false`，只有 candidate-to-candidate repeat bitwise exact。

AITER exact decode 使用 `M16/token tile16/segments16`，S32 改为
`M8/tile64/segments32` 和 MI16/kpack2 codegen，QK/P@V、online-softmax 与 reducer
树均不同。历史最接近的 `M8/L16/S16` 仍是 0/10 bitwise exact，虽有
`185 VGPR/8 KiB/0 spill` 和 `+16.2191%` weighted GPU 历史收益，也不能转移到
AITER-equivalent tree。唯一可信 exact 方向是保留 AITER 同一树/代码生成；literal
fallback 收益为 0，specialized clone 也没有可冻结的正收益上界。因此 GPU 前
static NO-GO，不 compile/launch/timing，不改 production。

### R28 P10k 最终结论：矩阵单元覆盖综合完成

当前 baseline 不是“Attn/GDN/Linear 都未使用 Tensor 单元”。Full Attention
Prefill、Attention Decode stage、GDN Prefill KKT/solve/recompute/delta-h/output
和多 token Prefill Linear 都已有动态 `SQ_INSTS_MMOP>0` 与 `v_mmac_*` 静态证据。
例外是 Attention reducer、GDN cumsum/norm/gating/conv、GDN packed Decode，
以及 H10.8 `n=1` GEMV 的 FMA/reduction 路径。

Prefill Linear 的真实调用为 hipBLAS→rocBLAS/Tensile MMAC；automatic index 0
只是 sentinel。1319 个正 solution 已枚举，独立确认保留的候选为三个 key 的
`Gemm_Rocblas_20981` 和 MLP down 的 `20979`，但 portfolio wall reduction
`5.7185% < 5.8136%`，未进 production。R28 A-J 已统一关闭：已有 MMAC 的分支
在性能、资源、bitwise、I/O 或 correctness 门失败；强行把向量算子矩阵化则在
有效 tile 与 Amdahl 上界失败。R28-K 仅综合既有证据，无新 GPU/性能。

并行完成的 H10.20 只读 feasibility audit 判定为 NO-GO。C100 距离 all3 `1%`
门仅 `0.041462 pp`，但 M4096 step-scoped TunableOp 只能删除约 `38.65%` 的
runtime untuned misses；按固定 all3 的 tail/decode 数量折算，需要每个被删除的
lookup 平均至少约 `8.7 us` 才能补齐缺口，当前没有该成本证据。PyTorch
`TuningContext.enable_` 还是非 atomic `bool`，动态全局开关存在通用线程安全
风险。因此不为 H10.20 启动性能窗口，最小 patch 仅作为 future/defer 设计保留。

### R27 已结束目标（8 小时源码优化与矩阵单元探索）

R27 最终以冻结的 H11.5 + H10.8 为唯一直接基线：full x3 综合分均值
`88.5484555040153`、相对 R24 加权吞吐 `+10.2361569769%`、`K=1.0`。
goal start epoch 为 `1783874322`；终止条件是综合分达到 `95`、相对当前最佳
同口径性能再次超过 `20%`，或持续 8 小时至 epoch `1783903122`。除明确的
编译、资源、数值或 profiler 门禁错误外，standalone 和固定 all3 性能窗口
均不得短于 600 秒。

首轮三个结构性方向已完成独立预门禁，尚无候选进入 production：

- H11.10 重新确认当前 H11.5 在 `q=512,seq=12000` 上执行
  `20,686,848` 条 MMOP，同时仍有 `93,189,120` 次 LDS bank conflict。
  gfx936 不支持所试 MMAC `lit/lts` layout；调度提示为二进制 no-op 或编译
  失败；等价 `V^T @ P^T` 虽可降低 bank conflict，但 8-warp 版本 LDS wait
  增至 `2.51x`、MMOP 增至 `1.50x`，其余版本超 VGPR/spill 门，故 no-go。
- H4.6 的 `delta_h + chunk_o` strict fusion 在源码上能消除 `h/v_new`
  共 144 MiB allocation、288 MiB 写读和一个 launch，但保持足够 CTA 的
  BV16 方案会把重复 QK 增至 `4x`、output 总 FLOP 增至 `2.2x`；当前只保留
  独立原型，不进入 production 或性能长测。
- D3 local4 stage 在五个 context 上通过数值/资源门：stage
  `216 VGPR/32768 B LDS/0 spill`，workspace 减少 75%。但 600.029 秒、
  4995 轮随机交错长测表明它加权比冻结 S32 慢 `48.29%`（GPU）、比 AITER
  慢 `17.41%`，且三个长度均相对 S32 回归，故明确 reject。

production repo 仍为 clean 的 `3754870`，未 build/install 新 wheel、未启动
服务。后继 Hg3 grouped-head QK hoist 已完成 600.002 秒长测：相对 production
kernel `+20.634%`、相对同配置 ungrouped 的纯结构收益 `+14.689%`。raw R23
share 的筛选折算约 `+0.487%`，但按正式输入/chunk 重建，20% kernel 降时只
预测 all3/full throughput `+0.150%/+0.105%`；即使 exact kernel 免费也只有
`+0.757%/+0.529%`，故只保留为可组合候选。随后完成了 prefill 大投影
rocBLAS solution-index 枚举，以及 D3 S20 的 compile/resource/numeric 前门。

R27 后续已完成 H10.18/P9 闭环：组合 wheel 的输出漂移由 C101/C100 fresh-state
差分定位到 Hg3；H10-only 保持冻结输出 exact。H10-only 的三个完整 all3 round
自然窗口为 `664.215310295 s`，三档/SLA/输出均通过，但 weighted
`+0.958538041%` 未达到 `+1%` 晋级门，因此不进入 full/accuracy。最终 pinned
H11.5+H10.8 baseline wheel 已回装，source/runtime/8001/GPU 均 clean。

系统与目标双时钟于 epoch `1783903135` 复核：`1783903135 >= 1783903122`，
目标计时为 `28813 >= 28800 s`。最终综合分 `88.5484555040153 < 95`，也没有
相对 current-best 同口径性能严格超过 `20%` 的新合格结果；因此 R27/R28
**仅按 8 小时时间条件结束**。终止证据位于
`/public/home/tangyu408/testdata/goal_runs/20260712_r28_8h_termination_audit`。

### R26 已结束目标（6 小时源码优化续轮）

R26 以已完成 full x3、SLA 和固定 accuracy 闭环的 H11.5 + H10.8 为
唯一直接增量基线；其三轮综合分均值为 `88.5484555040153`、相对 R24
20/50/30 加权吞吐提升为 `+10.2361569769%`、`K=1.0`。本轮不重新解释
R25 的历史终止条件，也不把 standalone/microbenchmark 当作计分结果。

R26 最新 goal start epoch 为 `1783847801`，三个终止条件为：综合分达到 `95`；
相对当前最佳 H11.5 + H10.8 的同口径性能提升超过 `20%`；或持续工作满
6 小时，终止 epoch 为 `1783869401`。用户在续轮开始后将 score 门槛由
`90` 明确提高到 `95`。两个性能条件最终均未满足；只读终止审计于
epoch `1783869431` 确认 6 小时时间条件已满足，R26 据此结束。

按 official 三档 baseline、R25 三轮均值和赛题非线性公式复算，若三档
吞吐近似等比例提高，score 从 `88.54846` 到 `95` 约需再提高
`33.61%`；相对当前最佳提高 `20%` 时 score 约为 `93.02`。因此本轮不能把
触及旧 `90` 当作结束，wave10/GDN 配置包只作为累计栈的前两步；若它们
没有达到新终止条件，继续转向 GDN 中间张量消除/融合等结构性候选。

启动审计确认源码仍是冻结的 H11.5 + H10.8：`skinny_gemms.cu`、
`utils.py`、`rocm_aiter_unified_attn.py` 和 GQA6 新文件 SHA256 分别为
`fb7635f3...6884`、`ddb9ee5d...f661`、`2d038eff...c92`、
`82a52c2b...cfd`；installed `_rocm_C.abi3.so` 仍为
`51e4839b...4ab1`。服务 health 为 `000`，无残留 vLLM/accuracy 进程。

R26 优先进行最终栈上的候选筛选和固定 `all 3` 小样本。每个进入服务测试
的候选必须先完成源码归因、数值验证、重新构建/安装 wheel；除明确报错外，
小样本外层窗口不少于 `600 s`。第一轮并行筛选 H10/GEMV 新 config、
Q/K norm + RoPE 现有 fusion 接入，以及 GDN/FLA 未验证的安全 tiling/fusion；
不得重开已被 R25 证据否决的 H10.9、H10.10 或 H11.6 配置。

第一轮只读审计把 H10.11 `K=5120` 的 640-thread wave10 直接归约列为
decode 首选：删除 H10.8 的 `pair_smem[4][320]` 和一次 barrier，改为
10 个 wave leader 写入约 `160 B` LDS，再顺序完成 cross-wave sum；先在
standalone 中对 r4/r8 做三 shape、多 seed、特殊值、metadata 和不少于
`600 s` 的交错微基准，未通过前不改生产源码。prefill 首选是 gfx936、
BF16、单序列 `T=4096/Hg=16/H=48/K=V=128/BT=64` 的 GDN exact-shape
compiler 配置包；它必须用独立 JIT 副本和严格 gate，避免现 autotune key
未包含 `T` 导致的跨长度错误复用，也不能直接复活 R25 数值失败的
`chunk_delta_h` MFMA 配置。

H10.11 standalone 已完成 `600.113 s / 5721 groups`。数值与 metadata
全部过门，但 r4 serial/q16 相对 H10.8 的真实调用频次加权提升只有
`+0.02845% / +0.00204%`，r8 为 `-6.23387%`；这证明 H10.8 的配对 LDS
和额外 barrier 并非当前瓶颈。H10.11 已 reject，未改 production 源码、
未构建 wheel。执行焦点已转到 GDN T=4096 exact-shape compiler 配置包。

GDN 有效 retry5 已在真实 int32 T=4096 请求与 FP32 recurrent-state 契约下
完成 `600.569 s / 126750 groups`；五 seed 的最终 o/state 严格门禁通过。
四 kernel 降时分别为 `21.20% / 4.33% / 8.38% / 7.65%`，profile-weighted
合计 `14.0646%`、零 spill，但折算全 trace 仅约 `0.48%`。因此它保留为
后续组合补丁，不单独构建/启服；当前先测试能直接影响 TPOT 主体的 H10.12
persistent-row GEMV 和 GQA6 segmented decode attention。

H10.12 已完成 `600.079 s / 5468 groups` 且数值位级通过，但 G2/G4/G8
加权全部回归 `-7.20% / -8.27% / -8.63%`，因此 reject、未改 production。
GQA6 decode 的 9 个资源合格 segmented attention 配置已完成
`600.284 s / 1581 rounds`；另 3 个 logical/padded `32/32` 配置因
stage code object 高达 `251 VGPR` 而在计时前剔除。90/90 数值及
9 stage + 9 reduce 资源门禁通过，最佳 S32 在三个 decode 长度上
都无回归，但 weighted wall/GPU 降时仅 `20.5218% / 20.5722%`，
未达预设的双 `40%` production 门槛。它暂不改 production，保留为
可继续改进/组合的正数候选；当前 GPU 转入 H11.7 prefill logical-64
跨 784-token cache-page 的 standalone 长测。

H11.7 前两次尝试都在计时前按协议停止，不是缩短性能窗口：
首次因 logical tile 改变 online-softmax 分块顺序，相对 H11.5 的过严
`max_abs<=1e-3/mean<=5e-6` 固定门槛失败；retry1 加入冻结 H11.4
同输入参考后，又发现已闭环的 H11.5 本身在随机非连续 cache page
上可达 `0.0078125` 最大差，因此不能用候选无法影响的基线绝对值
否决。retry2 现要求 H11.4/H11.5/H11.7 全 finite、candidate repeat-exact、
资源通过，且 H11.7 相对同输入 H11.5 的新增 max/mean 误差受限。
该协议 75/75 通过，随机交错计时完成
`601.005 s / 19448 groups`。但 logical64 的跨页逐 lane 地址选择成本
压过少算的 dot 列：13 个真实 shape 全部回归，频次加权
`-7.6748%`，最差 `-22.5399%`。H11.7 因此 reject、不改 production；
当前转入不改数学/资源的 H11.8 reverse causal q-block launch-order
候选，同时准备 K=17408 MLP down-projection 新 standalone。
但新 linear probe 必须使用 cold-stream 口径：H10.10 回归审计确认其
K6144 standalone 对同一 60 MiB weight 预热并连续 50 次，而 production
每 decode token 命中 64 份不同权重、仅 K6144 就流过约 3.75 GiB。
该 cache-residency 失配使 standalone 每调用快 `5.93 us`，production 却反转为
慢 `12.55–13.29 us`。已排除 ABI/gate/build/dispatch/graph 问题；因此
H10.10 保持 no-go，H10.13 K17408 在运行前必须加多权重 ring 或
256 MiB L2 flush，production 晋级只看 cold-stream 结果。

H11.8 reverse causal q-block 也已完成 `600.831 s / 605 cycles`；
39/39 同输入结果相对 H11.5 位级一致，code object 与资源完全
相同，但真实频次加权只有 `-0.0110%`，13 shape 全在约
`-0.053%–+0.022%` 的噪声区间。这说明硬件 CTA 调度没有按简化
FIFO 模型产生可计收益；H11.8 reject。当前 GPU 已转入 H10.13
K17408 的 `flush256m/single-call` cold-stream `>=600 s` 测试。

H10.13 实际完成 `600.015 s / 21219 batches`，每方法 `169752`
次单调用，正式区间共 `1527768` 次 256 MiB flush。资源/数值均
过门，但 cold H10.10 reference/rocBLAS 为 `153.20/154.16 us`，最佳
L544 direct 为 `161.04 us`，回归 `-5.12%`；7 个候选全部 reject。
H11.9 logical28/pad32 随后在资源门禁早停：pad32/pad16 的 LDS
仍是 `32768 B`，并未将 CTA/CU 上限从 2 提到 4/8，却必然将
online-softmax 循环增至约 2x/4x，因此无可计时候选。当前转入
GQA6 S32 normalized-BF16 workspace 候选；它的 15 组数值与 4 个
code-object 门禁已通过，`>=600 s` 计时进行中。

BF16 workspace 正式窗口完成 `600.077 s / 5618 rounds`；三个长度都
无回归，相对 FP32 S32 的 weighted wall/GPU 分别为
`+2.0090% / +2.0162%`，但低于预设双 `5%` 门槛，因此不独立
production。由于它在三长度上稳定优于 FP32 S32，当前仅作为
GDN+GQA Python 组合栈的 alternate 复议版准备。H10.14 fused-SiLU
down-projection 则已定量 no-go：激活核免费消除上限只有约
`0.594% TPOT`，而 4-row GEMV 会将 sigmoid/exp 按 1280 row groups 重复，
不进入 HIP 编译或 GPU 长测。

H10.15 K6144 direct/serial cold-stream 也已完成正式性能否决：窗口
`600.001501 s`，每方法 `256944` 次随机 paired single-call，共执行
`2055552` 次 256 MiB changing-value flush。六个候选全部通过资源、反汇编、
多 seed、特殊值和确定性门禁，但 `passing=[]`。更快的冻结基线 H10.10 C2
为 `51.360 us`，rocBLAS 为 `53.760 us`，最佳新候选 768x1 C2 也只有
`51.520 us`，相对 H10.10 为 `-0.311529%`，远低于 `+8%` 晋级线。
因此 H10.15 closed no-go：production 源码前后相同，未生成/应用 production
patch，未 build/install/启服或运行固定 `all 3`。R26 三个终止条件仍未满足，
goal 保持进行中。

R11 `FillFunctor<int>` 的 source attribution 也已闭环：`83740`
次/`18320.211 ms` 中，`83476` 次/`18319.614 ms`（`99.9967%`）
是 Triton autotune 每次计时前执行的 `256 MiB` L2 cache flush，只出现
在服务 ready/初次 warmup 附近，固定正式 benchmark 区间为 `0` 次。正式
区间真实 int metadata fill 只有 `144` 次/`0.30096 ms`；连同已知
BF16 output clear 全部理想消除，公式分上限也仅约 `+0.00423`。因此
H12/Fill 明确 no-go；不禁用 autotune cache flush，也不为仅影响冷启动的
warmup dtype 修正占用 production 测试周期。

### R25 已结束目标（5 小时终止条件）

R24 是 R25 的冻结直接比较基线。R25 的 H11.5 + H10.8 已完成
full x3、SLA 和固定 accuracy 计分闭环；精度系数 `K=1.0`，三轮综合分
均值 `88.5484555040153`，成为当前已闭环的可计分最佳。R25 未达到综合分
`90`，相对 R24 三次 full 均值的 20/50/30 加权吞吐提升
`+10.2361569769%` 也未超过 `20%`。但从 goal start epoch
`1783824849` 到 `1783842849` 已恰满 `18000 s`；终止复验在 epoch
`1783842892` 再次确认累计 `18043 s`。因此第三个 5 小时终止条件已满足，
R25 按时间条件结束；该结论不得误写为达到 90 分或相对 R24 +20%。

当前 R25 最终计分栈已经稳定为 R24 + H11.5 + H10.8。
H10.10 已完成 production 实现和 600 秒小样本，但端到端 TPOT
退化，判定 reject；候选服务在最终 health=200 后停止，源码和运行
wheel 均已回退到 H11.5 + H10.8。该栈的三次固定 full `all`、
TTFT/TPOT SLA 和固定 accuracy 均已完成。最终服务在停止前
models/health 均为 200，已清洁停止且无残留服务；final evidence 与
5 小时终止审计均已完成，冻结结果如下：

- H11.5 对 `cache_block_size=784`、单序列、长 prefill 使用逻辑 56-token
  K/V tile、64-token MFMA padding、`BLOCK_Q=32` 和 causal query 上界；其他
  shape 回退 H11.4。独立 wheel SHA256 为
  `7f26c14ca059826cdeb3bc293411fc662523e88810f23bb834882d87de31e892`。
- H11.5 固定 `all 3` 小样本在 `608 s` 外层窗口内 9/9 成功、最终 health
  通过；相对 R24 小样本三档 TTFT 改善 `15.89% / 20.47% / 24.65%`，TPOT
  变化 `+0.27% / +0.18% / +0.05%`。三档输出 token 数发生漂移，因此 raw
  output throughput 只记录、不用于晋级判断，正确性等待 full 与 accuracy。
- H10.8 保持 H10.7 的三个大投影 exact shape gate，配置改为
  `LLMM1Strided(4,640)` 的 wave-pair reduction。组合 wheel SHA256 为
  `03568ba87ff64fd0a8aade299026d7ee78cbf40d9c1ed5884fb584250b2031f2`；
  三 shape 各 10 seeds 均与 t320 位级一致，正式交错微基准提升
  `8.47% / 8.24% / 7.73%`，无 spill。
- H11.5 + H10.8 固定 `all 3` 小样本在 `602 s` 外层窗口内 9/9 成功、
  最终 health 通过。相对 H11.5，TPOT 下降 `5.23% / 5.03% / 4.91%`，
  TTFT 基本不变，三档 generated text 和 output length 逐项完全一致；因此
  H10.8 没有引入新的输出漂移。当时该结果仅用于晋级 full，不单独计分。
- H11.5 + H10.8 的 full run1 在 epoch `1783833368–1783835766`
  完成，窗口 `2398 s`、status=0、最终 health=200，三档均
  50/50 成功。三档 output throughput 为
  `19.589185 / 17.025545 / 13.003920 tok/s`；假设 `K=1` 的单轮
  pre-accuracy 公式分为 `88.490349`，相对 R24 三轮均值的 20/50/30
  加权相对提升为 `+10.021654%`，均未达 `90` 或 `+20%`。相对
  R24 固定输出，length/text 相同分别仅 `110/150` 和 `97/150`，
  总输出 token `37345 -> 36867`；因此该分数仍是受输出漂移影响的
  provisional raw 结果，不能替代三轮统计和 accuracy。
- full run2 在 epoch `1783835836–1783838224` 完成，窗口
  `2388 s`、status=0、最终 health=200，三档 50/50 成功；吞吐为
  `19.587006 / 17.127751 / 13.003707 tok/s`，pre-K 分
  `88.578484`，相对 R24 加权 `+10.346212%`。run1/run2 三档
  output length 和 generated text 逐请求完全一致；当时的两轮均值为
  pre-K `88.534416`、相对 R24 `+10.183933%`，仍未达 `90/+20%`。
  该跨轮一致性不消除相对 R24 的输出漂移 caveat，也不替代 accuracy。
- full run3 在 epoch `1783838265–1783840653` 完成，窗口
  `2388 s`、status=0、最终 health=200，三档 50/50 成功；吞吐为
  `19.584775 / 17.126010 / 13.004623 tok/s`，pre-K 分
  `88.576534`，相对 R24 加权 `+10.340605%`。三轮平均 pre-K 分
  `88.548456`（95% CI `[88.423426, 88.673485]`），相对 R24 平均
  `+10.236157%`（95% CI `[9.774638%, 10.697676%]`），均未达 `90/+20%`。
  三轮的全局请求 TPOT P99 为
  `47.194475 / 47.184350 / 47.189230 ms`，TTFT/TPOT SLA 全部通过。
  三轮输出签名完全一致，但相对 R24 仍只有 text `97/150`、
  length `110/150` 相同，总 token 少 `478`（`-1.27996%`）。
- 固定 `run_accuracy.sh all` 在 epoch `1783840674–1783841621` 完成，
  窗口 `947 s`、status=0、结束后 models/health=200，output directory 为
  `output/local_accuracy_qwen35/20260712_151818`。固定脚本 Final Results 为
  `77.96 / 33.05 / 100 / 100`，official baseline 为
  `77.96 / 32.96 / 100 / 100`，四项 `k_i=1`，最终 `K=1.0`。
  因此三轮最终综合分与 pre-K 分相同，依次为
  `88.490349137758 / 88.578483694186 / 88.576533680101`，均值
  `88.548455504015`，仍未达 `90` 或相对 R24 `+20%`。
- GDN MFMA 配置探测覆盖 `T=16/32/64/512/4096` 和 `2/4/8` warps。
  在最重要的 `T=4096` 点，缺失数值有效的
  `chunk_delta_h_stateful`；探测 JSON 中 `21.4894%` 时间降幅和
  `1.2737x` 加速是其余四个可用 kernel 重新归一化的上界，不是五
  kernel 的完整加权结果。若保守地把缺失项视为 `1x`，五 kernel
  降幅/加速为 `16.2701% / 1.1943x`，同样低于约 `40%` 的端到端
  `3%` 筛选线；
  `chunk_fwd_o` 单 kernel 虽达到 `36.0648%`，其余有效 kernel 仅
  `5.42%–7.81%`，且 stateful/no-state delta 在该点没有通过数值门槛的
  候选。因此未改生产源码。探测 JSON SHA256 为
  `48b0dd6aca769e10be1aecb8b899205846a102ecd712924f81c439f5a204f7ec`。
- H10.9 曾构建 wheel
  `09501b99beb15c4a3481e398b811b54c25213282293e8fc5240bdae6a64d6ed4`，
  试图强制 PyTorch hipBLAS/rocBLAS；服务 marker 记录调用前 backend 已是
  `_BlasBackend.Cublas`，故该补丁是运行时 no-op。它在进入吞吐测试前即
  停止、回滚并重装 H10.8 wheel，不计作提前结束的有效小样本。
- H11.6 standalone 矩阵在 `q=512, seq=12000` 上确认 H11.5 当前配置为
  `3.9024 ms`、`216 VGPR`、零 spill；所有可执行的 heads2/BQ/warps 变体
  均慢 `1.46x–6.96x`。heads3/6 因 `BLOCK_M=96` 非 2 的幂无法编译，逻辑
  tile 48/64 又不整除 cache block 784。因此没有生产补丁，继续保留 H11.5。
- H10.10 为 gfx936 BF16、`n=1`、无 bias、连续张量、
  `weight[5120,6144]` 增加 `LLMM1Strided(2,768)` exact gate。两轮独立
  `31 groups x 50 calls` 交错微基准均由约 `50.94/50.89 us` 降至
  `45.00/44.97 us`，对应 `+13.1987% / +13.1796%`；K17408 候选只有
  `+2.92% / +2.98%`，未进入生产。H10.10 被拒候选 wheel SHA256 为
  `5c76f909b5fed93ec27fcd2de6555f09d4ba1fdaa5429fd07ce63c73e19272a4`，
  构建、安装和 installed validation 均返回 0，repo/installed `utils.py`
  共同 SHA256 为
  `4329939ee47d417f9af2869ea554993b15973501ec76bd2d1e6b1d100090fcef`。
- H11.5 + H10.8 + H10.10 的固定 `all 3` 小样本在 epoch
  `1783832387–1783832987` 完成，外层窗口恰为 `600 s`，三档
  9/9 成功、status=0、最终 health=200。相对 H10.8，三档 TPOT
  全部回归
  `+1.785% / +1.823% / +1.820%`，20/50/30 weighted raw tok/s 为
  `-0.4393%`。16-32K tok/s 虽为 `+2.744%`，但只来自输出长度
  `[23,259,76] -> [23,284,76]`，不能抵消 paired TPOT 退化；8-16K 也有
  `605 -> 601` 的长度/文本漂移。H10.10 因端到端负结果判定 reject。
  `/public/home/tangyu408/testdata/goal_runs/20260712_restore_h10_8_after_h10_10`
  记录回退 status=0，重装 wheel 为
  `03568ba87ff64fd0a8aade299026d7ee78cbf40d9c1ed5884fb584250b2031f2`，当前 repo 和
  installed 文件均为 H11.5 + H10.8。最终服务目录为
  `/public/home/tangyu408/testdata/goal_runs/20260712_h11_5_h10_8_final_serve`；
  full run1/2/3、SLA 汇总和固定 accuracy 均已完成。最终服务在
  epoch `1783841652–1783841659` 停止；停止前 models/health 均为 200，
  停止后 health=000，无残留服务。

### R24 冻结直接比较基线

R24 作为 R25 的上一可计分最佳和冻结直接对照，其栈为
H6.1c + H4.1/H4.2 + D1 + H11.4 + H10.7。该栈完成三次固定 full all：
三轮 450/450 请求成功，综合分分别为
`85.711178 / 85.704307 / 85.706985`，均值 `85.707490`；相对上一最佳
R23 的 20/50/30 加权吞吐提升三轮均超过 20%，均值 `+20.583336%`。
固定 `run_accuracy.sh all` 得到最终精度系数 `K=1.0`，TTFT 和全局
TPOT SLA 均通过。

| 项目 | official baseline | H6.1c | R23 历史最佳 | R24 上一最佳 |
| --- | ---: | ---: | ---: | ---: |
| 吞吐公式分 | 60.0000 | 69.5665 | 79.0289（三轮均值） | 85.7075（三轮均值） |
| 相对 official 的 20/50/30 加权提升 | 0.0000% | +21.8737% | +50.6725% | +82.2576% |
| 相对 H6.1c 的 20/50/30 加权提升 | — | 0.0000% | +23.7651% | +49.1687% |
| 相对 R23 的 20/50/30 加权提升 | — | — | 0.0000% | +20.5833% |
| 全局 TPOT P99 | 71.8034 ms | 70.3850 ms | 最大 55.6700 ms | 最大 49.5467 ms |
| 精度系数 | 1.0 | 1.0 | 1.0 | 1.0 |
| 综合分 | 60.0000 | 69.5665 | 79.0289（三轮均值） | 85.7075（三轮均值） |

H11.4 通过长 prefill 自适应 launch/MFMA 参数继续降低 attention TTFT；
H10.7 以 gfx936 BF16 两路 FP32 累加的 strided LLMM1 替代三个单 token
大投影，进一步降低 decode TPOT。H10.7 使用严格 shape/device/dtype/
alignment gate，非目标路径保持原 GEMM。三次固定 full 的文本和长度逐条
完全一致；相对 R23 仍有输出漂移及少数重复推理循环样本，但固定 accuracy
四项均为 `k_i=1`。本轮没有另做独立 API token-id/finish-reason 重放，
因此不能宣称改变请求序列后仍逐 token 确定。

R25 进入时的增量工作基线（历史记录）：

- R25 进入时的计分基础为 H6.1c + H4.1/H4.2 + D1 + H11.4 + H10.7；
  H11.5/H10.8 当时只作为 candidate overlay。它们随后已完成 full x3、
  SLA 和 accuracy 闭环并成为 R25 最终最佳；R24 仅继续作为冻结直接对照与
  回退基线。
- 冻结 wheel SHA256：`399c7a847c8607269b41d77f189e96505882094286f6c31a4beedd39194a4fbc`。H10.7 只对 gfx936、BF16、单 token、`k=5120`、无 bias、`m in {96,14336,16384,34816}` 使用 exact shape gate；`m=96` 走原 LLMM1 `rows=4`，三个大投影走 `LLMM1Strided(4,320)`。H11.4 对 `max_seqlen_q >= 128` 使用 `num_warps=2, num_stages=1, waves_per_eu=1, matrix_instr_nonkdim=16, kpack=2`，短 prefill 保留 H11.3 配置。
- 冻结 evidence：R24 build 目录中的 `source_runtime_manifest.sha256` 与 `final_evidence.sha256` 均已完整校验；H11.4 新文件仍为 untracked，迁移时必须显式携带。H10.6 是被 H10.7 取代的中间候选，不属于源码失败；首次 503 仅是代理访问无效测量。
- 可保留的工程基础：`H4.1/H4.2` 的 GDN prefill metadata/no-initial-state 快路径，以及 `D1` 的 GDN decode packed validate 跳过。它们未形成可单独计分的显著收益，但小样本未观察到降速或功能错误，可作为后续源码栈的一部分继续叠加。
- 必须排除：`D2` AITER decode 强制 2D 分支、`H4.3` L2 norm 变体、`H10.5` gfx936 `wvSplitK` 兼容变体。H10.5 首次 BF16 数值对照最大绝对误差 `4.4443`，已回退并重建纯 H11.3 wheel。`H4.4` `wy_fast` tile 变体虽小样本为正，但当前源码已回退；若纳入增量基础，需要单独重放并完成同口径确认。
- 本轮已否决：`H8.1` padded block table fill cache、`H12.1` GDN CUDA graph padding fill cache、`H10.1` 直接启用 ROCm skinny GEMM、`H11.1` AITER descale expand view cache、`H4.5` Qwen3.5 GDN core output `empty` allocation、`H10.2` unquantized GEMM dispatch cache、`H11.2` AITER KV cache view cache。它们均未形成可计入收益，不作为后续基础。

历史通过多请求并发、修改测试脚本、修改数据集、serve 参数扫描、chunk budget 扫描、prefix cache 或未重新编译 wheel 得到的结果，全部不进入结论。

阶段性结论只引用后两个文档中的索引，不在主计划重复展开完整实验日志或约束细节。

| 结论 ID | 结论 | 依据索引 | 后续动作 |
| --- | --- | --- | --- |
| S0 | official baseline 已完成吞吐和精度闭环，可作为固定比较基线。 | [R0](./vllm_cscc_stage_experiment_results.md#r0-official-baseline-full-all-基准), [C0](./vllm_cscc_optimization_constraints.md#c0-赛题口径), [C1](./vllm_cscc_optimization_constraints.md#c1-固定实验契约), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议) | 后续候选统一对齐 R0 的 full `all` 口径。 |
| S1 | H6.1c 是历史可计分基础，也是 R23 的固定增量对照。 | [R1](./vllm_cscc_stage_experiment_results.md#r1-h61c-rocm-aiter-unified-attention), [R2](./vllm_cscc_stage_experiment_results.md#r2-h61c-accuracy-对照), [C4](./vllm_cscc_optimization_constraints.md#c4-正确性门槛) | 作为历史计分基础和增量归因对照保留。 |
| S2 | H5.1 的 upstream ROCm FlashAttention 强制优先改变了输出行为，不能计入收益。 | [R3](./vllm_cscc_stage_experiment_results.md#r3-h51-上游-rocm-flashattention-强制优先), [C4](./vllm_cscc_optimization_constraints.md#c4-正确性门槛) | 只有先修复输出、停止原因和精度后才能重开。 |
| S3 | 早期单条 profile 只能指导定位，不能作为验收结论。 | [R4](./vllm_cscc_stage_experiment_results.md#r4-早期单条-profile), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议), [C5](./vllm_cscc_optimization_constraints.md#c5-profiler-规范) | profiler 结果必须回到固定 full `all` 和 accuracy 闭环验证。 |
| S4 | profiling 后的主优化面是 GEMM/linear、AITER attention2d prefill、Fill/elementwise 和 GDN prefill；decode 影响 TPOT，但不是当前唯一首要热点。 | [R11](./vllm_cscc_stage_experiment_results.md#r11-固定配置-dcuhipprof-profiling), [C2](./vllm_cscc_optimization_constraints.md#c2-允许与禁止), [C5](./vllm_cscc_optimization_constraints.md#c5-profiler-规范), [C6](./vllm_cscc_optimization_constraints.md#c6-源码边界) | 下一轮先做 GEMM/attention2d 的 shape/source 归因，再选一个最小源码候选实施。 |
| S5 | H4.1/H4.2 可作为增量工作基线保留，但不单独计入有效性能突破；H4.3/H4.4 不默认保留。 | [R5](./vllm_cscc_stage_experiment_results.md#r5-h41-gdn-prefill-state-初始化快路径), [R6](./vllm_cscc_stage_experiment_results.md#r6-h42-gdn-prefill-no-initial-state-specialization), [R7](./vllm_cscc_stage_experiment_results.md#r7-h43-fla-l2-norm-kernel-变体), [R8](./vllm_cscc_stage_experiment_results.md#r8-h44-gdn-prefill-wy_fast-tile-变体) | 后续在当前优化栈上叠加；若重开 GDN prefill，必须先有 profiler 归因。 |
| S6 | D1 可作为增量工作基线保留；D2 明确降速并已回滚。 | [R9](./vllm_cscc_stage_experiment_results.md#r9-d1-gdn-decode-packed-validate-跳过), [R10](./vllm_cscc_stage_experiment_results.md#r10-d2-aiter-unified-attention-decode-2d-分支强制) | decode 专项保留，但只有 TPOT 专项 profiling 证明其占主导时才上升为首要候选。 |
| S7 | 后续实验采用“累计优化栈 + 单候选增量归因”的双对照。 | [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议), [C7](./vllm_cscc_optimization_constraints.md#c7-evidence-card-模板) | 每轮同时报告相对 official baseline、H6.1c 和上一轮工作基线的变化。 |
| S8 | 固定配置 profiling 已完成；PCIe 不是瓶颈，显存容量接近打满，主要 kernel 时间集中在 GEMM/linear 与 AITER attention2d。 | [R11](./vllm_cscc_stage_experiment_results.md#r11-固定配置-dcuhipprof-profiling), [C5](./vllm_cscc_optimization_constraints.md#c5-profiler-规范) | 下一轮优先定位 top GEMM 的源 shape/backend 和 attention2d wrapper/kernel。 |
| S9 | H8.1 padded block table fill cache 已实现并验证，但收益为噪声级，已回退。 | [R12](./vllm_cscc_stage_experiment_results.md#r12-h81-padded-block-table-fill-cache), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议), [C4](./vllm_cscc_optimization_constraints.md#c4-正确性门槛) | 不继续做同类 Python cache 补丁；若重开 FillFunctor，先做 stack/source attribution。 |
| S10 | H12.1 GDN CUDA graph padding fill cache 已实现并验证，吞吐 `+0.055%`，属于噪声级。 | [R13](./vllm_cscc_stage_experiment_results.md#r13-h121-gdn-cuda-graph-padding-fill-cache), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议), [C4](./vllm_cscc_optimization_constraints.md#c4-正确性门槛) | 不继续做 metadata padding cache 补丁；下一步转向 GEMM/linear 和 AITER attention2d。 |
| S11 | H10.1 直接启用 ROCm skinny GEMM 不可行，因为当前 wheel 没有注册 `wvSplitK` 扩展。 | [R14](./vllm_cscc_stage_experiment_results.md#r14-h101-rocm-skinny-gemm-可用性探测), [C6](./vllm_cscc_optimization_constraints.md#c6-源码边界) | H10 继续前必须做真实 GEMM shape/source/backend 归因，不能只解除 env gate。 |
| S12 | H11.1 AITER descale expand view cache 已实现并验证，吞吐 `+0.015%`，属于噪声级。 | [R15](./vllm_cscc_stage_experiment_results.md#r15-h111-aiter-descale-expand-view-cache), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议) | AITER attention2d 后续转向 kernel 参数、shape 和 shared-memory bank conflict，不继续 wrapper 小对象缓存。 |
| S13 | H4.5 Qwen3.5 GDN core output 改为 `torch.empty` 会改变输出 token 数，正确性失败。 | [R16](./vllm_cscc_stage_experiment_results.md#r16-h45-qwen35-gdn-core-output-empty-allocation), [C4](./vllm_cscc_optimization_constraints.md#c4-正确性门槛) | 保留 `torch.zeros` 语义；后续不能通过未初始化 output buffer 消除 GDN fill。 |
| S14 | H10.2 unquantized GEMM dispatch cache 已实现并验证，吞吐 `+0.033%`，属于噪声级。 | [R17](./vllm_cscc_stage_experiment_results.md#r17-h102-unquantized-gemm-dispatch-cache), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议) | Python dispatch 不是可见瓶颈；H10 后续必须回到 GEMM shape/source/backend 归因。 |
| S15 | H10.3 已完成 Linear/GEMM source attribution：主目标是 MLP gate/up、MLP down、GDN `qkvz/out` 和 attention `qkv/o`，不是 `in_proj_ba` 小 GEMM；当前 AITER Triton GEMM 与 ROCm skinny GEMM 均不可直接接入。 | [R18](./vllm_cscc_stage_experiment_results.md#r18-h103-lineargemm-source-attribution), [C5](./vllm_cscc_optimization_constraints.md#c5-profiler-规范), [C6](./vllm_cscc_optimization_constraints.md#c6-源码边界) | H10 下一步做这些大投影的真实 backend/kernel 归因，避免继续做 wrapper/dispatch 补丁。 |
| S16 | H11.2 AITER KV cache view cache 已实现并验证，吞吐 `-0.055%`，无收益并已回退。 | [R19](./vllm_cscc_stage_experiment_results.md#r19-h112-aiter-kv-cache-view-cache), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议) | H11 后续不再做 wrapper view/cache 小补丁，转向 attention2d kernel 参数、shape、block table 和 bank conflict。 |
| S17 | H10.4 恢复 `_rocm_C` 并只 gate 实测获胜的 gfx936 LLMM1 shapes；小样本相对 H6.1c 加权 `+13.52%`，三档 TPOT 降至约 `53.47–55.26 ms`。 | [R20](./vllm_cscc_stage_experiment_results.md#r20-h104-gfx936-llmm1-exact-shape-gate), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议), [C4](./vllm_cscc_optimization_constraints.md#c4-正确性门槛) | 作为晋级工作栈的 decode/GEMV 基础保留；其输出变化必须由 accuracy 闭环审计。 |
| S18 | H11.3 消除了 AITER GQA6 prefill attention2d 的重叠行；叠加 H10.4 后，小样本三档相对 H6.1c 均超过 `+20%`，加权 `+22.11%`。 | [R21](./vllm_cscc_stage_experiment_results.md#r21-h113-gqa6-prefill-non-overlap-attention2d), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议) | full、accuracy 与重复性闭环已在 R23 完成。 |
| S19 | H10.5 gfx936 `wvSplitK` 兼容补丁构建成功但 BF16 数值错误，已在进入服务测试前回退。 | [R22](./vllm_cscc_stage_experiment_results.md#r22-h105-gfx936-wvsplitk-correctness-probe), [C4](./vllm_cscc_optimization_constraints.md#c4-正确性门槛) | 不启用 `wvSplitK`；保留 H10.4 的 LLMM1 exact gate。 |
| S20 | H10.4 + H11.3 三次 full 相对 H6.1c 加权提升均值 `+23.7651%`、最差轮 `+23.5860%`；450/450 请求成功，accuracy 系数 `1.0`，公式分均值 `79.0289`。 | [R23](./vllm_cscc_stage_experiment_results.md#r23-h104-和-h113-full-all-accuracy-与重复性闭环), [C0](./vllm_cscc_optimization_constraints.md#c0-赛题口径), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议), [C4](./vllm_cscc_optimization_constraints.md#c4-正确性门槛) | 作为上一可计分最佳与 R24 的直接增量对照保留。 |
| S21 | H11.4 + H10.7 三次 full 共 450/450 成功；综合分均值 `85.7075`，相对 R23 加权提升均值 `+20.5833%`，accuracy `K=1.0`，SLA 全部通过。 | [R24](./vllm_cscc_stage_experiment_results.md#r24-h114-adaptive-gqa6-prefill-和-h107-gfx936-strided-llmm1-闭环), [C0](./vllm_cscc_optimization_constraints.md#c0-赛题口径), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议), [C4](./vllm_cscc_optimization_constraints.md#c4-正确性门槛) | 上一目标闭环，R24 晋升为 R25 的当前可计分基线；R25 终止条件重新按 90 分、相对 R24 +20% 或 5 小时计算。 |
| S22 | R25 已排除 GDN MFMA、H10.9 backend no-op、H11.6 attention 配置矩阵和 H10.10 K6144 exact gate；H10.10 虽 standalone 微基准为正且完成 build/install/validation，但三档端到端 TPOT 均回归。 | [R25](./vllm_cscc_stage_experiment_results.md#r25-h115h108-候选筛选与-h1010-provisional), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议), [C4](./vllm_cscc_optimization_constraints.md#c4-正确性门槛) | H10.10 的 600 秒窗口、final health 和回退已完成；最终栈 H11.5 + H10.8 已完成 full x3/SLA/accuracy，`K=1.0`、三轮均分 `88.548456`，仍未达 90/+20%。5 小时条件已于 epoch `1783842849` 满足；R25 按时间条件结束并冻结全部闭环 evidence。 |
| S23 | R26 以 H11.5 + H10.8 为直接基线重新开启 6 小时源码优化；启动时源码、installed native extension 与冻结 manifest 一致，服务已停止。 | [R26](./vllm_cscc_stage_experiment_results.md#r26-六小时源码优化续轮), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议), [C4](./vllm_cscc_optimization_constraints.md#c4-正确性门槛) | 先并行筛选 GEMV、Q/K norm+RoPE fusion 和 GDN/FLA 候选；晋级候选完成数值/build/install 后优先运行不少于 600 秒的固定 all3 小样本。 |
| S24 | R26 的 GDN + normalized-BF16 GQA6 production 在 603 秒 all3 中仅加权 `+0.229142%` 且 2/9 输出漂移，已 reject；GDN-only 虽以 624 秒 all3 `+0.931708%` 晋级，但 full×3 均值相对 current-best 为 `-1.679841%`。 | [R26](./vllm_cscc_stage_experiment_results.md#r26-六小时源码优化续轮), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议), [C4](./vllm_cscc_optimization_constraints.md#c4-正确性门槛) | GDN-only full 共 450/450、跨轮输出一致、SLA 与 accuracy `K=1.0` 均通过，但最终均分仅 `88.131406`，故 reject。源码/runtime 已回退 H11.5 + H10.8，服务已停止；epoch `1783869431` 的只读审计确认 6 小时时间条件满足，R26 已结束。 |
| S25 | 下一轮按用户指定优先处理三个结构性方向，不再让配置级 GEMV、wrapper cache、Fill 或普通 tile 扫描占用前排周期。 | [R11](./vllm_cscc_stage_experiment_results.md#r11-固定配置-dcuhipprof-profiling), [R26](./vllm_cscc_stage_experiment_results.md#r26-六小时源码优化续轮), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议) | 严格按 H11.10 attention2d LDS bank-conflict layout → H4.6 GDN 跨内核融合 → D3 GQA6 decode stage/reduce/workspace 融合执行；每项先做上限、数值和不少于 600 秒的同输入门禁，再决定是否进入 production。 |
| S26 | R27/R28 已完成 H11.10、H4.6、D3/Hg3、rocBLAS solution、MMAC 覆盖及 R28 A-K 的门禁与综合；所有正式性能窗均不少于 600 秒，未通过候选均未进 production。Attention/GDN Prefill 主链与 Prefill Linear 已由动态 `SQ_INSTS_MMOP` 和 ISA 双重确认使用 MMAC，向量例外不应为覆盖率强行矩阵化。 | [R27](./vllm_cscc_stage_experiment_results.md#r27-八小时源码优化与矩阵单元探索), [R28-K](./vllm_cscc_stage_experiment_results.md#r28-k--p10k-dcu-矩阵单元覆盖最终综合), [C3](./vllm_cscc_optimization_constraints.md#c3-测量协议), [C5](./vllm_cscc_optimization_constraints.md#c5-profiler-规范) | epoch `1783903135` 已满足 8 小时条件；最终保留 H11.5 + H10.8，R27/R28 结束，不再启动候选。 |

## 文档索引

完整实验结果索引：

| 索引 | 引用 | 用途 |
| --- | --- | --- |
| R0 | [Official baseline full all 基准](./vllm_cscc_stage_experiment_results.md#r0-official-baseline-full-all-基准) | baseline 吞吐、accuracy 与综合基准 |
| R1 | [H6.1c ROCm AITER Unified Attention](./vllm_cscc_stage_experiment_results.md#r1-h61c-rocm-aiter-unified-attention) | 历史可计分基础的构建、吞吐与 SLA |
| R2 | [H6.1c accuracy 对照](./vllm_cscc_stage_experiment_results.md#r2-h61c-accuracy-对照) | 精度系数和正确性阶段结论 |
| R3 | [H5.1 上游 ROCm FlashAttention 强制优先](./vllm_cscc_stage_experiment_results.md#r3-h51-上游-rocm-flashattention-强制优先) | 已排除的无效候选 |
| R4 | [早期单条 profile](./vllm_cscc_stage_experiment_results.md#r4-早期单条-profile) | 仅作定位起点的历史 profile |
| R5 | [H4.1 GDN prefill state 初始化快路径](./vllm_cscc_stage_experiment_results.md#r5-h41-gdn-prefill-state-初始化快路径) | GDN prefill 小样本未达标候选 |
| R6 | [H4.2 GDN prefill no-initial-state specialization](./vllm_cscc_stage_experiment_results.md#r6-h42-gdn-prefill-no-initial-state-specialization) | GDN prefill 小样本未达标候选 |
| R7 | [H4.3 FLA L2 norm kernel 变体](./vllm_cscc_stage_experiment_results.md#r7-h43-fla-l2-norm-kernel-变体) | 输出变化且吞吐下降的回滚候选 |
| R8 | [H4.4 GDN prefill `wy_fast` tile 变体](./vllm_cscc_stage_experiment_results.md#r8-h44-gdn-prefill-wy_fast-tile-变体) | GDN prefill 小样本微弱收益、未达标候选 |
| R9 | [D1 GDN decode packed validate 跳过](./vllm_cscc_stage_experiment_results.md#r9-d1-gdn-decode-packed-validate-跳过) | Decode wrapper 检查开销不是主瓶颈 |
| R10 | [D2 AITER unified attention decode 2D 分支强制](./vllm_cscc_stage_experiment_results.md#r10-d2-aiter-unified-attention-decode-2d-分支强制) | AITER single-query 3D 分支优于强制 2D，候选回滚 |
| R11 | [固定配置 DCU/hipprof profiling](./vllm_cscc_stage_experiment_results.md#r11-固定配置-dcuhipprof-profiling) | 当前瓶颈排序和 PMC 证据 |
| R12 | [H8.1 padded block table fill cache](./vllm_cscc_stage_experiment_results.md#r12-h81-padded-block-table-fill-cache) | 已实现但无收益的 Fill/metadata 候选 |
| R13 | [H12.1 GDN CUDA graph padding fill cache](./vllm_cscc_stage_experiment_results.md#r13-h121-gdn-cuda-graph-padding-fill-cache) | 已实现但无收益的 GDN metadata 候选 |
| R14 | [H10.1 ROCm skinny GEMM 可用性探测](./vllm_cscc_stage_experiment_results.md#r14-h101-rocm-skinny-gemm-可用性探测) | 已排除的 GEMM env gate 候选 |
| R15 | [H11.1 AITER descale expand view cache](./vllm_cscc_stage_experiment_results.md#r15-h111-aiter-descale-expand-view-cache) | 已实现但无收益的 AITER wrapper 候选 |
| R16 | [H4.5 Qwen3.5 GDN core output `empty` allocation](./vllm_cscc_stage_experiment_results.md#r16-h45-qwen35-gdn-core-output-empty-allocation) | 输出 token 数变化的正确性失败候选 |
| R17 | [H10.2 unquantized GEMM dispatch cache](./vllm_cscc_stage_experiment_results.md#r17-h102-unquantized-gemm-dispatch-cache) | 已实现但无收益的 linear dispatch 候选 |
| R18 | [H10.3 Linear/GEMM source attribution](./vllm_cscc_stage_experiment_results.md#r18-h103-lineargemm-source-attribution) | GEMM 热点绑定到 Linear 层 shape/source，排除小 GEMM 优先级 |
| R19 | [H11.2 AITER KV cache view cache](./vllm_cscc_stage_experiment_results.md#r19-h112-aiter-kv-cache-view-cache) | 已实现但无收益的 AITER wrapper view cache 候选 |
| R20 | [H10.4 gfx936 LLMM1 exact-shape gate](./vllm_cscc_stage_experiment_results.md#r20-h104-gfx936-llmm1-exact-shape-gate) | 已显著降低 TPOT、进入晋级工作栈的 decode Linear 候选 |
| R21 | [H11.3 GQA6 prefill non-overlap attention2d](./vllm_cscc_stage_experiment_results.md#r21-h113-gqa6-prefill-non-overlap-attention2d) | 小样本相对 H6.1c 加权超过 20% 的晋级候选 |
| R22 | [H10.5 gfx936 wvSplitK correctness probe](./vllm_cscc_stage_experiment_results.md#r22-h105-gfx936-wvsplitk-correctness-probe) | 构建成功但数值失败并回退的 kernel 候选 |
| R23 | [H10.4 和 H11.3 full all accuracy 与重复性闭环](./vllm_cscc_stage_experiment_results.md#r23-h104-和-h113-full-all-accuracy-与重复性闭环) | 上一可计分最佳和 R24 的直接增量基线 |
| R24 | [H11.4 adaptive GQA6 prefill 和 H10.7 gfx936 strided LLMM1 闭环](./vllm_cscc_stage_experiment_results.md#r24-h114-adaptive-gqa6-prefill-和-h107-gfx936-strided-llmm1-闭环) | R25 的上一最佳/冻结直接基线：源码、10min 筛选、三次 full、SLA、accuracy、输出审计和最终计分 |
| R25 | [H11.5/H10.8 候选筛选与 H10.10 provisional](./vllm_cscc_stage_experiment_results.md#r25-h115h108-候选筛选与-h1010-provisional) | H11.5/H10.8 实现与小样本、GDN/H10.9/H11.6/H10.10 reject、H10.10 回退、full x3/SLA/accuracy/输出审计、服务清理、final evidence 与 5 小时终止闭环 |
| R26 | [六小时源码优化续轮](./vllm_cscc_stage_experiment_results.md#r26-六小时源码优化续轮) | H11.5/H10.8 直接基线、候选筛选、GDN+BF16-GQA reject、GDN-only full×3/accuracy reject、回退、final manifest 与 6 小时时间终止闭环 |
| R27 | [八小时源码优化与矩阵单元探索](./vllm_cscc_stage_experiment_results.md#r27-八小时源码优化与矩阵单元探索) | H11.10/H4.6/D3/Hg3/rocBLAS 与 R28 A-K 闭环、Attention/GDN/Linear MMAC 覆盖综合、baseline 恢复及 epoch `1783903135` 的 8 小时时间终止审计 |

执行和测试约束索引：

| 索引 | 引用 | 用途 |
| --- | --- | --- |
| C0 | [赛题口径](./vllm_cscc_optimization_constraints.md#c0-赛题口径) | 指标、SLA、评分公式、精度约束 |
| C1 | [固定实验契约](./vllm_cscc_optimization_constraints.md#c1-固定实验契约) | 固定脚本、固定参数、no-proxy 和 accuracy 命令 |
| C2 | [允许与禁止](./vllm_cscc_optimization_constraints.md#c2-允许与禁止) | 可改源码范围与禁止策略 |
| C3 | [测量协议](./vllm_cscc_optimization_constraints.md#c3-测量协议) | wheel 构建、服务启动、吞吐和精度闭环 |
| C4 | [正确性门槛](./vllm_cscc_optimization_constraints.md#c4-正确性门槛) | 输出、finish reason、hash 与 OpenCompass 门槛 |
| C5 | [Profiler 规范](./vllm_cscc_optimization_constraints.md#c5-profiler-规范) | HIP timeline、counter 与瓶颈分类要求 |
| C6 | [源码边界](./vllm_cscc_optimization_constraints.md#c6-源码边界) | 只读锁定文件与可优化源码地图 |
| C7 | [Evidence Card 模板](./vllm_cscc_optimization_constraints.md#c7-evidence-card-模板) | 候选进入结论前必须补齐的证据 |

## 优化计划

### 目标和边界

目标是在单卡、单请求并发、固定测试脚本的条件下提升 Qwen3.5-27B vLLM CSCC/DCU 在线服务的 `output_throughput`。

不可改变的边界：

- 不修改 `run_throughput.sh`、`run_accuracy.sh`、模型权重、tokenizer、chat template 或评测解析口径。
- 不通过多请求并发、serve 参数调优、修改 batch scheduler、prefix cache、speculative decoding、持久化量化或模型结构变化获得收益。
- 允许的主路径是修改 `remote-home/vllm_cscc` 源码，重新编译 wheel，使用固定启动和固定测试脚本验证。

所有候选必须按 C3 完成源码变更、wheel 构建、服务启动、吞吐 `all`、accuracy `all` 和证据记录闭环。

### Workload 与硬件判断

单请求长上下文包含 prefill 与 decode 两段。

Prefill 阶段处理 prompt/context 并生成 KV cache，是 8K-32K 档 TTFT 的主要候选来源。vLLM 的 `chunked prefill` 是同一请求内部的 context 切分，不改变请求数、不改变上下文语义、不拆数据集。chunked prefill 可以作为路径分析对象，但不能扫描或调整 `MAX_NUM_BATCHED_TOKENS`，也不能修改 batch scheduler 代码来改变 chunk 调度策略。

Decode 阶段每次生成一个或少量 token，是 TPOT P99 的核心来源。decode 优化应优先围绕 Attention、Linear、GDN state update、KV 读取、kernel launch 和 HBM 字节流展开。

模型与硬件先验：

- Qwen3.5-27B 使用 `bfloat16`。
- full-attention `head_dim=256`。
- `layer_types` 共 64 层，其中每 4 层 1 个 `full_attention`，约 16 个 full-attention 层和 48 个 `linear_attention`/GDN 层。
- DCU 目标卡按讲义归纳为海光 DCU `gfx936`，`wavefront=64`。
- 微基准给出的可持续量级：HBM 峰值约 `1206 GB/s`，bf16 算力峰值约 `395 TFLOPS`。
- Roofline 拐点约 `327 FLOP/byte`。算术强度低于该拐点时优先按 memory-bound 分析，高于该拐点时优先按 compute-bound 分析。

这些先验只用于决定先测哪里；进入结论必须以 C5 的 profiler 证据和固定脚本结果为准。

### Backend 证据矩阵

| 路径 | 预期源码 | 必需证据 | 边界 | 决策 |
| --- | --- | --- | --- | --- |
| full-attention prefill | `selector.py`、`flash_attn.py`、`rocm_aiter_fa.py`、`triton_prefill_attention.py` | selector 日志、kernel timeline、chunk `q_len` | 不改 batch scheduler 或 chunk 参数 | 证实热点后优化 wrapper/Triton/HIP kernel |
| full-attention decode | `paged_attn.py`、`triton_decode_attention.py`、`csrc/rocm/attention.cu` | paged decode kernel 名称与耗时 | gfx936 custom paged attention 只支持 head 64/128 | 未命中时降低 `attention.cu` 优先级 |
| KV cache allocation/block manager | `kv_cache_manager.py`、`single_type_kv_cache_manager.py`、`block_pool.py`、`kv_cache_utils.py` | block 分配/释放次数、碎片率、HBM 占用、cache miss | 不改锁定参数或 scheduler 策略 | 若开销高，优化内部实现 |
| KV cache write/update | `triton_reshape_and_cache_flash.py`、`cache_kernels.cu`、`cache_kernels_fused.cu` | reshape/cache kernel 次数、字节流、slot mapping shape | 不改变 KV 语义或输出口径 | 若 memory-bound，减少 HBM 往返 |
| Q/K norm + RoPE | `qk_norm_rope_fusion.py`、`fused_qknorm_rope_kernel.cu`、`torch_bindings.cpp` | custom op 命中、kernel 数下降、数值对比 | 不生成可复用模型图或改变模型结构 | 只在 full-attention 路径成立时优化 |
| GDN/linear attention prefill | `gdn_attn.py`、`qwen3_next.py`、`layers/fla/ops/` | GDN kernel timeline、48 层占比、state/update shape | `--gdn-prefill-backend` 只是已有开关，不能单独计收益 | 若占比高，减少中间张量和 launch |
| GDN/linear attention decode | `layers/fla/ops/`、`gdn_attn.py` | per-token state update kernel 与 HBM 读写 | 不改变 recurrent state 语义 | 若 TPOT 热，优化 state update |
| Decode 算子融合 | `gdn_attn.py`、`layers/fla/ops/`、`qwen3_next.py`、`linear.py`、`gpu_model_runner.py` | decode token 级 kernel 序列、launch gap、相邻 elementwise/norm/reshape/copy 证据 | 不改采样、stop、输出 token、batch scheduler 或请求间状态复用 | 只融合数学等价子路径 |
| Linear/GEMM/GEMV | `linear.py`、`kernels/linear/`、`_aiter_ops.py`、`platforms/rocm.py` | rocBLAS/hipBLASLt/AITER/Triton 命中证据 | 不做持久化权重量化或权重重排压缩 | 只修正真实 fallback |
| 非持久化运行时量化 | `quantization/kv_cache.py`、`torch_utils.py`、attention/linear kernels | KV/activation 动态 scale、临时 dtype、精度审计 | 禁止生成量化权重文件、压缩缓存或持久化转换 | 只有精度系数通过时保留 |
| 非 batch-scheduler runtime overhead | `gpu_model_runner.py`、`gpu_input_batch.py`、`workspace.py` | Python timeline、launch gap、metadata 构造次数 | 禁止修改 batch scheduler 相关代码 | 仅缓存/消除执行路径重复 work |

### 假设 Backlog

优先级由当前增量工作基线上的 profiler 结果决定。每个候选一次只验证一个假设，避免把 backend 切换、fusion、调度变更和环境开关混成不可归因数字。新增候选不从 official baseline 重新开始，而是在已有可行优化栈上叠加；归因时同时记录相对上一轮工作基线和相对 H6.1c 的差异。

| ID | 触发条件 | 源码路径 | 预期瓶颈 | 最小改动 | 验证信号 | 失败判据 |
| --- | --- | --- | --- | --- | --- | --- |
| H0 路径表征 | 尚未证明固定脚本实际命中哪些 backend | `selector.py`、`registry.py`、`model_runner.py`、相关 backend | 结论风险来自路径假设错误 | 增加一次性日志或轻量插桩 | backend 矩阵填完整 | 日志影响吞吐或无法关联到 EngineCore |
| H1 Decode 专项：Attention/GDN/算子融合 | H6.1c 后 TPOT 只小幅改善 | `paged_attn.py`、`triton_decode_attention.py`、`gdn_attn.py`、`layers/fla/ops/`、`linear.py`、`qwen3_next.py` | HBM 带宽、GEMV、GDN state update、launch-bound | 数学等价融合、减少读写字节、修正 backend fallback | TPOT P99/Mean TPOT 下降，output throughput 提升 | 输出 token、finish reason、OpenCompass 精度或 SLA 失败 |
| H1.1 full-attention decode | attention decode 占 TPOT 主体 | `paged_attn.py`、`triton_decode_attention.py`、`rocm_aiter_unified_attn.py`、`attention.cu` | KV 读取、block table address、softmax/reduction、非合并访存 | shape/capability gating、减少 wrapper/copy、优化 block table 访问 | attention decode kernel 时间下降 | head_dim=256 路径不命中目标 kernel，或输出异常 |
| H1.2 GDN/linear attention decode | 48 个 GDN/linear 层 state update 累计占比高 | `gdn_attn.py`、`layers/fla/ops/`、`qwen3_next.py` | recurrent state 读写、gate/update 中间张量、reshape/copy、launch-bound | 融合 gate/state/update/norm 等价子路径 | GDN decode kernel 数、HBM bytes、TPOT 下降 | state 语义改变或精度下降 |
| H1.3 decode 算子融合 | 单 token decode 小 kernel 多且 gap 明显 | `qwen3_next.py`、`linear.py`、`gpu_model_runner.py`、`workspace.py`、fusion/custom op | launch-bound、metadata 重建、中间 tensor HBM 往返 | 融合相邻无副作用算子，缓存固定 shape 元数据 | kernel 数和 EngineCore gap 下降 | 触碰 scheduler、改变采样/stop 或引入持久化缓存 |
| D3 GQA6 segmented decode stage/reduce/workspace 融合 | R26 的 S32 segmented decode kernel 加权快约 `20.52%`，但两阶段 workspace/reduce 使端到端理想上限仅约 `+1.46%` | `rocm_aiter_unified_attn.py`、GQA6 decode 专用 op、workspace 管理 | segment workspace HBM 往返、独立 reducer launch、FP32 vector workspace | 严格 shape gate 下合并 stage/reduce 或减少 segment/workspace，vector 可用 BF16、max/expsum 保持 FP32 | decode wall/GPU 同时下降、workspace bytes 和 launch 数下降 | 资源超限、上下文档位回归、输出漂移或理论端到端上限仍不足以晋级 |
| H2 KV cache allocation/block manager | 显存碎片、block 管理或 cache 分配开销可观 | `kv_cache_manager.py`、`single_type_kv_cache_manager.py`、`block_pool.py`、`kv_cache_utils.py` | 块管理开销、碎片率、无效 block 访问 | 优化内部块管理和 layout | HBM 占用/碎片下降，TTFT/TPOT 不退 | 改变上下文容量、scheduler 或请求语义 |
| H3 KV cache write/update | cache reshape/write 在 prefill 或 decode 中高占比 | `triton_reshape_and_cache_flash.py`、`cache_kernels.cu`、`cache_kernels_fused.cu` | 非合并访存、重复地址计算、额外 copy | 合并 slot mapping/address 计算，向量化 K/V store | cache kernel 时间和 HBM bytes 下降 | 破坏 KV 正确性或输出哈希 |
| H4 GDN/linear attention prefill | 48 个 GDN/linear 层合计占 TTFT 主体 | `gdn_attn.py`、`qwen3_next.py`、`layers/fla/ops/` | 中间张量 materialization、HBM 往返、小 kernel launch | 融合安全的 gate/state/norm 子路径 | GDN kernel 数和字节下降，TTFT P99 不退 | GDN 占比低，或精度下降 |
| H4.6 GDN 跨内核中间张量/state 融合 | R26 的 T=4096 compiler config 虽使四 kernel 加权降时 `14.06%`，折算全 trace 仅约 `0.48%` | `layers/fla/ops/chunk.py`、`chunk_scaled_dot_kkt.py`、`wy_fast.py`、`chunk_delta_h.py`、`chunk_o.py`、`gdn_attn.py` | `A/w/u/h/v_new` materialization、重复 HBM 读写和相邻 launch | 每次只融合一条相邻边界，优先消除可证明不再复用的中间量，保持 FP32 recurrent final state | GDN 链 kernel 数、HBM bytes、链路 GPU 时间和 TTFT 同时下降 | 只得到 config 级收益、理论全链路上限低于 `2%`、state/output 数值失败或资源溢出 |
| H5 full-attention prefill backend | full-attention chunk kernel 是 TTFT 热点 | `flash_attn.py`、`rocm_aiter_fa.py`、`triton_prefill_attention.py`、`prefix_prefill.py` | IO-bound attention、fallback、wrapper/copy | 对实际命中 path 做 shape 特化或减少 wrapper/copy | TTFT P99 下降，output throughput 提升 | head_dim=256 路径未命中目标 kernel |
| H6 GEMM/GEMV backend gating | profiler 显示 Triton/eager fallback 或 AITER 未命中 | `_aiter_ops.py`、`platforms/rocm.py`、`linear.py`、`kernels/linear/` | backend 选择错误、API 能力检测不足、权重带宽受限 | 基于实际 API 和 dtype/shape 做 capability gating | backend 稳定命中，TPOT 或 TTFT 改善 | 伪装设备能力、fallback 抖动或权重持久化变更 |
| H7 非持久化运行时量化 | KV/activation 带宽是主瓶颈且精度有余量 | `quantization/kv_cache.py`、`torch_utils.py`、attention/linear kernels | KV 或 activation 字节流过大 | 动态 scale、临时低精度、kernel 内部转换 | output throughput 提升，OpenCompass 通过 | 生成持久化量化产物或精度下降不可接受 |
| H8 非 batch-scheduler runtime overhead | HIP timeline 存在 Python gap、launch gap 或 metadata 重建 | `gpu_model_runner.py`、`gpu_input_batch.py`、`workspace.py` | Python overhead、metadata 重建、workspace resize | 缓存固定 shape 元数据，减少重复对象构造 | EngineCore gap 下降，TTFT/E2E 改善 | 修改 scheduler 代码或锁定参数 |
| H9 Q/K norm + RoPE fusion | full-attention 层中 norm/rope 小 kernel 多 | `qk_norm_rope_fusion.py`、`fused_qknorm_rope_kernel.cu`、`torch_bindings.cpp` | launch-bound 与中间张量写回 | 源码级 custom op/fusion | kernel 数下降，数值/精度通过 | 被判定为模型图重构或数值不稳定 |
| H10 GEMM/linear source attribution | R11 显示 `linear_gemm` 占 `53.221%`，top GEMM PMC 低 L2 hit/低有效 GFLOPS | `linear.py`、`kernels/linear/`、`_aiter_ops.py`、`qwen3_next.py`、ROCm backend gating | 小/瘦 GEMM、GEMV、权重流式访存、backend 选择或融合不足 | 给 top rocBLAS kernel 加 shape/source 归因，再做单点 gating/fusion 候选 | top GEMM 时间下降，TTFT/TPOT 至少一项改善 | 无法绑定到源码调用点，或候选改变权重/模型结构 |
| H11 AITER attention2d prefill | R11 显示 `kernel_unified_attention_2d` 占 `26.707%`，PMC 共享内存 bank conflict `28.134%` | `rocm_aiter_unified_attn.py`、AITER wrapper、attention metadata | attention2d tiling/shared-memory、wrapper tensor 构造、descale/copy | 保持 H6.1c backend，减少 wrapper/copy 或做 shape-safe gating | attention2d kernel 或 wrapper 总时间下降，TTFT 改善 | 重复 D2 式强制 2D/3D 分支或改变输出行为 |
| H11.10 H11.5 attention2d LDS bank-conflict-aware layout | H11.5 已是最佳 prefill 路径，但 PMC 仍显示 `28.134%` shared-memory bank conflict；H11.7/8/9 已排除改 tile/page/launch-order | `vllm/v1/attention/ops/rocm_aiter_unified_attention_gqa6.py`、`rocm_aiter_unified_attn.py` | K/V LDS bank conflict、共享内存寻址和 load/store layout | 保持 BQ32/BM64、logical56、causal/page mapping，只做 LDS padding/skew/swizzle 和 bank-aware lane mapping | 同输入 profile-weighted kernel wall/GPU 至少下降 `8%`，无关键 shape 回归 | spill、LDS/VGPR 导致 occupancy 下降、跨页地址成本增加、数值预算失败或小样本无收益 |
| H12 Fill/metadata source attribution | R11 显示 `FillFunctor<int>` 占 `8.386%`，H8.1/H12.1 两类 padding cache 无收益 | `gpu_model_runner.py`、attention metadata、block table 构造、临时 tensor 初始化点 | GPU tensor fill、metadata 初始化、图捕获 padding | 先用 stack/source attribution 锁定具体 fill，再决定是否合并或消除 | FillFunctor 时间下降且输出/finish 不变 | 未归因前继续做缓存补丁，或触碰 batch scheduler |

R25 最终状态：H10.8 t640 wave-pair reduction 与 H11.5 wide-causal tile
已完成各自和组合小样本；GDN MFMA、H10.9 backend selector、H11.6 attention
配置矩阵和 H10.10 K6144 exact gate 均已否决。H10.10 虽完成两轮微基准、
production build/install/validation，但固定 all3 的三档 TPOT 全部回归；其
600 秒外层窗口、最终 health、服务停止和 H10.8 wheel/source 回退均已完成。
最终计分栈为 H11.5 + H10.8。三次固定 full `all`、SLA 和 accuracy
均已完成；`K=1.0`，最终平均分 `88.548456`、相对 R24 平均加权
`+10.236157%`，未达两个性能终止条件。SLA 全部通过，三轮输出
一致，但仍有相对 R24 的漂移；固定 accuracy 未观察到计分精度下降。
服务已清洁停止。R24 wheel/evidence 继续冻结作为直接比较与回退基线；
5 小时终止条件已满足，R25 按时间条件结束，final evidence 已冻结并校验。

### 终止后冻结与交接

1. 保留 R24 wheel、源码快照和 evidence 为 R25 的冻结直接对照/回退基线；
   冻结 R25 full x3、SLA、accuracy、输出审计和服务停止 evidence。
2. H11.5、H10.8 的独立/组合构建、数值和至少 600 秒小样本已完成；继续保留其 evidence，不重复把已完成步骤写成待办。
3. H10.10 的 600 秒窗口、最终 health 和回退证据已完成；保留被拒
   wheel/源码/小样本 evidence，不得用 K6144 standalone 微基准覆盖端到端
   负结果。运行时已重装 H10.8 wheel，repo 与 installed hashes 已对齐。
4. accuracy 已完成且 `K=1.0`；保留固定脚本 Final Results、OpenCompass
   output directory、原生 aggregation `0.00` 与固定 Counter 复算 `100.00`
   的口径说明。5 小时终止 epoch `1783842849` 已到达，R25 按时间条件结束；
   固定 full x3/SLA/输出审计、服务清理、accuracy 与 final evidence manifests
   均已冻结，R25 下不再启动新的性能实验。
5. 提交或迁移时必须同时携带 R23/R24 累计栈 evidence、R25 tracked diff、完整 untracked GQA6 文件、source/runtime manifest 和 final evidence manifest；不能只携带 tracked diff。
6. 保留 R24 相对 R23 的输出漂移记录。任何 R25 候选必须继续做 paired comparison、停止原因/长度审计和固定 OpenCompass accuracy。
7. H10.8 只替换 H10.7 三个大投影的等价 config，保持 gfx936、BF16、`n=1`、`k=5120`、无 bias 和精确 m gate；`m=96` 保留原 LLMM1，不重新启用失败的 wvSplitK。
8. H11.5 只对 gfx936、BF16、head256、GQA6、单序列长 prefill 使用 wide-causal path；短 query、多序列、decode 和非目标 shape 回退 H11.4/AITER 原路径。
9. D2、H4.3、H4.5、H10.5 以及 H8.1/H11.1/H11.2/H12.1 等无收益或错误候选继续排除，不作为 R25 基础。
10. 固定 `run_throughput.sh`、`run_accuracy.sh`、`start_vllm.sh`、模型权重、tokenizer、chat template、serve 参数和 scheduler 边界继续保持只读；每轮小样本除明确报错外外层窗口至少 600 秒。

### R26 已完成执行计划与终局

1. H11.5 + H10.8 继续作为直接比较、运行回退和最终计分基线，不修改其
   已冻结 evidence。
2. 先做无需服务的源码/shape/ABI 筛选；候选只有在独立数值与交错微基准
   显示足以影响端到端时才进入 production patch。
3. 每个 production 候选必须重新 `build_py --force`、构建 wheel、
   `pip install --force-reinstall --no-deps`，并冻结 repo/runtime hashes。
4. 固定 `run_throughput.sh all 3` 不传第二参数；除脚本报错外，外层窗口
   必须不少于 `600 s`，并记录每档 text/length、TTFT、TPOT 和 health。
5. 小样本晋级后才运行 full x3、SLA 和固定 accuracy；未晋级候选立即回退
   H11.5 + H10.8 wheel/source，不能用 standalone 数字覆盖端到端负结果。
6. 本轮只按 score `95`、相对 H11.5 + H10.8 `+20%` 或 epoch
   `1783869401` 三个条件结束。

当前终局状态：

1. GDN T=4096 + normalized-BF16 GQA6 decode 已完成 production
   validation、build/install 和固定 all3。外层窗口 `603 s/status=0`；加权
   throughput 仅 `+0.2291416%`、TPOT 回归 `+0.0305292%`，并有 2/9
   输出漂移，故组合栈已 reject，BF16 decode 已移除。
2. GDN-only transition 已把 attention backend 恢复为冻结 hash
   `2d038eff…c92`，删除 decode 文件并保留六个 GDN 文件；production-wrapper
   数值门通过，构建 wheel SHA256 为 `469f83fc…5466`。
3. GDN-only 固定 all3 完成 `624 s/status=0`，三档 throughput 为
   `12.954590848 / 16.058930524 / 9.893121789 tok/s`，相对 R25 加权
   `+0.9317080%`，加权 TPOT 回归仅 `+0.0004890%`。但 2/9 输出漂移，且
   主要增量来自 8-16K 第 3 条 output length `498→537`，所以该结果仅支持
   晋级，不是最终收益结论。
4. fixed full×3 已完成，窗口为 `2347 / 2324 / 2324 s`，三轮均
   `status=0`，合计 450/450 成功。三档 throughput 分别为：run1
   `19.574725732 / 16.556590106 / 12.548636950`，run2
   `19.562743160 / 17.049593698 / 12.547661543`，run3
   `19.563605453 / 17.048709058 / 12.549713716 tok/s`；三轮逐条输出一致，
   TTFT/TPOT SLA 全部通过。
5. fixed accuracy 完成 `953 s/status=0`；hotpotqa/gov_report/retrieval/
   aggregation 为 `77.959706960 / 32.875225385 / 100 / 100`，最终 `K=1.0`。
   三轮综合分为 `87.836326413 / 88.278637194 / 88.279255341`，均值
   `88.131406316`；相对 current-best 加权吞吐 `-1.679840604%`。因此
   GDN-only reject，既未达到 score `95`，也未达到 current-best `+20%`。
6. GDN overlay 已撤销，tracked source diff 与 pre-overlay/R25 final diff
   按字节一致；runtime 已恢复 wheel `03568ba8…031f2`，服务停止并记录
   `health=000, alive=0`。H11.5 + H10.8 继续作为冻结最终提交栈。
7. 只读终止审计已于 epoch `1783869431` 完成，满足
   `1783869431 >= 1783869401`，R26 按 6 小时时间条件结束，不再启动新实验。
   最终 evidence manifest 共 `153` 项并已 `153/153` 校验通过，manifest
   SHA256 为
   `58f721374ef9cb189eebee0a7c493ebf2334dc8149d4271ac754a3a428fb6723`。

### R27/R28 源码优化执行状态

用户指定的前三项及其硬件计数后继候选均已完成；下表按历史执行顺序冻结最终
状态，R27/R28 已按 8 小时时间条件结束：

| 优先级 | 候选 ID | 当前状态 | 已得到的结论 | 后续动作/门槛 |
| ---: | --- | --- | --- | --- |
| P1 | H11.10 | **no-go** | 当前 H11.5 确有 MMOP 和 LDS conflict，但可用 layout/schedule 开关均在编译、资源或 PMC 门失败 | 冻结证据，不 build/install；除非得到 gfx936 可执行的新 LDS lane mapping，不重扫同类开关 |
| P2 | H4.6 strict `delta_h+o` | **defer/no production** | 可删除 288 MiB handoff 流量，但 BV16 重复 QK 4x、output FLOP 2.2x；尚无可信正上限 | 不运行低信息量长测；改做 `chunk_fwd_o` Hg3 grouped-head QK hoist，先编译/数值/资源门 |
| P3 | D3 local4 | **600 s reject** | workspace -75%，但加权 GPU 相对 S32 `-48.29%`、相对 AITER `-17.41%` | 冻结反例；不做继续减少 producer CTA 的 workspace-only 变体 |
| P3b | D3 S20 | **resource reject / model withdrawn** | 15 个随机数值 case 均通过，但 stage 217 VGPR 超预注册 `<=216` 门；后续证明旧的一 CTA/CU critical-depth 模型错误，S20 调度优势无可靠上限 | 冻结本版、不计时、不启动 S20r1；217/216 实际 residency 相同也不能事后改写候选门 |
| P4 | H10.16 | **600 s confirmation positive / defer production** | discovery `7.395%/7.332%` 经新 seed 确认缩水为 GPU/wall `5.786%/5.718%`；shared out 未过 3% GPU 门而回退 auto，最终只有 4 explicit keys | 低于 all3 前门 `5.8136%` 约 0.095 pp，不单独写 CSV/build/all3；只在可消融组合精确上限 `>2%` 时复议 |
| P5 | H4.7 | **600 s kernel pass / defer production** | Hg3 相对 production `+20.634%`、纯结构 `+14.689%`；156 VGPR/8 KiB/0 spill，数值/repeat 通过；正式 exact-T chunk 重建的免费 all3 上限也仅 `+0.757%` | 冻结为可组合源码候选；不单独 patch/build/all3，只有与其它独立收益组合预测超过门槛才复议 |
| P6 | H4.8 | **static no-go** | KKT/recompute/delta 的 Hg3 grouping 只能分别省 K read，不能在严格语义下消除 MMAC；delta 还会降至 64 CTA 且 live-set 下界 60 KiB | 不建 JIT；KKT+solve 即使整段免费 endpoint 也仅约 `1.597%`，继续聚焦 rocBLAS solution |
| P7 | H10.17 tail-M rocBLAS | **static no-go** | 全部 tail 投影免费的 full 上限为 `+2.1248%`，但达 full `+1.5%` 需尾块整体降时 `71.0618%`；首轮 6 exact keys 全免费也仅 `+0.0682%` | 不启动低上限的 600 s solution 枚举；只有能一次覆盖大量 M 且不增 lookup 风险的结构证据才重开 |
| P8 | H10.18 composable bundle | **661.94 s all3 reject / rollback complete** | 三轮三档吞吐均正、weighted `+1.59295%`、SLA/failed 门 PASS，但 8–16K idx2 `498→497` 和 16–32K idx1 `259→265` 的文本/长度漂移在三轮稳定复现 | `advance_to_full=false`，不运行 full/accuracy；source clean HEAD、pinned baseline wheel `03568ba8…`、四方运行文件与 runtime/8001 clean 均已恢复 |
| P9 | H10.19 drift attribution/subset salvage | **Hg3 correctness reject；C100 664.22 s screen reject；baseline restored** | C000 精确复现 baseline；C101(H10+Hg3) 两条均漂移；C100(H10-only) 两条恢复 baseline exact，归因指向 Hg3。C100 三轮输出/SLA/三档均通过，但 weighted `+0.958538% < 1%` | 不进 full/accuracy；回装 pinned baseline wheel `03568ba8…`，source/runtime/8001/GPU clean。H10-only 保留为正但未达门候选，Hg3 production 路径冻结 reject |
| P9b | H10.20 M4096-scoped TunableOp | **read-only NO-GO / defer patch** | 只可移除约 `38.65%` runtime untuned misses；补齐 `0.041462 pp` 需约 `8.7 us/removed lookup`，无实测成本支持；PyTorch 全局 enable flag 为非 atomic `bool` | 不 build/install、不启动性能；仅保留“unset enable env + M4096 model-forward/dummy-run scope + owner-thread fail-closed”的未来最小 patch |
| P10a | R28-A | **600.043 s performance NO-GO** | 两个 shape、四个 config 均通过前门且使用 MMAC/0 spill，但 best Triton 相对 auto 加权 `-37.2634%` | 冻结 reject；不做 production integration/build/install，不重扫相同 blocked GEMM config |
| P10b | R28-B/B2 | **resource NO-GO before PMC** | 首轮 P/V/P+V 为 `222/232/217 VGPR`；B2 的 both-operand `vec4/m8`、`vec2/m16`、`vec8/m8` 仍为 `225/233/219 VGPR`。全部保持 128 MMAC/0 spill/32 KiB，但均 `>216` | 两轮均 `survivors=[]`，无 PMC/性能；不放宽门，不做 production patch/build/install |
| P10c | R28-C exact BK32 Hg3 group2+1 | **bitwise/resource Pareto empty；NO-GO before timing** | `w2/s2` 在 5 seeds×2 modes 为 10/10 bitwise exact，但 pair `227 VGPR > 216`；`w4/s1`、`w8/s1` 分别 `130/76 VGPR`、8 KiB、0 spill，却均 0/10 bitwise exact，最大差 `6.1035e-05`。pair/singleton 均保持 MMAC | 无 passing config，禁止短测/长测；不改 repo/安装。H10-only 尚缺 `0.041462 pp`，未来 exact 候选至少需约 6% `chunk_fwd_o` 降时，但本路径已在前门冻结 |
| P10d | R28-D both-shared perPhase | **final LDS resource NO-GO** | `vec4/pp2/mp8`、`vec4/pp2/mp16`、`vec4/pp4/mp8` 均保持 128 MMAC/0 spill/32 KiB，但为 `223/233/229 VGPR > 216` | `survivors=[]`；不做 bitwise/PMC/性能，结束该 LDS encoding 分支 |
| P10e | R28-E BQ32/BK32 group2+1 | **front PASS；600.001 s performance NO-GO** | BQ32 `w2/s2` 为 10/10 bitwise exact，pair `201 VGPR/14 KiB/0 spill`；唯一正式窗 188864 sweeps、755456 samples。pair+singleton median 相对 explicit BK32 baseline 在 no-state/stateful 分别 `-34.5489%/-34.5981%`，p90 也回归 | correctness 保持但性能远低于所需 `+6.01%`；冻结关闭，不改 production、不 build/install/启服 |
| P10f | R28-F non-MMOP boundary audit | **offline static NO-GO** | F1 将 GDN BT64×H48 scan 改为 FP32 triangular MMAC，dense MAC/add 比约 65x，整 kernel 免费上界仅 `+0.030009%`；F2 将 W4 depthwise causal-conv 改为 block-diagonal MMAC，仅 6.25% 乘积有效、至少 16x 计算。Attention reducer/L2 norm 免费上界也仅 `+0.078061%/+0.084071%` | 全程 no GPU/no compile/no PMC/no timing；不改 production。两项均不值得唯一 `>=600 s` 窗，关闭“为覆盖率强行 MMAC”分支 |
| P10g | R28-G BQ16/BQ32 Hg3 group3 | **GPU 前 static NO-GO** | 源码级 qtile 重载计数：BQ16 `791.5 MiB`、比 baseline `+105.3178%`；BQ32 `436.5 MiB`、`+13.2296%`。两者 FLOPs `-26.6667%`，但 BQ32 相对 R28-E 的极乐观 I/O+FLOPs 恢复仅 `18.7069% < 30.1443%` 所需 | 删除未执行 harness；无 GPU init/compile/launch/bitwise/resource/timing，不申请正式窗，不改 production |
| P10h | R28-H C110/C010 exact replay | **S32 correctness NO-GO before performance** | C110 request payload/usage 与 route 全部通过但两条输出非 baseline exact；C010 关闭 H10 后逐字节复现同一 `572/4655f859...`、`289/6cb59007...`，S32 hit、Hg3 fallback | 两态均 `frozen_baseline_exact_match=false/false`，证明 S32 单独充分；禁止 `>=600 s all 3` 和 full/accuracy。顶层 canary `status=pass` 仅表示请求执行，不是 correctness pass |
| P10i | R28-I S32 bitwise-exact salvage audit | **GPU 前 static NO-GO** | 旧 validator 的 9 条 S32 record 均 `allclose=true/mismatch_count=0/repeat_exact=true`，但 9/9 `numeric.exact=false`；AITER 为 `M16/L16/S16`，S32 为 `M8/L64/S32`，stage/reducer 树不同；历史 `M8/L16/S16` 也 0/10 bitwise exact | 只有 AITER-equivalent clone/fallback 有可信 exact 前景；literal fallback 0% gain，clone 无保留收益证据。无 GPU/compile/timing/production 改动；未来必须 `atol=rtol=0` 且 service exact 后才可测性能 |
| P10j | R28-J raw-A two-kernel | **GPU 前 bandwidth static NO-GO** | producer 按 `(chunk,Hg)` 一次 QK，写 16 MiB FP32 `A_raw`；consumer 的 3 heads×2 BV 共读 96 MiB。FLOPs `16.1061→10.7374 GF`（`-33.3333%`），但 major I/O `385.5→433.5 MiB`（`+12.4514%`），CTA `6144→7168`、launch `1→2` | 不建 kernel、不用 GPU/compile/bitwise/resource/timing；带宽门失败且 R28-E 经验上界不足，不申请 `>=600 s` 窗、不改 production |
| P10k | R28-K matrix-unit coverage synthesis | **read-only closure** | Attention Prefill/Decode stage、GDN Prefill 主链、Prefill Linear 已用 MMAC；reducer/scan/norm/gate/conv/GDN Decode 与 H10.8 n1 GEMV 是向量例外。Linear confirmed IDs 为 `20981/20979`，但未过 production 门 | 冻结 A-J 综合、源码路径和动态/静态证据；无 GPU/compile/timing。后续只优化有效工作、bytes/launch/归约正确性，不追求 MMOP 覆盖率本身 |

### DCU 矩阵单元覆盖与新增优化原则

gfx936 上本文所称 “Tensor 单元” 是 MMAC/MMOP matrix datapath，不是 NVIDIA
Tensor Core。是否命中由动态 `SQ_INSTS_MMOP` 与 HSACO 中
`v_mmac_*` 双重确认；不能仅凭源码出现 `matmul`、`tl.dot` 或库名判断。

| 路径 | 当前是否使用 MMAC/MMOP | 证据与优化含义 |
| --- | --- | --- |
| Full Attention prefill | 是 | H11.5 `kernel_unified_attention_2d_gqa6` 在真实 shape 有 `20,686,848` MMOP；QK/PV 均为 BF16 MMAC。问题是 LDS/PV layout 与有效工作，不是“未用 Tensor”。 |
| Full Attention decode | stage 是，reducer 否 | `kernel_unified_attention_3d` 有 `384,000` MMOP，`reduce_segments` 为 0；reducer 是 max/sum/vector merge，不应强行矩阵化。 |
| GDN prefill 主链 | 是 | KKT、solve、recompute、delta-h、output 的 production code objects 均含 MMAC；应减少重复 QK、materialization 和低利用 tile。 |
| GDN prefill 辅助链 | 否 | cumsum、L2 norm、gating、causal-conv 为 elementwise/scan/stencil，算术强度和形状不适合 MMAC；重点是 fusion/bytes/launch。 |
| GDN decode core | 否 | packed recurrent kernel 动态 `MMOP=0`、`VALU=258,048`；状态流式更新算术强度约 `0.875 flop/B`，且历史 full trace 占比约 `0.472%`，暂不为“增加 MMAC”重写。 |
| Attention/GDN 外围 Prefill Linear | 是 | `F.linear` 实际经 hipBLAS→rocBLAS/Tensile；单 dispatch 有 `22,282,240` MMOP。automatic 等价正 ID 为 `20844/20845/20846/20838`，确认候选为 `20981/20979`，但未过 production 门。 |
| H10.8 目标 Decode Linear | 否（GEMV 例外） | `n=1,K=5120` exact shapes 走 `LLMM1/LLMM1Strided`，源码为 packed/vector FMA、FP32 `fmaf` 和 wave/shared reduction；不要与多 token GEMM 混写。 |

Decode 的实际运行源码由 backend 在构造期导入
`/usr/local/lib/python3.10/dist-packages/aiter/ops/triton/unified_attention.py`
（SHA256 `004d569a...`），不是仓库中的 vLLM 同名参考实现。该 AITER 文件固定
`BLOCK_M=16/NUM_SEGMENTS=16`，并由 cache-block/head-size 推出 token block 16；
R28-I 已用此实际路径复核 S32 的 M8/L64/S32 归约树差异。

GDN decode 的静态 MMAC rewrite 上限已冻结在
`/public/home/tangyu408/testdata/goal_runs/20260712_r27_gdn_decode_mmac_feasibility`：
双 matvec 填入 N16 MMAC 只有 12.5% 有效列且使核心 FLOP 变为 5x；rank-1
outer 也只有 12.5% 有效 K，全部矩阵化为 7x FLOP。即便免费删除该 kernel，
按 0.472% trace share 的端到端上限也只有 `+0.47424%`，因此在编译前否决。

新增矩阵单元探索遵循四条规则：

1. 对已经命中 MMAC 的 kernel，优先提高有效矩阵工作占比、减少重复 dot、
   LDS/HBM 等待和 padding；MMOP 数增加本身不是收益。
2. 对 rocBLAS/hipBLAS GEMM，记录真实 kernel/solution、shape、转置、workspace、
   MMOP/VALU 与冷流口径；不能把 PyTorch 的 backend preference 当作已选算法。
3. 只有能把 scan/reduction 重写为尺寸充足、数据复用充分且数值等价的块矩阵
   时才考虑 MMAC；普通 norm/gate/reducer 不为追求覆盖率而矩阵化。
4. 所有新 kernel 仍服从独立数值、零 spill/资源、>=600 秒性能和固定 all3
   门槛；“使用 Tensor 单元”不能替代端到端收益与正确性。

共同基线与实验协议：

1. 唯一直接基线为冻结 H11.5 + H10.8；开始任何新候选前先复核源码、
   installed wheel/native hashes、服务停止状态，并在正式 benchmark 区间重做
   一次无扰动 kernel timeline/PMC，更新三个目标路径的真实占比。
2. 每个候选只改变一个可归因结构；先保存 source diff、shape gate、理论字节/
   launch 上限和失败判据，再做数值、code-object 和同输入性能测试。
3. standalone 或 production 小样本除明确报错外，外层窗口都必须不少于
   `600 s`。production 筛选固定使用 `./run_throughput.sh all 3`；只有晋级
   候选才使用不带第二参数的 `./run_throughput.sh all` 完成 full×3。
4. production 候选必须重新执行 `build_py --force`、构建 wheel、强制安装，
   并从非源码目录验证 repo/build/wheel/site-packages marker 和 hashes 一致。
5. 小样本晋级统一要求：三档 9/9 成功、20/50/30 加权 output throughput
   至少 `+1.0%`、任一档不为负、TTFT/TPOT 不触发 SLA、文本/长度/finish
   reason 不漂移。未满足即回退，不用额外 output token 制造的吞吐正数晋级。
6. full×3 后必须完成 450/450、paired output comparison、全局 TPOT P99、
   分档 TTFT P99 和固定 `run_accuracy.sh all`；只有 `K=1.0` 且重复性通过的
   结果才可替换直接基线。

#### P1 / H11.10：attention2d LDS bank-conflict-aware layout（本轮 no-go）

下列协议已执行到编译、资源和 PMC 门。所有可用变体均在长测前达到预注册的
失败条件，结果见 R27；本轮不再按同一开关矩阵重复执行。

1. 只修改
   `vllm/v1/attention/ops/rocm_aiter_unified_attention_gqa6.py` 及必要的
   `rocm_aiter_unified_attn.py` shape gate；保持 H11.5 的 BQ32/BM64、
   logical56、两 query heads/CTA、causal 顺序和跨 784-token cache-page 映射。
2. 候选仅覆盖 K/V LDS padding、skew/swizzle、bank-aware lane mapping 和
   等价 vector load/store 布局；不得重新尝试 H11.7 logical64、H11.8 launch
   order 或 H11.9 half-width tile。
3. 编译门要求零 spill/private、LDS 不超过 H11.5 的 `32768 B`，VGPR 不高于
   H11.5 的 `216`；若资源使 CTA/CU ceiling 下降，计时前直接否决。
4. 数值门覆盖 13 个真实 prefill shapes、多 seed、连续和随机非连续 block
   table、跨页位置及 repeat-exact；沿用 H11.4/H11.5 同输入动态误差预算，
   不再用候选无法影响的固定绝对阈值误杀。
5. 完成不少于 600 秒的随机交错同输入计时。只有 profile-weighted wall/GPU
   同时至少下降 `8%`、任一高频 shape 回归不超过 `2%`，且按最新 profiler
   折算端到端至少 `+1.5%` 时，才 build/install 并进入固定 all3。

#### P2 / H4.6：GDN 跨内核中间张量与 state 融合（strict 方案暂缓）

`delta_h+chunk_o` strict fusion 已完成 source/byte/FLOP/grid 上限审计；
BV16 的重复 QK 和 2.2x output FLOP 使其没有可信性能上限，故未进入 JIT 或
长测。下面保留原门禁作为未来有新并行结构时的协议记录。

1. 先从 `layers/fla/ops/chunk.py` 记录 `chunk_scaled_dot_kkt`、
   `recompute_w_u`、`chunk_delta_h`、`chunk_o` 之间每个中间量的 shape、dtype、
   生命周期、读写字节和 launch 时间；只按正式区间计数，不含 Triton autotune
   flush/warmup。
2. 每次只融合一条相邻边界，优先评估
   `chunk_scaled_dot_kkt.py + wy_fast.py` 和
   `chunk_delta_h.py + chunk_o.py`；必须证明被消除的中间量没有其它消费者。
   recurrent final state 保持 FP32，公开 output/state 契约不变。
3. 若理论上把目标中间量和 launch 全部免费消除仍不足当前端到端 `+2%`，
   该边界只记录 no-go，不进入 HIP/Triton production 实现。不得把 R26 已验证
   仅约 `0.48%` 全 trace 的 T=4096 compiler config 再包装成融合收益。
4. 数值门覆盖 T=16/32/64/4096、真实 int32 metadata、no-state/stateful、
   多 seed、low-amplitude 和 NaN/Inf 分类；同时比较公开 output、final state
   及融合边界内部参考，中间误差不得超过已建立的 BF16/FP32 预算。
5. 融合链同输入长测必须达到 GPU 时间下降至少 `10%`、零 spill且实际减少 HBM bytes
   与 kernel 数；通过后才进入固定 all3，晋级门沿用共同协议。

#### P3 / D3：GQA6 decode stage/reduce/workspace 融合（local4 已 reject）

local4 已通过资源/数值门并完成 600.029 秒长测，但加权比 S32 慢 48.29%；
下面的协议记录已经闭环。任何新 D3 设计必须保持至少 S32 的 producer
并行度或提供真正并行的 CTA 内 reduction，不能只继续压缩 workspace。

后继 S20 的 source/static/CPU gate 已完成，证据在
`/public/home/tangyu408/testdata/goal_runs/20260712_r27_d3_decode_fusion_probe/s20_standalone`。
它保持 logical/padded64、MMAC tile 和页映射，只把 producer segments 改为20，
在 4 KV heads 上形成 80 CTA；五个 context 均被 64-token tile exact-cover，
compact stride20/padded32 reducer identity 已验证。当前尚未 import Triton、
compile 或 launch；GPU 前门被显式环境锁保护，等待 P5 长测释放设备。

该 GPU 前门随后已执行：五 context×三 seeds 和特殊值数值门全部通过，
workspace 仍为 495360 B；reducer `38 VGPR/4096 B/0 spill`。stage 为
`217 VGPR/32768 B/0 spill`，比事前冻结的 `<=216` 门高 1，因此本版
`status=reject` 且没有性能 runner/计时。不能用 general `<=224` 上限事后
覆盖更严格的候选自设门。后续 HSACO/sysfs 审计还撤回了旧的一 CTA/CU
critical-depth 优势模型，因此不再立 S20r1。

1. 以 installed AITER decode 和 R26 的 S32 segmented probe 为双参考；严格
   gate gfx936、BF16、head256、GQA6、`max_query_len == 1` 和已验证 context，
   prefill、非目标 shape 与多序列路径全部回退原实现。
2. 优先减少 segment 数、把局部 reduction 前移到 stage、复用一次性 workspace
   或消除独立 reducer launch；若 vector workspace 使用 BF16，max/expsum 和
   最终归一化继续保持 FP32。禁止跨请求缓存和非确定性 atomic 累加。
3. 资源门保持 stage/reducer 零 spill/private、单 code object 不超过
   `224 VGPR`；记录 workspace 实际字节数和每 token launch 数，不能只报告
   kernel 内部百分比。
4. 数值门覆盖至少 5 个 context 长度、3 seeds、连续/随机 cache page、特殊值
   和 repeat，对照 AITER 与冻结 FP32 S32；所有目标长度必须 finite 且无新增
   超预算误差。
5. 不少于 600 秒的随机交错计时必须使 weighted wall/GPU 同时下降 `40%`，
   或由最新 production profiler 证明端到端上限达到 `+2%`。否则即使继续快于
   FP32 S32，也只冻结为组合候选，不单独 build/install/启服。

#### P4 / H10.16：prefill rocBLAS solution 与 MMAC 利用率审计

1. 从固定 phase trace 绑定 MLP gate/up、MLP down、GDN qkvz/out 和 full
   attention qkv/o 的真实 `(M,N,K,dtype,transA,transB)`，同时记录调用次数。
2. 用 profiler kernel name/marker 和 ISA/`SQ_INSTS_MMOP` 确认实际 solution；
   PyTorch `preferred_blas_library` 只作 preference，不能当作 solution 证据。
3. 只比较库已暴露且数值等价的 solution/workspace/layout；测量必须同时覆盖
   warm 和 production-like 多权重 cold stream，避免重现 H10.10 cache 假正数。
4. 只有 exact-shape 候选 standalone 足以预测至少 `+1.5%` 端到端，且完成
   >=600 秒随机交错门，才允许生产 gate/build/install/all3。
5. 若 shape-specific solution 经独立确认，优先使用 PyTorch ROCm TunableOp
   结果 CSV 的 `Gemm_Rocblas_<id>` 精确映射，并关闭在线 tuning；必须先审计
   未映射 shape 的 Default 回退、lookup 开销、validator 和 graph capture，
   不直接全局切 backend，也不生成权重重排产物。

纯静态 production 上限与 TunableOp 审计已在 discovery 结束前完成：
六个逻辑投影中 `full_attention_out` 与 `gdn_out_proj` 共用同一
`tn_5120_4096_6144_ld_6144_6144_5120` 签名，因而 CSV 只能部署
五个 exact key，不能按 Python 来源为这两类调用分配不同 ID。按固定
all3/full 输入重建，这六类 M4096 投影的加权时间需分别下降
`5.813563% / 12.361010%`，才能预测 all3 `+1%` / full throughput
`+1.5%`；严格 mean-E2E `-1.5%` 门为 `12.570410%`。即使六类在
所有 M4096 full chunk 上全免费，乐观上限也只有 full 加权吞吐
`+13.9474%`、score `91.410759`，所以本组 solution 无法单独达到 95。
该审计是否进入 production 的硬门，不用 discovery 进行中的部分排名。

正式 discovery 已自然完成 `1727.0307 s / 39600 samples`，7920 组合
每个恰好 5 次，终止记录为 `complete=true`。排名使用实际降时
`1-candidate/auto`，而非会放大百分比的 speedup 口径。先按逻辑 shape
选择会产生不可部署的 out-projection 组合；改为五个 TunableOp key
后，shared out key 用 `48+16` 层联合决策 ID 20981，可部署 portfolio
的 GPU/wall 降时为 `7.394655% / 7.331939%`。该结果只超过
all3 `5.813563%` 前门，低于 full `12.361010%` 端点门，因此不能
直接写 production CSV。

入围并集 `20981/20980/20979/20846` 的新 seed 确认随后自然完成
`600.000075 s / 286646 samples`，30 组合每个 `9554--9555`
样本。gate/up=20981、down=20979、GDN qkvz=20981 和 attention
qkv=20981 通过 GPU>=3%、wall>=2% 且上下半窗口同向门；shared out
ID 20981 的联合 GPU 降时只有 `2.843%`，因此保留 auto。最终
四个 explicit keys 的 GPU/wall portfolio 降时为
`5.786412% / 5.718477%`，比 all3 前门低约 `0.0951 pp`。H10.16
不单独 patch/build/all3，只作为已独立确认的可消融项进入 P8 精确组合审计。

#### P5 / H4.7：GDN `chunk_fwd_o` Hg3 grouped-head QK hoist

该方向的独立门已完成。exact BV64 候选把 grid 从 `(2,64,48)` 改为
`(2,64,16)`，QK FLOP 降 66.67%、总 output FLOP 降 26.67%、q+k read
从 192 MiB 降到 64 MiB。600.002 秒三方法随机窗口证明纯结构收益 14.689%，
且两种 state mode 的 median/p90 均通过。R23 raw share 的启发式折算约
0.487% kernel-trace 时间，但正式 benchmark chunk 重建更严格：20% 降时只
预测 all3/full throughput `+0.150%/+0.105%`，免费 exact kernel 上限也低于
all3 1% 门，故以下条目作为已执行协议保留。

1. exact gate 固定为 gfx936/BF16、`B1,T4096,Hg16,H48,K=V=128,BT64`；
   grid 从 value-head 维改为 grouped-head 维，同一 CTA 内三个 value heads 共享
   原始 FP32 QK，公开 output/state 和 reduction tree 保持等价。
2. 独立 JIT 必须先过 parse、zero-spill、LDS/VGPR 和多 seed/repeat 数值门；
   若跨三个 heads 保持 QK live 导致 spill 或 occupancy ceiling 下降，计时前否决。
3. 理论记录 QK MMAC、q/k read 和 CTA 变化；目标是减少冗余 MMAC，不能通过
   增加 padding/无效 dot 伪造更高 MMOP 使用率。
4. 前门通过后运行 >=600 秒随机交错同输入长测；只有 kernel 降时与最新
   profile 预测达到 production 门槛，才进入固定 all3。

#### P6 / H4.8：其余 GDN prefill grouped-head MMAC 去重（static no-go）

只读 exact-T4096 审计没有找到第二个可实现的 MMAC 去重候选：KKT 把
`beta_h` 移出 head dot 会改变 BF16 舍入；recompute/delta grouping 只能省
32/128 MiB K read，不能减少各 value head 独立的 MMAC；delta 还需同时保留
三份 recurrent state，grid 仅 64 CTA，而当前 kernel 已 194 VGPR/36 KiB LDS。
solve 没有 Hg/K/Q 共享。证据冻结在
`/public/home/tangyu408/testdata/goal_runs/20260712_r27_h4_8_remaining_hg3_static_audit`，
不 import/JIT/计时，也不改 production。

#### P7 / H10.17：non-4096 tail-M rocBLAS（static no-go）

固定 full 每轮有 150 个 tail，tail 占六类投影时间的加权份额约
`18.1855%`。但要让 full 吞吐单靠 tail 增加 `1.5%`，需把全部
tail 投影时间降低 `71.0618%`。首轮最值得枚举的六个 exact
keys 即使完全免费，full 也只预测 `+0.0682%`，因此不启动 GPU
solution 枚举。证据在
`/public/home/tangyu408/testdata/goal_runs/20260712_r27_h10_17_tail_m_rocblas_static_audit`。

#### P8 / H10.18：H10.16 + D3 S32 + Hg3 可消融组合

组合上限按每条固定请求重建：H10.16 对每个 full M4096 chunk 减去
confirmed wall `35.685664 ms`；D3 按 `(output_len-1)*16` 次 full-attention
decode 调用与三档 S32-vs-AITER wall delta 计算；Hg3 按首个 no-state
exact chunk 和后续 stateful exact chunks 分别乘 48 个 GDN 层。三项节省以
绝对毫秒相加后再重算 duration/吞吐，不直接相加百分比。

| 套件 | H10.16 | D3 S32 | Hg3 | 三项 bundle |
| --- | ---: | ---: | ---: | ---: |
| all3 加权吞吐上限 | `+0.9877%` | `+1.1962%` | `+0.1424%` | `+2.3552%` |
| full 加权吞吐上限 | `+0.6902%` | `+1.4664%` | `+0.0998%` | `+2.2800%` |

bundle 的 all3/full 三档均为正，预测 full 加权 TTFT/TPOT 分别降
`3.0550% / 1.9310%`，score 从 `88.548456` 到 `89.170302`。这仅是允许
实施的必要上限，不是性能结论。production 必须为 TunableOp、S32 和
Hg3 提供三个独立 env gate，新进程可做逐项消融；完成 wheel/安装/
数值、Tunable hit+Default miss、S32 decode hit+fallback 和 Hg3 exact+fallback 门后，
仅运行一次固定 `all 3` 且外窗口 >=600 秒。加权吞吐未达 `+1%`、
任一档为负或输出漂移时立即回退，不进 full。

上述必要门通过后已实施三个 fail-closed 源码开关：TunableOp 仅在
wheel 内 profile 名和 SHA256 显式给出时开启；S32/Hg3 默认值为 `0`，
只接受 `{1,true,yes,on}` 开启，未知值也关闭。combined wheel SHA256 为
`f877d08fdf2380a87298006c915d14077ca947225e50e5bcf56e028fc9075d80`；
repo/build/lib/wheel/site-packages 六个关键 Python/CSV 文件四方按字节一致。

安装态 S32 的 3 contexts×3 seeds 和 Hg3 的 3 seeds×no/stateful 数值门
全部 mismatch=0/repeat-exact；S32 stage `216 VGPR/32 KiB/0 spill`、Hg3
`156 VGPR/8 KiB/0 spill`，两者 HSACO 都包含 MMAC。Tunable 独立安装态
canary 证明 4 个 exact `ResultEntry` 命中、shared out miss-to-Default、输出 finite，
且 online tuning/record 均关闭。fresh combined route service 还确认 graph capture
完成、S32/Hg3 各命中一次、三个长 prompt 都正确返回，清理后 8001
和进程组为空。

性能服务使用另一个 fresh confirmed/1/1 进程，固定 all3 完成
`661.937322 s`、3 个完整 active rounds，idle padding=0。三轮的 weighted
吞吐改善为 `+1.57313% / +1.60257% / +1.60315%`，均值
`+1.59295%`；三档均为正，failed=0，TTFT 与 pooled TPOT SLA 均通过。
但逐请求对照发现两个稳定漂移：8–16K 第 3 条从 `498→497`，
16–32K 第 2 条从 `259→265`，且文本 hash 也改变；两者在三轮中
完全复现。因此输出硬门失败，`advance_to_full=false`、
`rollback_required=true`；没有运行 full 或 accuracy。candidate 服务已停止，
三项源码已使用冻结补丁精确回退，工作树恢复 clean HEAD；安装态正回装
pinned H11.5+H10.8 baseline wheel。

#### P9 / H10.19：targeted output exact replay 与跨态合同门

C000 的 retry0–5 统一归类为 harness/contract-only：这些尝试只用于修正
service readiness、清理和请求合同，没有生成两条均满足 exact-replay 合同的
可比输出，不能并入任何 feature-state 对照。retry5 发送的是 JSONL raw prompt，
绕过了历史 `CustomDataset.sample(skip_chat_template=false)` 的客户端预渲染，
因此该请求即使返回成功也不是历史 `custom + openai-chat` 请求，必须剔除。

retry6 使用冻结的 rendered prompt 再作为 `user` content 发送，精确复刻历史
benchmark 的客户端预渲染与服务端二次 chat-template 路径。candidate wheel 的
C000（H10/S32/Hg3=`0/0/0`）output canary 与 route-marker audit 均为 PASS：

| 请求 | request payload SHA256 | `usage.prompt_tokens` | 冻结 baseline 输出 |
| --- | --- | ---: | --- |
| 8–16K index2 | `990c3aa5fba4fa045932e8c5e0217acca699bfdabcabbacf85bcde480c2cfacc` | `10207` | `498` tokens，文本 SHA256 `63f45927cfa19bd66bc19b8e7146974de0e0d476d9f0c09afb4b354c981de09a`，exact |
| 16–32K index1 | `5d6f3d217c5c9cb04888fc6076511d8301afe135d875e9d7e62cead1413f74cd` | `22305` | `259` tokens，文本 SHA256 `729dd61de7b35eea3d821e657d89729d9fe436c5ce48d9a6a184c6aa0234fffe`，exact |

这里 benchmark 的 `input_lens=10196/22294` 是客户端 `SampleRequest.prompt_len`
记账值，服务端 usage 是二次渲染后实际 engine prompt token 数；两者定义不同，
不设置彼此相等门。后续每个 state 必须逐请求保持上述 payload SHA256 与
`usage.prompt_tokens=10207/22305` 完全一致；任一不一致先判为 harness/contract
无效，不进入输出归因。合同门通过后才比较 completion token 数与完整文本 SHA256。
证据位于
`/public/home/tangyu408/testdata/goal_runs/20260712_r27_p9_targeted_output_attribution/02_C000_retry6`。
C101（H10/S32/Hg3=`1/0/1`）与 C000 保持相同 payload SHA 和服务端 usage，
H10 四个预期 key、Hg3 HIT、S32 fallback 路由也全部通过；但 correctness 失败：
8–16K index2 为 `517` tokens、文本 SHA256
`5847dbdaf03979d40d54823b98110f278fd47c7c0667e41c36d9ec3f17333484`，
16–32K index1 为 `265` tokens、SHA256
`549e21adb28dfea6769941404eb19ab000830afae945492f6125b7768ace441e`。
后者与 H10.18 C111 exact，前者是新的稳定 decoding basin；两条都不是 baseline。

随后 fresh C100（`1/0/0`）保持同一合同，H10 四 key 命中、S32/Hg3 均走
fallback，两条输出分别恢复为 baseline 的 `498/63f45927…` 和
`259/729dd61d…`，均 exact。C100 与 C101 的唯一 feature 差异是 Hg3，故本次
服务级归因把输出漂移定位到 Hg3 路径；独立微核审计也发现 Hg3 BK64 在
`5 configs × 5 seeds × 2 modes` 共 `40/40` case 相对 production baseline
非 bitwise exact，最大绝对差 `6.103515625e-05`，与该归因一致。

correctness 通过后，C100 使用 candidate wheel、H10 profile on、S32/Hg3 off，
并关闭 TunableOp verbose/tuning/record，连续运行三个完整
`./run_throughput.sh all 3` round。自然活动窗口为 `664.215310295 s`、
idle padding=0、27/27 completed、failed=0，全部逐请求 input/output length 和
完整文本 hash 与冻结 baseline exact；TTFT 和 pooled TPOT SLA 均通过，三档
均无回归。三轮均值为 `13.056762 / 15.886378 / 10.030270 tok/s`，相对 baseline
分别 `+0.835668% / +0.727130% / +1.426132%`，20/50/30 weighted
`+0.958538041%`，未达到冻结的 `+1%` all3 晋级门。因此
`advance_to_full=false`，不运行 full/accuracy；这不是 correctness 或 SLA 失败。

最终已停止服务并回装 pinned H11.5+H10.8 baseline wheel SHA256
`03568ba87ff64fd0a8aade299026d7ee78cbf40d9c1ed5884fb584250b2031f2`；
site/direct_url、clean HEAD source、8001/runtime 和 HCU use `0.0%` 均复验通过。
输出归因证据根为
`/public/home/tangyu408/testdata/goal_runs/20260712_r27_p9_targeted_output_attribution`，
C100 性能与回退证据根为
`/public/home/tangyu408/testdata/goal_runs/20260712_r27_p9_c100_fixed_all3_600`。

#### P9b / H10.20：M4096-scoped TunableOp feasibility（只读 NO-GO）

C100 weighted `+0.958538041%` 与冻结 `+1%` 门之间只差
`0.041461959 pp`。只读调用路径审计确认，当前服务设置
`PYTORCH_TUNABLEOP_ENABLED=1`；PyTorch 会缓存该环境值并使其优先于 Python
setter，所以现有部署不能在每个 step 之间可靠地 `enable(False)`。若未来实施，
必须不设置该环境变量，先加载/校验四行 profile，再由 Python API 保持默认关闭。

历史 verbose route canary 在 application ready 后记录 1728 个 H10 hit 和
1643 个 untuned miss。按实际签名拆分：M4096 forward 内的 shared-out/N96
共有 1008 个 miss，整个 M4096 step 开启时仍无法移除；两个 tail 为 608 个，
graph 外 M1 lm-head 为 27 个。因此该设计只能删除 `635/1643=38.65%` 的
runtime misses。固定 all3 每轮有 26 个 full chunk、9 个 tail 和 1135 个
decode output token，外推约删除 `9*304+1135=3871` 次 lookup/round；结合三档
实际 duration/miss 分布，要补足 `0.041462 pp`，每次被删除的 lookup 平均需
约 `8.7 us`。当前没有证明该成本达到 8.7 us 的测量，且 C100 三轮
`0.937260/0.972017/0.966338%` 的跨度已与缺口接近。

API 静态证据也不支持直接生产化：`TunableOp` 启用时会构造签名并进入带 mutex
的 `TuningResultsManager::Lookup`，但 `TuningContext.enable_` 本身是普通、
非 atomic 的 `bool`，setter 是进程级全局 byte store。当前单 worker 范围可通过
owner-thread 断言 fail closed，但该接口不是通用线程安全的逐请求开关。decode
model GEMM 已由 CUDA graph 捕获，关闭 TunableOp 不会从 graph replay 中再删除
逐 GEMM lookup；可删除的 decode 部分主要是 graph 外 lm-head。

所以 H10.20 为明确 NO-GO：不修改源码/安装，不启动 `>=600 s` 性能。未来只有
在独立证据证明 miss 成本至少约 `10 us/call`、且预计留出超过噪声的裕量时才复议。
最小 defer patch 为：

1. loader 拒绝/取消 `PYTORCH_TUNABLEOP_ENABLED` 环境固定值，profile 加载后默认
   `tunable.enable(False)`；
2. 仅在 `gpu_model_runner.execute_model()` 的 `_model_forward` 周围按实际
   `num_tokens_padded == 4096` 开启，并在 `finally` 中关闭，使后续
   `compute_logits`、tail 和 decode 保持 Default；
3. `_dummy_run(4096)` 使用同一 scope 预热 selected solution，当前小尺寸 decode
   graph capture 保持关闭；
4. 保留 TP/PP/DP=1 范围校验，再增加 owner-thread/非重入断言；任何跨线程调用
   fail closed。

#### P10a / R28-A：正式性能 NO-GO

R28-A 的两个目标 shape、四个 config 全部通过 compile、correctness 和 resource
前门；四个 config 均含 MMAC 且 `0 spill`。单一正式随机交错窗口自然完成
`600.0428034258075 s`、`8664` 个完整 balanced sweeps、`103968` samples。
gate/up 的 best Triton `aiter_256x256x64` median `6.320472 ms`，auto/H10 为
`4.373435/4.102876 ms`；down best Triton `3.047676 ms`，auto/H10 为
`2.452157/2.302397 ms`。双 shape 加权 reduction `-37.2634%`，故 P10a 最终
NO-GO，不进入 production。证据冻结在
`/public/home/tangyu408/testdata/goal_runs/20260712_r28_static_candidate_screening`。

#### P10b / R28-B：H11.5 P/V LDS physical swizzle 资源 NO-GO

source-level `offs_n`/K/V token XOR 不满足语义门：统一 token permutation 会改变
softmax sum 与 P@V K-reduction 的浮点顺序，feature-dependent permutation 则会
计算错误的 QK token。R28-B 因此只在冻结 H11.5 TTGIR 的物理 shared encoding
上测试 P-only、V-only、P+V 三项 `vec4/perPhase1/maxPhase16` 自反 XOR；逻辑
P/V index、全局 load、两个 dot、softmax 和 MMOP 数均保持不变。

三项都经 installed Triton IRSource 成功编译，均为 128 个静态 BF16 MMAC、
32768 B LDS、零 spill；首个 launch 后 VGPR 分别为 `222/232/217`，全部超过
预注册 H11.5 上限 `216`。因此 front gate 给出
`rejected_before_pmc/survivors=[]`，没有 PMC、latency 或性能窗口。证据冻结在
`/public/home/tangyu408/testdata/goal_runs/20260712_r28_b_lds_swizzle_preflight`；
不事后放宽 217/216 门，不做 production integration。

B2 随后只保留三种同时改变 P/V 的新 physical encoding：
`vec4/perPhase1/maxPhase8`、`vec2/perPhase1/maxPhase16` 和
`vec8/perPhase1/maxPhase8`。exact diff/hash 由 `PREPARED_B2.json` 冻结；三者
compile/assembly 均保持两个 dot、128 MMAC、32768 B LDS、0 spill，但首次 untimed
launch 分别暴露 `225/233/219 VGPR`，仍全部超过 216。B2 因此同样
`rejected_before_pmc/survivors=[]`，没有继续 39-case correctness，没有 PMC 或性能。
post-run 8001/runtime/GPU clean。

#### P10c / R28-C：exact BK32 Hg3 group2+1 前门 NO-GO

R28-C 只在 evidence 目录构造两个 kernel launch：第一 launch 让每组三个 value
heads 中的前两个共享一次 raw QK；第二 launch 只计算第三 head，并逐行保持
single-head BK32 baseline math。每组三 head 的 dot work 从
`3*(QK+QH+AV)=9` 降到 8，只删除一份重复 QK；每个 dot 的 K 累加仍按 BK32
顺序，pair 和 singleton code object 都含 MMAC。

compile、5 seeds×no-state/stateful bitwise 与 resource 前门结果为：

| pair config | bitwise exact | pair VGPR/LDS/spill | singleton VGPR/LDS/spill | 决定 |
| --- | ---: | --- | --- | --- |
| `w2/s2 baseline layout` | 10/10 | `227 / 16 KiB / 0` | `117 / 12 KiB / 0` | resource FAIL，超过 `216` 11 VGPR |
| `w4/s1/mi16/kpack2` | 0/10；repeat 10/10 | `130 / 8 KiB / 0` | `117 / 12 KiB / 0` | bitwise FAIL |
| `w8/s1/mi16/kpack2` | 0/10；repeat 10/10 | `76 / 8 KiB / 0` | `117 / 12 KiB / 0` | bitwise FAIL |

两个 resource-valid config 的最大绝对差均为 `6.103515625e-05`，且候选自身
repeat exact；这仍不满足生成输出的 bitwise 门。唯一 bitwise config 又违反
`<=216 VGPR`，因此 exact/resource Pareto 为空，按协议直接
`NO_GO_FRONT_GATE_FAILED`，没有 event、短测或 `>=600 s` 性能字段。

H10-only measured `+0.958538041%`，距离 all3 `+1%` 还差
`0.041461959 pp`。按旧 Hg3 `20.634% kernel -> +0.142432 pp all3` 线性筛选，
未来 exact R28-C 至少需约 6.0% `chunk_fwd_o` reduction；group2+1 的半份
group3 结构收益曾有约 `+0.0712 pp` 的纸面上限，但本次未过前门，不能把它写成
性能结果或与 H10 合并晋级。

证据冻结在
`/public/home/tangyu408/testdata/goal_runs/20260712_r28_c_hg3_bk32_group2plus1_preflight`；
repo、wheel、service 均未修改，post-run 8001/runtime 为空、HCU use `0.0%`、
memory use `0%`。

#### P10d / R28-D：both-shared perPhase 最终资源 NO-GO

R28-D 只改变 R28-B 合法物理 LDS XOR 的 `perPhase/maxPhase`，三项为
`vec4/pp2/mp8`、`vec4/pp2/mp16`、`vec4/pp4/mp8`。逻辑 P/V、全局 offset、
两个 dot、softmax、MMOP 数和 token 累加顺序不变；8/16 个 bank shifts 仍保留
理论 `>=20%` conflict 降幅假设。实际 IRSource 均编译为 128 MMAC、0 spill、
32768 B LDS，但 VGPR 为 `223/233/229`，全部超过冻结 216 门。
因此在 bitwise/PMC/性能前 `survivors=[]`，该 LDS encoding 分支结束。
证据冻结在
`/public/home/tangyu408/testdata/goal_runs/20260712_r28_d_lds_perphase_preflight`。

#### P10e / R28-E：BQ32/BK32 group2+1 正式性能 NO-GO

R28-E 仅把 R28-C pair 的 query tile 从 BQ64 降到 BQ32，在一个 BT64 CTA 内
串行两个 q tiles；singleton 源码逐字节复用 R28-C。passing config
`bq32_bk32_pair_w2_s2` 在 5 seeds×no-state/stateful 为 10/10 bitwise/repeat
exact；pair 为 `201 VGPR/14336 B LDS/0 spill/128 static MMAC`，singleton 为
`117 VGPR/12288 B/0 spill/144 static MMAC`，通过资源与 MMAC 前门。

预授权冻结静态表曾把 major IO 估为 355.5 MiB（baseline 385.5 MiB），并与
`-13.3333%` FLOPs ceiling 一起作为一次正式窗的依据。窗后源码复核发现该表漏计
pair 第二个 q tile 内的 H/V 重载：修正后 H/V 为 `160/80 MiB`，candidate total
为 `451.5 MiB`，相对 baseline **增加 `17.1206%`**，不是降低 `7.7821%`。
冻结原 JSON 不改，另有 `STATIC_UPPER_BOUND_CORRECTION` 显式覆盖其 IO 字段。

窗口使用 explicit raw BK32/BV64/w2/s2 production math 为 baseline，candidate
sample 包含 pair+singleton 两次 launch。每个完整 sweep 随机平衡两 mode×两
method；每个 cell 前在同 stream 写 256 MiB changing-value int32 flush，且 flush
位于 start event 外。窗口自然完成
`600.000565983 s / 188864 sweeps / 755456 samples`：

| mode | baseline median / p90 | candidate median / p90 | median reduction | p90 reduction | bitwise |
| --- | --- | --- | ---: | ---: | --- |
| no-state | `0.434399009 / 0.436639994 ms` | `0.584478974 / 0.587998986 ms` | `-34.5489%` | `-34.6645%` | pre/post exact |
| stateful | `0.434240013 / 0.436479986 ms` | `0.584478974 / 0.587998986 ms` | `-34.5981%` | `-34.7138%` | pre/post exact |

两 mode 的 median `>=6.01%` 与 p90-not-slower 门均失败；harness 与独立审计一致
给出 `NO_GO_PERFORMANCE`。修正后的流量方向也解释了约 34.55% 回归。证据冻结在
`/public/home/tangyu408/testdata/goal_runs/20260712_r28_e_hg3_bq32_bk32_group2plus1_preflight`。
不做 production integration、源码修改或 build/install；post-run 8001 closed、
无相关进程、HCU use `0.0%`、memory use `0%`。

#### P10f / R28-F：剩余 non-MMOP 边界离线静态 NO-GO

R28-F 在 R28 A-E 的失败边界之外，只读检查仍未使用矩阵指令的
Attention/GDN 算子；没有 import Torch/Triton/vLLM、GPU 初始化、编译、launch、
PMC、event 或 latency，也没有修改 production、build/install wheel 或启动服务。
历史 R23 full trace 只用于计算“整 kernel 免费”的乐观 Amdahl 上界，不是性能
结果或 all3 预测。

排序第一的 F1 是把当前 `B1/T4096/H48/BT64` GDN scalar cumsum 按 `BS16`
聚合成每 chunk 三个 `64x64 @ 64x16` FP32 dot CTA，使该边界理论上出现 MMAC。
但 dense 表达每 chunk 为 `196608` MAC，48 个标量 scan 只需 `3024` 次 prefix
addition，MAC/add 比约 65x；下三角非零密度也只有 `50.78125%`，FP32 dot reduction
tree 还存在 bitwise 风险。现有 cumsum trace share 仅 `0.030%`，即使完全免费，
端点也只有 `+0.030009%`，低于 H10-only 尚缺的 `0.041461959 pp`，故在编译前
static NO-GO。

排序第二的 F2 是把 width-4、`dim=10240` 的 GDN prefill depthwise causal-conv
写成 16 token×16 channel 的 block-diagonal MMAC。其 `K64×N16` weight tile 中
只有 `4*16/(64*16)=6.25%` 乘积有效，至少增加 16x 计算，却不删除必要的输入、
conv-state、SiLU 或输出流量；M 约为 1 的 decode update 更差，不纳入候选。
prefill fwd 的整 kernel 免费端点虽为 `+0.370367%`，连同 decode update 也只有
`+0.585407%`，但这种矩阵化无法接近免费假设，因此同样 static NO-GO。

更低优先级的 Attention decode `reduce_segments` 仍须以 VALU 完成 max/exp/sum，
q=1 时 M16 tile 至多 1/16 有效；GDN L2 norm 的 Gram 表达也只使用 16×16 输出的
对角 1/16。二者整 kernel 免费端点分别只有 `+0.078061%/+0.084071%`，不进入
R28-F 候选。最终结论是不启动任何 R28-F `>=600 s` 窗，也不重开 R28-A blocked
GEMM、R28-B/B2/D LDS encoding 或 R28-C/E Hg3 QK 路线。

冻结证据位于
`/public/home/tangyu408/testdata/goal_runs/20260712_r28_f_static_mmac_boundary`：
`SUMMARY.md` SHA256 为
`1835cabcbad5126950a1b90c5145d0c55153f45c25d077265aee8413282b92d6`，
`SOURCE_INPUTS.sha256` SHA256 为
`de3b0f397a0acd130af867b9791b7b72fcefb23ad8632fb44a66492ed775e2b9`，
`EVIDENCE.sha256` SHA256 为
`81393f26e26854e41a77ea9689b87dcd06edce9d0f9b36599ade91265c74870b`。

#### P10g / R28-G：BQ16/BQ32 group3 GPU 前静态 NO-GO

R28-G 仅复用冻结的 `hg3_qk_hoist_kernel.py`，拟检查 primary
`BQ16/BK32/BV64/w2/s2/GROUP_SIZE3` 与 companion BQ32；没有增加其它 config。
在 GPU compile 前按源码循环逐项复核发现，K、三份 H、三份 V 与 head-specific
G 都嵌套在 `i_qtile` 内，必须按 BQ16 四个 qtiles、BQ32 两个 qtiles 重载计数。

explicit raw baseline 的逻辑 major I/O 是
`Q96+K96+H96+V48+O48+G1.5=385.5 MiB`。BQ16 为
`Q32+K128+H384+V192+O48+G7.5=791.5 MiB`，增加 `406 MiB / 105.3178%`；
BQ32 为 `Q32+K64+H192+V96+O48+G4.5=436.5 MiB`，增加
`51 MiB / 13.2296%`。两者都把 FLOPs 从 `16,106,127,360` 降到
`11,811,160,064`（`-26.6667%`）。

R28-E candidate 已是 raw baseline 的 `1.345489x/1.345981x`；要达到所需
`6.01%` kernel reduction，需相对 R28-E 至少恢复
`30.144343%/30.169911%`。BQ32 相对修正后 R28-E 的 I/O 与 FLOPs 分别最多
减少 `3.322259%/15.384615%`，即使不合理地直接相加也只有
`18.706875%`，仍不足。因此在 GPU 前冻结 static NO-GO。

证据位于
`/public/home/tangyu408/testdata/goal_runs/20260712_r28_g_hg3_bq16_bk32_group3_front_gate`；
`FROZEN_INPUTS.json` SHA256 为
`7266dc0149df72821d0533f600bb6e1c395e5a69400875fe9f54d1579557fc7a`。
GPU harness 草稿已删除，未初始化 GPU、未 compile/launch、未做 bitwise/repeat/
resource/MMAC 捕获、未 timing；production repo/wheel/service 均未改变，也不请求
`>=600 s` 正式窗。

#### P10h / R28-H：C110 exact-replay correctness NO-GO

只读复议确认 P9 证据根只有 C000/C101/C100，没有 C110 exact-replay 或独立
performance。现有 C101 漂移只能证明 Hg3 在 S32 off 时足以导致漂移；C100 exact
也不能证明 S32 安全。R28-H 因此只启动一次 fresh C110 output-only service，
使用 candidate wheel `f877d08fdf2380a87298006c915d14077ca947225e50e5bcf56e028fc9075d80`，
配置为 H10 confirmed、S32 on、Hg3 off；没有收集任何 performance 字段。

两条冻结请求的 payload SHA256 与服务端 prompt token 数分别保持
`990c3aa5.../10207`、`5d6f3d21.../22305`。route audit 无 mismatch：H10 四个
exact result key 全部命中，S32 hit 且无 fallback/ablation，Hg3 仅 fallback、
没有 hit。但输出变为：

- 8–16K index2：572 tokens，SHA256
  `4655f859218e8781ac3e9788cf65f5f9feff6e181a699afa8bb60d27f03f8b36`；
- 16–32K index1：289 tokens，SHA256
  `6cb59007203df52ad9672063e751d3c3e380fe5a731770c17b3c926fccc60b18`。

两条 `frozen_baseline_exact_match` 均为 false。canary 顶层 `status=pass` 是旧 P9
脚本的“请求执行成功”状态；脚本只对 C000 强制 baseline control，不能覆盖上述
逐请求 correctness 字段。因此 R28-H 为 `NO_GO_CORRECTNESS`，S32 本身已足以
造成新的输出漂移。为排除 H10×S32 交互，随后 fresh C010 将 H10 也关闭；route
按预期为 H10 disabled、S32 hit/no-fallback、Hg3 fallback/no-hit，两条输出又
逐字节复现 C110 的 `572/4655f859...` 与 `289/6cb59007...`。因此 S32 单独充分，
不执行 `>=600 s all 3`、full 或 accuracy。两个 fresh service 均已
停止，post-service source/runtime/8001 clean；最终回装 pinned baseline wheel
`03568ba87ff64fd0a8aade299026d7ee78cbf40d9c1ed5884fb584250b2031f2`，
site/direct URL 与 candidate-only files absent 均通过。证据根为
`/public/home/tangyu408/testdata/goal_runs/20260712_r28_h_c110_revival`。

#### P10i / R28-I：S32 bitwise-exact salvage 静态 NO-GO

旧 `post_install_gpu_validator.py` 对 S32 reference/candidate 使用
`atol=0.015, rtol=0.01`，再要求 `max_abs<=0.00048828125`；其
`mismatch_count` 只统计超过 `atol+rtol*|reference|` 的元素。README 所写
“zero tolerance mismatches”因此只是超容差计数为零。真正的 exact 字段定义为
`max_abs==0`，九条 context×seed record 全部 `numeric.exact=false`；九条
candidate repeat 才是 bitwise exact。它证明 S32 稳定可重复但不等价于 AITER。

覆盖也只是一层孤立 attention 的随机 BF16 Q/K/V：contexts
`6315/13295/21562`×seeds `2603/2604/20260712`，每条 6144 elements。未覆盖真实
service contexts `10207/22305`、每个后续 decode 长度、64 层 residual/MLP/GDN
传播或 logit top-1 margin。模型含 16 个 full-attention 层；单 token decode 每步
可累计 16 次 S32。572/289-token candidate completion 在首个 prefill token 后约
对应 9136/4608 次 S32 调用，小 BF16 差异经 residual 与 KV-cache feedback 后触发
温度零 argmax 分岔，与 C010 的确定性大幅文本漂移并不矛盾。

源码树审计显示，AITER 在该 cache/head shape 下得到
`BLOCK_M16/BLOCK_SIZE16/NUM_SEGMENTS16`，按 16-token blocks 做 segment 内 online
softmax，再合并 16 个 segment；S32 改为 `M8/logical64/padded64/S32`，并用
`w4/s1/waves1/MI16/kpack2`。QK/P@V dot、online-softmax 更新边界和 32-way reducer
全部改变，仅修改 reducer 无法修复 stage 内误差。

历史 closest-tree `M8/L16/P16/S16` 保留相同 token tile/segment count，但仍改变
M tile/compiler，十条 case 均非 bitwise（max abs `1.5258789e-5` 至
`2.44140625e-4`）。其 stage 是 `185 VGPR/46 SGPR/8 KiB LDS/0 spill`，reducer
`30 VGPR/28 SGPR/4 KiB/0 spill`，历史 weighted GPU gain `16.2191%`；该收益不能
外推给 AITER-equivalent clone。

唯一可信 exact 方向是 literal AITER fallback，或保持 M16/L16/S16、相同
online-softmax/P@V/reducer 与 compiler codegen 的 AITER-equivalent clone。前者相对
AITER 是 0% gain；后者 FLOPs/bytes/segments/launch 不变，只可能删除很小的通用
控制开销，且 codegen 一变仍可能 non-bitwise，当前没有正收益证据。S32+AITER
双算再返回 AITER 约为 `214+269=483 us`，比 AITER 单算约慢 80%，同样 NO-GO。

R28-I 因而在 GPU 前冻结 static NO-GO。未来任何复议必须
`atol=rtol=0,numeric.exact=true`，并先过 frozen service exact replay。证据根：
`/public/home/tangyu408/testdata/goal_runs/20260712_r28_i_s32_exact_salvage_static_audit`；
`EVIDENCE.sha256` SHA256 为
`0bbf67e37fafbb024438033937b363f39c66c0a44956d9a5140eca76e23cbc36`。

#### P10j / R28-J：raw-A 两 kernel 物化静态 NO-GO

R28-J 只做 CPU/stdlib 静态数据流核算。producer 以
`(chunk,Hg)=(64,16)` 的 1024 CTA 网格计算一次 raw QK，并写
`[64,16,64,64]` FP32 `A_raw`，workspace 恰为 16 MiB。consumer 保持
三 value heads 和两个 BV64 tiles，网格仍为 `(2,64,48)=6144` CTA；同一
`A_raw` 因而被 `3*2=6` 个 consumer 读取，总 workspace read 为 96 MiB。

baseline 的 QK/QH/A@V 为 `6.442451/6.442451/3.221225 GF`，总计
`16.106127360 GF`。候选 producer QK 只剩 `1.073741824 GF`，QH/A@V 不变，
总计 `10.737418240 GF`（`-33.3333%`）。但 major I/O 从 baseline 的
`Q96+K96+H96+V48+O48+G1.5=385.5 MiB` 变为
`Q112+K16+H96+V48+O48+G1.5+Awrite16+Aread96=433.5 MiB`，增加
`48 MiB / 12.4514%`。同时 CTA 从 6144 增至 7168，launch 从一条增至两条，
并新增每次调用 16 MiB allocation；因此在 GPU 前即失败 bandwidth/launch/
workspace 门。

R28-E 的 measured candidate/baseline median ratio 为 `1.34549`；从该点恢复并
达到 `+6.01%` 需相对 R28-E 至少 `30.1443%` 降时。R28-J 相对修正后的 R28-E
只减少 `23.0769%` FLOPs 和 `3.9867%` major I/O，即使不合理地直接相加也只有
`27.0636%`，还未计第二 launch 与 allocation，故经验反证同样不支持窗口。

证据冻结于
`/public/home/tangyu408/testdata/goal_runs/20260712_r28_j_raw_a_two_kernel_static`：
`SUMMARY.md`、`STATIC_MODEL.json`、`SOURCE_INPUTS.sha256` 的 SHA256 分别为
`faf9a067c41880352ce9247e5b5c1416f798d67a7dc2e5a01f63dd1de02f0fc3`、
`1a9b98f174cbe12e1d1f124bf2b3c5be4f32e08ed33bd3eddab92e6f75474300`、
`939f7bc3e2f2ec04f603f2269d8d509094befe60abe8cc8977c4efbac62e5cc1`；
`EVIDENCE.sha256` 自身 SHA256 为
`200fbb009bdb15ab6914e95b1d12a7421f136bd97654c9f01b0359f281fdd89b`。
全程没有 GPU init、compile、launch、correctness/resource/MMAC/PMC/timing，
不生成 kernel 草稿、不修改 production，也不申请 `>=600 s` 窗。

#### P10k / R28-K：DCU 矩阵单元覆盖最终综合

R28-K 不生成新实验，只把既有 dynamic PMC、HSACO/ISA、运行态 source import、
rocBLAS solution 与 R28 A-J 边界合并。直接回答为：Full Attention Prefill 的
QK/PV、Attention Decode 3D stage、GDN Prefill 五段主矩阵链和多 token Prefill
Linear 都使用 gfx936 MMAC/MMOP；Attention `reduce_segments`、GDN 辅助
scan/norm/gate/conv、packed recurrent Decode 与 H10.8 `n=1` GEMV 是明确的
标量/向量例外。

关键动态值为 Attention Prefill `20,686,848` MMOP、Decode stage `384,000`，
reducer `0`；GDN Prefill 五类实际 dispatch 全部 MMOP-positive，GDN Decode
为 `MMOP=0/VALU=258,048`；Prefill Linear one-dispatch 为 `22,282,240` MMOP。
Linear 默认实际路径是 hipBLAS→rocBLAS/Tensile `ISA936/MAC_MMAC`，不是
hipBLASLt fallback。显式确认 key 中 full-attention qkv、GDN qkvz、MLP gate/up
使用 `20981`，MLP down 使用 `20979`，但 portfolio wall `5.7185%` 低于
`5.8136%` 门；不写入 production。

R28 A-J 证明“使用矩阵单元”从来不是充分条件：A/E 因性能失败，B/C/D 因资源或
bitwise Pareto 失败，F 因无效矩阵化失败，G/J 因完整 I/O/launch/workspace
失败，H/I 因 S32 reduction-tree correctness 失败。未来优先提高已有 MMAC 的
有效工作率、减少 LDS/HBM wait 与 materialization；向量例外只在 bytes/launch/
Amdahl 有实质上界时优化，且任何改变归约树的候选必须先过 service output exact。

证据根为
`/public/home/tangyu408/testdata/goal_runs/20260712_r28_k_matrix_unit_coverage_synthesis`；
其中 `SOURCE_INPUTS.sha256` 对 45 个历史/源码输入校验通过。全程无 GPU init、
compile、launch、PMC、timing、服务或 production 变更。

### 研究循环

Inner loop：

1. 选择最高优先级的未验证假设。
2. 写实验协议：触发条件、源码路径、预测、失败判据。
3. 做最小源码改动。
4. 重新编译并安装 wheel。
5. 用固定启动和固定吞吐脚本运行。
6. 记录 evidence card，必须列出“累计优化栈”和“本轮新增 diff”。
7. 若失败，写明它排除了什么；若成功，再进入消融。

Outer loop：

1. 每 3-5 个候选或遇到矛盾结果时，重新综合 backend 矩阵和瓶颈归因。
2. 调整 H1-H9 及 H1.x decode 子项优先级。
3. 删除没有路径证据、SLA 证据或精度证据的候选。
4. 只有完整 evidence card 支持的结果才能进入最终结论。
