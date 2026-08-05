# B：GDN/FLA 闭卷实施卡

## 负责范围

- `qwen35_rocm_opt/gdn.py`
- `vllm/model_executor/layers/fla/ops/gfx936.py`
- `chunk.py`、`chunk_o.py`、`chunk_scaled_dot_kkt.py`、`solve_tril.py`、`wy_fast.py`
- `vllm/v1/attention/backends/gdn_attn.py`
- `vllm/model_executor/models/qwen3_next.py` 的 GDN 类
- `vllm/model_executor/models/qwen3_5.py` 的 GDN forward

不要修改 attention backend、HIP GEMV 或 M-RoPE runner。

## vLLM/FLA 依赖

1. 五个 FLA kernel 必须继续使用宿主原实现；本模块只向 Triton autotune
   decorator 注入 `early_config_prune`。
2. packed decode 的 recurrent kernel 由宿主提供给
   `launch_packed_decode(kernel, **args)`。必须逐项核对参数顺序和 stride。
3. `GDNAttentionMetadataBuilder` 必须能在 CPU 侧访问每个 request 已计算 token
   数，避免从 GPU `has_initial_state` 做同步。
4. `Qwen3NextGatedDeltaNet` 必须仍有 prefill warmup、prefill/decode 分流、
   state cache、spec/non-spec merge 和调用方 output buffer。

## 不变量

- Q heads=16，V heads=48，K/V dim=128，chunk BT=64。
- T=16/32/64/4096 的固定配置与 compiler flags 不变。
- packed decode grid=`(4,B*48)`，不是固定 48。
- 无历史 state 且 native GDN 时允许 `initial_state=None`；混合 batch 必须 gather
  并按 mask 清零。
- `chunk_o` 可写调用方 output；只清未写 padded tail，不能重新整块 zeros。
- fused RMSNorm 的 Z stride 为 `(16384,128,1)`，公式仍是 RMSNorm×SiLU(Z)。

## 闭卷步骤

1. 给五个 autotune kernel 增加同一个 pruner import 和 decorator 参数。
2. 把 optional `output` 从 `chunk_gated_delta_rule` 逐层传到 `chunk_fwd_o`。
3. metadata builder 计算三态值：全无 state=`False`、全有=`True`、混合=`None`。
4. warmup 必须覆盖 T=4096、state/no-state 和 output-final-state 组合，并发生在 KV
   cache 分配前；这是 OOM 修复的一部分。
5. prefill 复用 output，decode 注入 host recurrent kernel，所有返回路径清 tail。
6. Qwen3.5 output 以 portable fused RMSNorm+SiLU 替换展开的 reshape/norm/gate。

## 独立验收

- pruner 对每个目标 shape 只返回一个冻结 config，对非 gfx936 返回原 configs。
- packed decode B=1/2/4/8，确认全部 `B*48` heads 被写入。
- no-state、stateful、mixed state 三类 prefill 与冻结 499 输出一致。
- output alias 的 data pointer 不变；真实 token 已写，tail 为零。
- fused norm 当前 portable/frozen 同输入误差不超过 `3.1e-5`，BF16 allclose。
- 首次冷启动完成 warmup 后再分配 KV cache，双 rank 不出现一侧 cache 较小。
