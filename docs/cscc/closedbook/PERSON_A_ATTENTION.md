# A：GQA6 Attention 闭卷实施卡

## 负责范围

- `qwen35_rocm_opt/attention.py`
- `qwen35_rocm_opt/target.py`（只读公共文件）
- `vllm/platforms/rocm.py::_get_backend_priorities`
- `vllm/v1/attention/backends/rocm_aiter_unified_attn.py`

不要修改 GDN、GEMV、M-RoPE 或服务参数。

## vLLM 依赖

1. `RocmAiterUnifiedAttentionImpl.__init__` 提供 head 数、KV head 数、head size、
   dtype、window/alibi/sinks 与 attention type，用来建立一次性的精确 gate。
2. `forward` 必须能取得 `query_start_loc`、`seq_lens`、`max_query_len`、
   `max_seq_len`、`block_table` 和外部 output buffer。
3. vLLM/AITER 负责先把本轮 K/V 写入 cache；portable kernel 只读 cache。
4. AITER `unified_attention` 是 decode、FP8、非 gfx936 和非精确模型 shape 的
   fallback，不属于可删除代码。

## 固定张量契约

| 张量 | 契约 |
| --- | --- |
| Q | `[tokens,24,256]`，BF16 |
| K/V cache | `[physical_pages,page_size,4,256]`，BF16；page size 当前为 64 或 784 |
| block table | `[sequences,max_logical_pages]`，每项是 physical page id |
| query starts | `[sequences+1]` prefix sum |
| sequence lengths | `[sequences]`，包含本轮 query 后的总长度 |
| output | 与 Q 同 shape，由宿主分配 |

## 闭卷步骤

1. 确认官方版本可导入 AITER unified attention；存在时把该 backend 放入
   ROCm backend priority，官方 fallback 始终保留。
2. 在 attention impl 初始化时建立 gfx936/Qwen3.5 精确 gate：24 Q heads、
   4 KV heads、head size 256、decoder、无 alibi/window/sinks/softcap。
3. forward 中只在 `max_seqlen_q>1` 且 Q/cache/output 都为 BF16 时调用
   `qwen35_rocm_opt.attention.prefill`；其他情况调用宿主原实现。
4. 不改变 KV cache shape、cache update 顺序、output ownership 或 decode 路径。

## 独立验收

- T=16 短 prefill：portable 与冻结 499 kernel 对齐。
- T=128、context=760、page=784：必须覆盖相邻 page 的跨页读取；当前同输入
  对照为逐位一致。
- T=1 decode：确认未进入 portable prefill。
- B>1：确认 grid 第三维等于 sequence 数，block table stride 来自宿主张量。
- 不安装 AITER：backend 选择不得在 import 阶段崩溃。

迁移到新框架时只需提供表中七个张量/标量；不得把 scheduler 或 request 类型
传入 portable kernel。
