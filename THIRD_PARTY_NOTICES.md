# 第三方代码与许可证说明

本文件与 README 顶部声明共同满足第三方知识产权、开源代码、公开算法和
第三方库的披露要求。

## 1. vLLM

- 原始第三方项目：vLLM
- 原始上游：<https://github.com/vllm-project/vllm>
- 直接代码基线：OpenDAS `vllm_cscc` fork
- fork 地址：<http://developer.sourcefind.cn/codes/OpenDAS/vllm_cscc.git>
- 提交前基线 commit：
  `fa718036bdb9dfd80a872b86c8ac16c9d02bfd31`
- 版本：0.18.1
- 许可证：Apache License 2.0
- 本仓库保留：`LICENSE`、各源码文件 SPDX/版权头和
  `README_UPSTREAM.md`

本作品在该基线上修改 ROCm custom ops、GDN 路径、AITER attention wrapper
及 Qwen3.5 执行路径。

## 2. AMD AITER unified attention

- 项目：AMD AITER
- 上游：<https://github.com/ROCm/aiter>
- 验证安装版本：`0.1.dev1+g9daa788.d20260401`
- 参考文件：`aiter/ops/triton/unified_attention.py`
- AITER 安装分发包随附许可证：MIT
- 参考源文件自身 SPDX：Apache-2.0，Copyright contributors to the vLLM project
- 本提交对应文件：
  `vllm/v1/attention/ops/rocm_aiter_unified_attention_gqa6.py`、
  `vllm/v1/attention/ops/rocm_aiter_decode_attention_gqa6.py`

前一文件是对非 segmented 2D unified-attention 算法的窄范围特化，保留
online-softmax 和 cache-block 选择语义，并为 gfx936/BF16/head256/GQA6
修正 query-head 行重叠及增加 H11.5 wide-causal 编译配置。后一文件复用 AITER
segmented 3D 主 kernel，并由 AITER `reduce_segments` 派生 masked 20-of-32
FP32 reduction，使四个 KV heads 恰好产生 80 个主 workgroups。两个提交文件
均保留 Apache-2.0 SPDX 与 vLLM contributors 版权头。

## 3. FlashAttention / 平台 ROCm 分发包

- 项目：FlashAttention / FlashAttention-2
- 上游：<https://github.com/Dao-AILab/flash-attention>
- 平台预装版本：
  `2.8.3+das.opt1.dtk2604.torch2100.20260330.g3f0061`
- 平台包 metadata 许可证分类：BSD License；上游许可证为 BSD-3-Clause
- 本提交调用接口：`flash_attn_2_cuda.varlen_fwd` 与
  `flash_attn.flash_attn_interface.varlen_fwd_unified`
- 本提交对应文件：
  `vllm/v1/attention/ops/rocm_page784_split_attention.py`

新 wrapper 调用评测容器预装的 contiguous/paged attention 二进制接口，不在
仓库中重新分发其二进制或复制其 kernel 源码。仓库内新增代码负责 page784 的
`768 + 16` 排布、Triton residual pack、有界 workspace 和 FP32 LSE state
merge；第三方 API 与自研封装边界据此区分。

## 4. flash-linear-attention

- 项目：flash-linear-attention
- 上游：<https://github.com/sustcsonglin/flash-linear-attention>
- 原作者：Songlin Yang、Yu Zhang
- 原始版权：Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
- 许可证：MIT
- 本提交对应文件：
  `vllm/model_executor/layers/fla/ops/fused_recurrent.py`

对应源码文件头已保留原作者、来源和 MIT 许可说明。

## MIT License text

The following text reproduces the AITER package-level MIT license and applies
to the flash-linear-attention-derived portions identified above. The referenced
`aiter/ops/triton/unified_attention.py` source file itself is marked
Apache-2.0 and is treated as Apache-2.0; the package-level MIT notice does not
override that file-level SPDX declaration:

```text
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 5. 平台预装依赖

PyTorch、Triton、transformers、DTK/ROCm、AITER 和 FlashAttention 由评测
容器预装，本仓库不重新分发这些二进制包，也不提交模型权重。其使用仍分别
受各自许可证约束。
