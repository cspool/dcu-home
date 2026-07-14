# Prefill GEMM DCU matrix-unit probe

Date: 2026-07-12

## Conclusion

Yes.  The representative Qwen3.5-27B BF16 prefill projection tested here is
dispatched through hipBLAS/rocBLAS, and the exact rocBLAS kernel observed both
in this micro-probe and in the existing production vLLM trace contains
`v_mmac_f32_16x16x16_bf16` instructions.  On gfx936 this is the BF16 matrix
multiply-accumulate path (the DCU matrix/Tensor compute unit, rather than only
ordinary scalar/vector FMA).

This establishes use, not the achieved matrix-unit occupancy or utilization.

## Environment

- Device ISA: `gfx936:sramecc+:xnack-`
- PyTorch: `2.10.0`
- Clean-process PyTorch BLAS preference: `_BlasBackend.Cublas`; on this ROCm
  build that name selects the hipBLAS/rocBLAS path.
- rocBLAS: `4.3.0` (`librocblas.so.4.3`)
- hipBLAS: `2.3.0` (`libhipblas.so.2.3`)
- hipBLASLt: `0.10.0` (`libhipblaslt.so.0.10`)

## Representative prefill operation

The tested operation is Qwen3.5's large MLP gate/up projection:

```text
x[512, 5120] @ weight[34816, 5120].T -> y[512, 34816]
dtype: BF16 input/weight/output, FP32 accumulation
```

The single-event time after two warmups was `0.715359 ms`, corresponding to
about `255.17 TFLOP/s` for `182,536,110,080` GEMM FLOPs.  This timing is only a
sanity signal; the instruction evidence below is the decisive result.

## rocBLAS API evidence

`rocblas_trace.log` records the call made by `torch.nn.functional.linear`:

```text
rocblas_gemm_ex,T,N,34816,512,5120,...,bf16_r,...,bf16_r,...,bf16_r,...,f32_r,...
```

Thus this representative prefill GEMM is not an unidentified eager/Triton
fallback; it reaches `rocblas_gemm_ex` through hipBLAS.

`rocblas_trace_clean_default.log` repeats the call in a fresh process without
setting a preferred backend.  The process reports
`preferred_clean _BlasBackend.Cublas` and emits the same `rocblas_gemm_ex`
record, excluding the possibility that rocBLAS was reached only because this
probe forced it.

## Runtime kernel evidence

The hipprof trace is under `trace/`.  Its top GEMM is:

```text
Cijk_Alik_Bljk_BBH_MT256x256x16_..._ISA936_..._MAC_MMAC_..._WGM8
```

It ran 3 times with an average kernel duration of `674.506 us`.  The exact same
kernel name is present in the existing production vLLM trace at:

```text
testdata/profile_runs/hipprof_fixed_trace_clean_16_32K_n4_20260711_003100/
  hipprof/vllm_fixed_trace.hipkernel.csv
```

There it ran 2,304 times, totaling `4.299497368 s`.

The earlier phase-labelled PyTorch production trace provides an independent
prefill/decode separation:

```text
testdata/profile_runs/torch_profile_20260707_codex/contexts/16-32K/raw_traces/
```

`profiler_out_0.txt` labels five
`execute_context_1(4096)_generation_0(0)` calls as 4096-token prefill chunks.
The compressed trace's external-id linkage is
`prefill -> vllm::rocm_unquantized_gemm -> aten::linear/matmul/mm ->
Cijk...ISA936...MAC_MMAC`, so the production match is not inferred only from a
mixed prefill/decode kernel summary.

## Instruction evidence

The matching kernel symbol is at `0x1791300` in:

```text
/opt/dtk-26.04-DCC2602-0317/lib/rocblas/library_gpu5/
TensileLibrary_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx936.co
```

Disassembling it with the installed LLVM objdump shows instructions such as:

```text
v_mmac_f32_16x16x16_bf16 v[0:3], v[128:129], v[192:193], v[0:3]
v_mmac_f32_16x16x16_bf16 v[4:7], v[132:133], v[192:193], v[4:7]
```

There are 640 static occurrences in the first 64 KiB of this kernel; the next
global kernel symbol starts after that inspected interval.

## Runtime MMOP counter evidence

`rocprof_mmop.txt` filters the same `MT256x256x16` kernel and collects the raw
matrix/vector issue counters while running `probe_prefill_gemm.py`.  The single
GEMM dispatch in `rocprof_mmop.csv` reports:

| counter | value |
| --- | ---: |
| `SQ_INSTS_MMOP` | `22,282,240` |
| `SQ_ACTIVE_INST_MMOP` | `44,564,480` |
| `SQ_INSTS_VALU` | `24,525,696` |
| `SQ_ACTIVE_INST_VALU` | `46,807,936` |
| `SQ_WAVES` | `3,264` |

The nonzero `SQ_INSTS_MMOP` is direct runtime evidence that the profiled
prefill GEMM issued matrix-operation instructions.  These are raw/emulated
per-SIMD counters, so they should not be converted directly into an occupancy
percentage without the platform counter semantics and elapsed-cycle counters.

## Scope and caveat

- Current custom `LLMM1/LLMM1Strided` gates only cover `n == 1` decode shapes.
  Normal multi-token prefill projections fall through to `F.linear`, so the
  rocBLAS result is the relevant production path for these large projections.
- This probe answers whether the matrix unit is used: yes.
- It does not prove that every tiny/ragged prefill GEMM uses MMAC, nor quantify
  matrix-unit occupancy across the whole prefill.  A phase-labelled counter run
  over all projection/token-chunk shapes would be needed for that utilization
  breakdown.
