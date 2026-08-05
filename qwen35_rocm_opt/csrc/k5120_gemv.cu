// SPDX-License-Identifier: Apache-2.0
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <hipcub/hipcub.hpp>

template <typename T>
__device__ __forceinline__ T load_nontemporal(const T* address) {
  return __builtin_nontemporal_load(address);
}

__device__ __forceinline__ float4 load_float4(const float4* address) {
  const auto* values = reinterpret_cast<const float*>(address);
  return make_float4(load_nontemporal(values), load_nontemporal(values + 1),
                     load_nontemporal(values + 2),
                     load_nontemporal(values + 3));
}

template <int Rows>
__global__ __launch_bounds__(640) void k5120_pairreduce_kernel(
    const c10::BFloat16* __restrict__ weight,
    const c10::BFloat16* __restrict__ input,
    c10::BFloat16* __restrict__ output) {
  constexpr int Threads = 640;
  using BFloat162 = __hip_bfloat162;
  using BlockReduce = hipcub::BlockReduce<float, Threads>;

  const auto* weights = reinterpret_cast<const float4*>(weight);
  const auto* inputs = reinterpret_cast<const BFloat162*>(input);
  auto* outputs = reinterpret_cast<__hip_bfloat16*>(output);
  __shared__ typename BlockReduce::TempStorage reductions[Rows];

  const int thread = threadIdx.x;
  const int row_start = blockIdx.x * Rows;
  float2 input_values[4];
  float4 weight_values[Rows];

#pragma unroll
  for (int index = 0; index < 4; ++index) {
    input_values[index] = __bfloat1622float2(inputs[thread * 4 + index]);
  }
#pragma unroll
  for (int row = 0; row < Rows; ++row) {
    weight_values[row] = load_float4(&weights[(row_start + row) * Threads + thread]);
  }

  float accumulators[Rows] = {};
#pragma unroll
  for (int row = 0; row < Rows; ++row) {
    auto* pairs = reinterpret_cast<BFloat162*>(&weight_values[row]);
    float low = 0.0f;
    float high = 0.0f;
#pragma unroll
    for (int index = 0; index < 4; ++index) {
      const float2 value = __bfloat1622float2(pairs[index]);
      low = fmaf(value.x, input_values[index].x, low);
      high = fmaf(value.y, input_values[index].y, high);
    }
    accumulators[row] = low + high;
  }

#pragma unroll
  for (int row = 0; row < Rows; ++row) {
    const float total =
        BlockReduce(reductions[row]).Reduce(accumulators[row], hipcub::Sum{});
    if (thread == 0) {
      outputs[row_start + row] = __float2bfloat16(total);
    }
  }
}

torch::Tensor qwen35_k5120_gemv(torch::Tensor weight, torch::Tensor input,
                                int64_t rows_per_block) {
  TORCH_CHECK(weight.is_cuda() && input.is_cuda(),
              "weight and input must be GPU tensors");
  TORCH_CHECK(weight.device() == input.device(),
              "weight and input must use the same device");
  TORCH_CHECK(weight.scalar_type() == at::kBFloat16 &&
                  input.scalar_type() == at::kBFloat16,
              "weight and input must be BF16");
  TORCH_CHECK(weight.is_contiguous() && input.is_contiguous(),
              "weight and input must be contiguous");
  TORCH_CHECK(weight.dim() == 2 && input.sizes() == at::IntArrayRef({1, 5120}),
              "expected weight [M,5120] and input [1,5120]");
  const int64_t rows = weight.size(0);
  TORCH_CHECK(weight.size(1) == 5120, "weight K must equal 5120");
  TORCH_CHECK(
      (rows_per_block == 4 && rows == 96) ||
          (rows_per_block == 2 &&
           (rows == 14336 || rows == 16384 || rows == 34816 ||
            rows == 248320)),
      "unsupported Qwen3.5 M/rows_per_block combination");

  auto result = torch::empty({1, rows}, input.options());
  const at::cuda::OptionalCUDAGuard guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* weight_ptr = weight.data_ptr<c10::BFloat16>();
  const auto* input_ptr = input.data_ptr<c10::BFloat16>();
  auto* output_ptr = result.data_ptr<c10::BFloat16>();
  if (rows_per_block == 4) {
    k5120_pairreduce_kernel<4><<<rows / 4, 640, 0, stream>>>(
        weight_ptr, input_ptr, output_ptr);
  } else {
    k5120_pairreduce_kernel<2><<<rows / 2, 640, 0, stream>>>(
        weight_ptr, input_ptr, output_ptr);
  }
  return result;
}
