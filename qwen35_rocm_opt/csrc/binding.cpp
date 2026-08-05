// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>

torch::Tensor qwen35_k5120_gemv(torch::Tensor weight, torch::Tensor input,
                                int64_t rows_per_block);

TORCH_LIBRARY(qwen35_rocm_opt, ops) {
  ops.def("k5120_gemv(Tensor weight, Tensor input, int rows_per_block) -> Tensor");
}

TORCH_LIBRARY_IMPL(qwen35_rocm_opt, CUDA, ops) {
  ops.impl("k5120_gemv", &qwen35_k5120_gemv);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {}
