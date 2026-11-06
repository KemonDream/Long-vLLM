// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

namespace onecat_v37 {
torch::stable::Tensor sm70_d256_gqa_v37_fwd(const torch::stable::Tensor& q,
                                              const torch::stable::Tensor& k,
                                              const torch::stable::Tensor& v,
                                              torch::stable::Tensor& out,
                                              double scale, bool causal);
}

// Separate symbol: a stale legacy FA2 library must never masquerade as v37.
STABLE_TORCH_LIBRARY_FRAGMENT(_vllm_fa2_C, ops) {
  ops.def(
      "sm70_d256_gqa_v37_fwd(Tensor q, Tensor k, Tensor v, Tensor(a!) out, "
      "float softmax_scale, bool causal) -> Tensor(a!)");
}
STABLE_TORCH_LIBRARY_IMPL(_vllm_fa2_C, CUDA, ops) {
  ops.impl("sm70_d256_gqa_v37_fwd", TORCH_BOX(&onecat_v37::sm70_d256_gqa_v37_fwd));
}
