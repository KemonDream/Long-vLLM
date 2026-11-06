// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <torch/csrc/stable/tensor.h>

namespace vllm::awq_sm70 {

bool sm70_fp8_prefill_cutlass_out(torch::stable::Tensor out, torch::stable::Tensor in_feats,
                                  torch::stable::Tensor dense_weight, bool gated_silu);

}  // namespace vllm::awq_sm70
