// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <cuda_runtime_api.h>
#include <memory>
#include <mutex>
#include <torch/csrc/stable/tensor.h>

namespace onecat_sm70_prefill {
struct ScoreWorkspace {
  const int64_t block_n;
  torch::stable::Tensor scores;
  std::mutex mutex;
  cudaEvent_t completion = nullptr;
  bool completion_recorded = false;
  explicit ScoreWorkspace(const torch::stable::Tensor& query, int64_t block_n);
  ~ScoreWorkspace();
};
std::shared_ptr<ScoreWorkspace> get_score_workspace(const torch::stable::Tensor& query,
                                                     int64_t block_n);
}  // namespace onecat_sm70_prefill
