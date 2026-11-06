// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <cuda_runtime.h>
#include <cstdlib>
#include <map>
#include <mutex>
#include <memory>
#include <optional>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include "shared_workspace.h"
#include "../sm70_stable_compat.h"

namespace onecat_sm70_prefill {
ScoreWorkspace::ScoreWorkspace(const torch::stable::Tensor& query, int64_t block_n)
    : block_n(block_n),
      scores(torch::stable::empty(
          {block_n * 8192 * 6}, query.scalar_type(), std::nullopt,
          query.device())) {
  TORCH_CHECK(cudaEventCreateWithFlags(&completion, cudaEventDisableTiming) ==
                   cudaSuccess,
              "failed to create score workspace completion event");
}
ScoreWorkspace::~ScoreWorkspace() {
  if (completion != nullptr) cudaEventDestroy(completion);
}
std::shared_ptr<ScoreWorkspace> get_score_workspace(const torch::stable::Tensor& query,
                                                     int64_t block_n) {
  if (const char* value =
          std::getenv("VLLM_FLASH_V100_PREFILL_SCORE_BLOCK_TOKENS")) {
    char* end = nullptr;
    block_n = std::strtol(value, &end, 10);
    TORCH_CHECK(end != value && *end == '\0' && block_n >= 8192 &&
                    block_n <= 16 * 8192 && block_n % 8192 == 0,
                "VLLM_FLASH_V100_PREFILL_SCORE_BLOCK_TOKENS must be a "
                "multiple of 8192 between 8192 and 131072");
  }
  static std::mutex mutex;
  static std::map<std::pair<int, int64_t>, std::shared_ptr<ScoreWorkspace>>
      cache;
  std::lock_guard<std::mutex> lock(mutex);
  auto& entry = cache[{query.get_device(), block_n}];
  if (!entry) entry = std::make_shared<ScoreWorkspace>(query, block_n);
  return entry;
}
}  // namespace onecat_sm70_prefill

extern "C" int64_t onecat_sm70_q8000_accumulation_bits();
extern "C" int64_t onecat_sm70_q8192_accumulation_bits();

int64_t sm70_d256_gqa_accumulation_bits() {
  return onecat_sm70_q8000_accumulation_bits() == 32 &&
                 onecat_sm70_q8192_accumulation_bits() == 32
         ? 32
         : 16;
}

namespace onecat_79t_q8192 {
torch::stable::Tensor sm70_d256_gqa_architecture_q8192_fwd(
    const torch::stable::Tensor& q, const torch::stable::Tensor& k,
    const torch::stable::Tensor& v, torch::stable::Tensor& out, double softmax_scale,
    bool causal);
}

STABLE_TORCH_LIBRARY_FRAGMENT(_vllm_fa2_C, ops) {
  ops.def("sm70_d256_gqa_accumulation_bits() -> int");
  ops.def(
      "sm70_d256_gqa_architecture_q8192_fwd(Tensor q, Tensor k, Tensor v, "
      "Tensor(a!) out, float softmax_scale, bool causal) -> Tensor(a!)");
}

STABLE_TORCH_LIBRARY_IMPL(_vllm_fa2_C, CUDA, ops) {
  ops.impl("sm70_d256_gqa_accumulation_bits", TORCH_BOX(&sm70_d256_gqa_accumulation_bits));
  ops.impl("sm70_d256_gqa_architecture_q8192_fwd",
             TORCH_BOX(&onecat_79t_q8192::sm70_d256_gqa_architecture_q8192_fwd));
}
