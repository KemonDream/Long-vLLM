// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <optional>
#include <libtorch_stable/torch_utils.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/HeaderOnlyArrayRef.h>

void sm70_sample_sorted_top20_philox_out(
    torch::stable::Tensor sampled_token_out, torch::stable::Tensor sparse_ids_out,
    torch::stable::Tensor sparse_probs_out, torch::stable::Tensor top_values,
    torch::stable::Tensor top_indices, const std::optional<int64_t>& generator,
    int64_t vocab_size, double top_p);

void sm70_sample_sorted_top20_philox_token_out(
    torch::stable::Tensor sampled_token_out, torch::stable::Tensor top_values,
    torch::stable::Tensor top_indices, const std::optional<int64_t>& generator,
    int64_t vocab_size, double top_p);

void sm70_sample_chunked_top20_philox_token_out(
    torch::stable::Tensor sampled_token_out, torch::stable::Tensor global_values,
    torch::stable::Tensor local_indices, torch::stable::Tensor global_positions,
    const std::optional<int64_t>& generator, int64_t vocab_size,
    double top_p, int64_t chunk_size);

// Keep these registrations in a wheel-safe sidecar. Adding the sampler must
// not relink or perturb the speed- and quality-frozen primary vllm._C module;
// Python loads this fragment only for the explicitly admitted SM70 route.
STABLE_TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "sm70_sample_sorted_top20_philox_out(Tensor(a!) sampled_token_out, "
      "Tensor(b!) sparse_ids_out, Tensor(c!) sparse_probs_out, "
      "Tensor top_values, Tensor top_indices, int? generator, "
      "int vocab_size, float top_p) -> ()");
  ops.impl("sm70_sample_sorted_top20_philox_out", TORCH_BOX(&sm70_sample_sorted_top20_philox_out));

  ops.def(
      "sm70_sample_sorted_top20_philox_token_out(Tensor(a!) "
      "sampled_token_out, Tensor top_values, Tensor top_indices, "
      "int? generator, int vocab_size, float top_p) -> ()");
  ops.impl("sm70_sample_sorted_top20_philox_token_out", TORCH_BOX(&sm70_sample_sorted_top20_philox_token_out));

  ops.def(
      "sm70_sample_chunked_top20_philox_token_out(Tensor(a!) "
      "sampled_token_out, Tensor global_values, Tensor local_indices, "
      "Tensor global_positions, int? generator, int vocab_size, "
      "float top_p, int chunk_size) -> ()");
  ops.impl("sm70_sample_chunked_top20_philox_token_out", TORCH_BOX(&sm70_sample_chunked_top20_philox_token_out));
}
