// SPDX-License-Identifier: Apache-2.0
// Exact sparse Philox sampling for the SM70 Qwen3.8 batch-one top-20 route.

#include <curand_kernel.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <mutex>
#include <optional>
#include <libtorch_stable/torch_utils.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/HeaderOnlyArrayRef.h>

#ifndef C10_CUDA_KERNEL_LAUNCH_CHECK
#define C10_CUDA_KERNEL_LAUNCH_CHECK()                                     \
  do {                                                                        \
    cudaError_t _cke_err = cudaGetLastError();                               \
    if (_cke_err != cudaSuccess) {                                            \
      fprintf(stderr,                                                        \
              "[sm70_top20_philox] CUDA kernel launch error: %s\n",         \
              cudaGetErrorString(_cke_err));                                  \
      std::abort();                                                          \
    }                                                                         \
  } while (0)
#endif

namespace {

constexpr int kTopK = 20;
constexpr int kWarpSize = 32;
constexpr int kNativeBlockSize = 256;
constexpr int kNativeUnroll = 4;
constexpr int kMaxGeneratorOffsetsPerCurandCall = 4;

using PhiloxCudaState = std::uint64_t;

struct Sm70PhiloxGenCtx {
  std::mutex mu;
  uint64_t philox_state = 0;
};

Sm70PhiloxGenCtx& sm70_default_philox_gen() {
  static Sm70PhiloxGenCtx ctx;
  return ctx;
}

__device__ __forceinline__ void phi_unpack(PhiloxCudaState s, uint32_t& seed,
                                            uint32_t& offset) {
  seed = static_cast<uint32_t>(s >> 32);
  offset = static_cast<uint32_t>(s & 0xFFFFFFFFu);
}

__device__ __forceinline__ bool top1_better(float value, int token,
                                             float best_value, int best_token) {
  return value > best_value || (value == best_value && token < best_token);
}

__device__ __forceinline__ float native_exponential_for_index(
    int64_t index, int64_t total_threads, PhiloxCudaState philox_args) {
  uint32_t seed, offset;
  phi_unpack(philox_args, seed, offset);
  const int64_t values_per_round = total_threads * kNativeUnroll;
  const int64_t round = index / values_per_round;
  const int64_t within_round = index - round * values_per_round;
  const uint64_t subsequence =
      static_cast<uint64_t>(within_round % total_threads);
  const int component = static_cast<int>(within_round / total_threads);

  curandStatePhilox4_32_10_t state;
  curand_init(seed, subsequence, offset, &state);
  float4 random = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int64_t iteration = 0; iteration <= round; ++iteration) {
    random = curand_uniform4(&state);
  }
  const float uniform = reinterpret_cast<const float*>(&random)[component];
  return -std::log1p(-uniform);
}

template <bool EmitMetadata, bool ChunkedIndices>
__global__ void sm70_sample_sorted_top20_philox_kernel(
    int64_t* sampled_token_out, int64_t* sparse_ids_out,
    float* sparse_probs_out, const float* top_values,
    const int64_t* top_indices, const int64_t* local_indices,
    const int64_t* global_positions, int64_t local_candidate_count,
    int chunk_size, int64_t vocab_size, int64_t total_threads, float top_p,
    PhiloxCudaState philox_args) {
  const int lane = threadIdx.x;
  __shared__ float selected_values[kTopK];
  __shared__ int64_t selected_ids[kTopK];
  __shared__ float selected_random[kTopK];

  int64_t lane_token64 = -1;
  if (lane < kTopK) {
    if (ChunkedIndices) {
      const int64_t position = global_positions[lane];
      if (position >= 0 && position < local_candidate_count) {
        const int64_t local_index = local_indices[position];
        if (local_index >= 0 && local_index < chunk_size) {
          lane_token64 = (position / kTopK) * chunk_size + local_index;
        }
      }
    } else {
      lane_token64 = top_indices[lane];
    }
  }
  const bool lane_valid =
      lane < kTopK && lane_token64 >= 0 && lane_token64 < vocab_size;
  const int lane_token = lane_valid ? static_cast<int>(lane_token64) : -1;
  const float lane_raw_value = lane < kTopK ? top_values[lane] : -FLT_MAX;
  const float lane_value = isfinite(lane_raw_value) ? lane_raw_value : -FLT_MAX;
  const int valid_count = __popc(__ballot_sync(0xffffffff, lane_valid));
  int lane_rank = 0;
#pragma unroll
  for (int source = 0; source < kTopK; ++source) {
    const int other_token = __shfl_sync(0xffffffff, lane_token, source);
    const float other_value = __shfl_sync(0xffffffff, lane_value, source);
    if (lane_valid && other_token >= 0 && other_token < vocab_size &&
        top1_better(other_value, other_token, lane_value, lane_token)) {
      ++lane_rank;
    }
  }
  if (lane_valid) {
    selected_values[lane_rank] = lane_value;
    selected_ids[lane_rank] = static_cast<int64_t>(lane_token);
  } else if (lane >= valid_count && lane < kTopK) {
    // Preserve the prior fallback for malformed candidate ids.
    selected_values[lane] = -FLT_MAX;
    selected_ids[lane] = lane;
  }
  __syncwarp();

  if (lane < kTopK) {
    selected_random[lane] = native_exponential_for_index(
        selected_ids[lane], total_threads, philox_args);
  }
  __syncwarp();

  if (lane == 0) {
    const float max_value = selected_values[0];
    float probability_sum = 0.f;
    for (int index = 0; index < kTopK; ++index) {
      const float probability = __expf(selected_values[index] - max_value);
      selected_values[index] = probability;
      probability_sum += probability;
    }

    float cumulative_probability = 0.f;
    float filtered_probability_sum = 0.f;
    for (int index = 0; index < kTopK; ++index) {
      const float probability = selected_values[index] / probability_sum;
      const bool keep = index == 0 || cumulative_probability < top_p;
      cumulative_probability += probability;
      selected_values[index] = keep ? probability : 0.f;
      filtered_probability_sum += selected_values[index];
    }

    float best_score = -FLT_MAX;
    int64_t sampled_token = selected_ids[0];
    for (int index = 0; index < kTopK; ++index) {
      const float probability =
          selected_values[index] / filtered_probability_sum;
      if (EmitMetadata) {
        sparse_probs_out[index] = probability;
        sparse_ids_out[index] = selected_ids[index];
      }
      const float score = probability / selected_random[index];
      if (score > best_score) {
        best_score = score;
        sampled_token = selected_ids[index];
      }
    }
    sampled_token_out[0] = sampled_token;
  }
}

template <bool EmitMetadata, bool ChunkedIndices>
void launch_sm70_sample_sorted_top20_philox(
    torch::stable::Tensor sampled_token_out, int64_t* sparse_ids_out,
    float* sparse_probs_out, torch::stable::Tensor top_values,
    const int64_t* top_indices, const int64_t* local_indices,
    const int64_t* global_positions, int64_t local_candidate_count,
    int chunk_size, const std::optional<int64_t>& generator,
    int64_t vocab_size, double top_p) {
  cudaDeviceProp properties_local;
  int properties_dev = 0;
  cudaGetDevice(&properties_dev);
  cudaGetDeviceProperties(&properties_local, properties_dev);
  const auto* properties = &properties_local;
  const int64_t blocks_per_sm =
      properties->maxThreadsPerMultiProcessor / kNativeBlockSize;
  const int64_t required_blocks =
      (vocab_size + kNativeBlockSize - 1) / kNativeBlockSize;
  const int64_t grid_blocks = std::min<int64_t>(
      required_blocks, properties->multiProcessorCount * blocks_per_sm);
  const int64_t total_threads = grid_blocks * kNativeBlockSize;
  const uint64_t counter_offset =
      ((vocab_size - 1) / (total_threads * kNativeUnroll) + 1) *
      kMaxGeneratorOffsetsPerCurandCall;

  Sm70PhiloxGenCtx* gen_ctx = &sm70_default_philox_gen();
  if (generator.has_value() && generator.value() != 0) {
    gen_ctx = reinterpret_cast<Sm70PhiloxGenCtx*>(generator.value());
  }
  PhiloxCudaState philox_args;
  {
    std::lock_guard<std::mutex> lock(gen_ctx->mu);
    gen_ctx->philox_state += counter_offset;
    philox_args = gen_ctx->philox_state;
  }

  sm70_sample_sorted_top20_philox_kernel<EmitMetadata, ChunkedIndices>
      <<<1, kWarpSize, 0, get_current_cuda_stream()>>>(
          sampled_token_out.mutable_data_ptr<int64_t>(), sparse_ids_out,
          sparse_probs_out, top_values.const_data_ptr<float>(), top_indices,
          local_indices, global_positions, local_candidate_count, chunk_size,
          vocab_size, total_threads, static_cast<float>(top_p), philox_args);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void sm70_sample_sorted_top20_philox_out(
    torch::stable::Tensor sampled_token_out, torch::stable::Tensor sparse_ids_out,
    torch::stable::Tensor sparse_probs_out, torch::stable::Tensor top_values,
    torch::stable::Tensor top_indices, const std::optional<int64_t>& generator,
    int64_t vocab_size, double top_p) {
  STD_TORCH_CHECK(sampled_token_out.is_cuda() && sparse_ids_out.is_cuda() &&
                      sparse_probs_out.is_cuda() && top_values.is_cuda() &&
                      top_indices.is_cuda(),
                  "sm70_sample_sorted_top20_philox_out: tensors must be CUDA");
  STD_TORCH_CHECK(sampled_token_out.scalar_type() == torch::headeronly::ScalarType::Long &&
                      sparse_ids_out.scalar_type() == torch::headeronly::ScalarType::Long &&
                      sparse_probs_out.scalar_type() == torch::headeronly::ScalarType::Float &&
                      top_values.scalar_type() == torch::headeronly::ScalarType::Float &&
                      top_indices.scalar_type() == torch::headeronly::ScalarType::Long,
                  "sm70_sample_sorted_top20_philox_out: dtype mismatch");
  STD_TORCH_CHECK(sampled_token_out.numel() == 1 &&
                      sparse_ids_out.numel() == kTopK &&
                      sparse_probs_out.numel() == kTopK &&
                      top_values.numel() == kTopK && top_indices.numel() == kTopK,
                  "sm70_sample_sorted_top20_philox_out: shape mismatch");
  STD_TORCH_CHECK(
      sampled_token_out.is_contiguous() && sparse_ids_out.is_contiguous() &&
          sparse_probs_out.is_contiguous() && top_values.is_contiguous() &&
          top_indices.is_contiguous(),
      "sm70_sample_sorted_top20_philox_out: tensors must be contiguous");
  STD_TORCH_CHECK(vocab_size >= kTopK && top_p > 0.0 && top_p <= 1.0,
                  "sm70_sample_sorted_top20_philox_out: invalid parameters");

  const torch::stable::accelerator::DeviceGuard device_guard(top_values.get_device_index());
  launch_sm70_sample_sorted_top20_philox<true, false>(
      sampled_token_out, sparse_ids_out.mutable_data_ptr<int64_t>(),
      sparse_probs_out.mutable_data_ptr<float>(), top_values,
      top_indices.const_data_ptr<int64_t>(), nullptr, nullptr, 0, 0, generator,
      vocab_size, top_p);
}

void sm70_sample_sorted_top20_philox_token_out(
    torch::stable::Tensor sampled_token_out, torch::stable::Tensor top_values,
    torch::stable::Tensor top_indices, const std::optional<int64_t>& generator,
    int64_t vocab_size, double top_p) {
  STD_TORCH_CHECK(
      sampled_token_out.is_cuda() && top_values.is_cuda() && top_indices.is_cuda(),
      "sm70_sample_sorted_top20_philox_token_out: tensors must be CUDA");
  STD_TORCH_CHECK(sampled_token_out.scalar_type() == torch::headeronly::ScalarType::Long &&
                      top_values.scalar_type() == torch::headeronly::ScalarType::Float &&
                      top_indices.scalar_type() == torch::headeronly::ScalarType::Long,
                  "sm70_sample_sorted_top20_philox_token_out: dtype mismatch");
  STD_TORCH_CHECK(sampled_token_out.numel() == 1 && top_values.numel() == kTopK &&
                      top_indices.numel() == kTopK,
                  "sm70_sample_sorted_top20_philox_token_out: shape mismatch");
  STD_TORCH_CHECK(
      sampled_token_out.is_contiguous() && top_values.is_contiguous() &&
          top_indices.is_contiguous(),
      "sm70_sample_sorted_top20_philox_token_out: tensors must be contiguous");
  STD_TORCH_CHECK(vocab_size >= kTopK && top_p > 0.0 && top_p <= 1.0,
                  "sm70_sample_sorted_top20_philox_token_out: invalid parameters");

  const torch::stable::accelerator::DeviceGuard device_guard(top_values.get_device_index());
  launch_sm70_sample_sorted_top20_philox<false, false>(
      sampled_token_out, nullptr, nullptr, top_values,
      top_indices.const_data_ptr<int64_t>(), nullptr, nullptr, 0, 0, generator,
      vocab_size, top_p);
}

void sm70_sample_chunked_top20_philox_token_out(
    torch::stable::Tensor sampled_token_out, torch::stable::Tensor global_values,
    torch::stable::Tensor local_indices, torch::stable::Tensor global_positions,
    const std::optional<int64_t>& generator, int64_t vocab_size,
    double top_p, int64_t chunk_size) {
  STD_TORCH_CHECK(
      sampled_token_out.is_cuda() && global_values.is_cuda() &&
          local_indices.is_cuda() && global_positions.is_cuda(),
      "sm70_sample_chunked_top20_philox_token_out: tensors must be CUDA");
  STD_TORCH_CHECK(sampled_token_out.scalar_type() == torch::headeronly::ScalarType::Long &&
                      global_values.scalar_type() == torch::headeronly::ScalarType::Float &&
                      local_indices.scalar_type() == torch::headeronly::ScalarType::Long &&
                      global_positions.scalar_type() == torch::headeronly::ScalarType::Long,
                  "sm70_sample_chunked_top20_philox_token_out: dtype mismatch");
  STD_TORCH_CHECK(sampled_token_out.numel() == 1 &&
                      global_values.numel() == kTopK && local_indices.dim() == 2 &&
                      local_indices.size(1) == kTopK &&
                      global_positions.numel() == kTopK,
                  "sm70_sample_chunked_top20_philox_token_out: shape mismatch");
  STD_TORCH_CHECK(
      sampled_token_out.is_contiguous() && global_values.is_contiguous() &&
          local_indices.is_contiguous() && global_positions.is_contiguous(),
      "sm70_sample_chunked_top20_philox_token_out: tensors must be contiguous");
  STD_TORCH_CHECK(chunk_size > 0 &&
                      local_indices.size(0) * chunk_size == vocab_size &&
                      vocab_size >= kTopK && top_p > 0.0 && top_p <= 1.0,
                  "sm70_sample_chunked_top20_philox_token_out: invalid parameters");

  const torch::stable::accelerator::DeviceGuard device_guard(global_values.get_device_index());
  launch_sm70_sample_sorted_top20_philox<false, true>(
      sampled_token_out, nullptr, nullptr, global_values, nullptr,
      local_indices.const_data_ptr<int64_t>(), global_positions.const_data_ptr<int64_t>(),
      local_indices.numel(), static_cast<int>(chunk_size), generator,
      vocab_size, top_p);
}
