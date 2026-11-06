/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

#include <cuda_fp16.h>

#include "marlin.cuh"
#include "core/registration.h"
#include "core/scalar_type.hpp"
#include <libtorch_stable/torch_utils.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/HeaderOnlyArrayRef.h>
#include <cstdio>
#include <cstdlib>

#ifndef C10_CUDA_KERNEL_LAUNCH_CHECK
#define C10_CUDA_KERNEL_LAUNCH_CHECK()                                     \
  do {                                                                      \
    cudaError_t _cke_err = cudaGetLastError();                               \
    if (_cke_err != cudaSuccess) {                                            \
      fprintf(stderr,                                                        \
              "[sm70-kernel-launch-check] CUDA kernel launch error: %s\n", \
              cudaGetErrorString(_cke_err));                                  \
      std::abort();                                                          \
    }                                                                         \
  } while (0)
#endif

namespace {

__device__ __forceinline__ int sm70_inverse_zp_interleave(int idx,
                                                          int num_bits) {
  if (num_bits == 4) {
    constexpr int inv[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    return inv[idx];
  }
  constexpr int inv[4] = {0, 2, 1, 3};
  return inv[idx];
}

__device__ __forceinline__ int sm70_inverse_scale_perm(int idx, int perm_len) {
  if (perm_len == 64) {
    return (idx % 8) * 8 + idx / 8;
  }
  constexpr int inv[32] = {0,  1,  8,  9,  16, 17, 24, 25, 2,  3,  10,
                           11, 18, 19, 26, 27, 4,  5,  12, 13, 20, 21,
                           28, 29, 6,  7,  14, 15, 22, 23, 30, 31};
  return inv[idx];
}

__global__ void sm70_unpermute_half_metadata_kernel(
    const half* __restrict__ src, half* __restrict__ dst, int64_t total,
    int perm_len) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int64_t block = idx / perm_len;
  const int pos = static_cast<int>(idx % perm_len);
  dst[idx] = src[block * perm_len + sm70_inverse_scale_perm(pos, perm_len)];
}

__global__ void sm70_unpack_dense_zp_to_half_kernel(
    const int32_t* __restrict__ packed_zp, const half* __restrict__ scales,
    half* __restrict__ out, int64_t total, int64_t size_n, int64_t packed_n,
    int num_bits, bool is_a_8bit) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const int pack_factor = 32 / num_bits;
  const int64_t n = idx % size_n;
  const int64_t block64 = n / 64;
  int packed_col_in_block =
      sm70_inverse_scale_perm(static_cast<int>(n % 64), 64);
  if (!is_a_8bit) {
    packed_col_in_block =
        (packed_col_in_block / pack_factor) * pack_factor +
        sm70_inverse_zp_interleave(packed_col_in_block % pack_factor, num_bits);
  }
  const int64_t packed_col = block64 * 64 + packed_col_in_block;
  const uint32_t word = reinterpret_cast<const uint32_t*>(
      packed_zp)[(idx / size_n) * packed_n + packed_col / pack_factor];
  const uint32_t zp = (word >> ((packed_col % pack_factor) * num_bits)) &
                      ((1u << num_bits) - 1u);
  out[idx] = __float2half(static_cast<float>(zp) * __half2float(scales[idx]));
}

torch::stable::Tensor sm70_dense_scales_as_logical(torch::stable::Tensor const& b_scales,
                                           int64_t size_k, int64_t group_size,
                                           bool is_a_8bit) {
  STD_TORCH_CHECK(b_scales.scalar_type() == torch::headeronly::ScalarType::Half,
              "SM70 Marlin adapter expects fp16 scales.");
  const bool uses_group_perm =
      group_size < size_k && group_size != -1 && !is_a_8bit;
  const int perm_len = uses_group_perm ? 64 : 32;
  STD_TORCH_CHECK(b_scales.numel() % perm_len == 0,
              "SM70 Marlin scale metadata size is not divisible by ", perm_len);

  auto out = torch::stable::empty_like(b_scales);
  const int64_t total = out.numel();
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  sm70_unpermute_half_metadata_kernel<<<blocks, threads, 0,
                                        get_current_cuda_stream()>>>(
      reinterpret_cast<const half*>(b_scales.const_data_ptr<half>()),
      out.mutable_data_ptr<half>(), total, perm_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::stable::Tensor sm70_dense_zp_as_half(torch::stable::Tensor const& b_zeros,
                                    torch::stable::Tensor const& b_scales,
                                    int64_t size_n, int num_bits,
                                    bool is_a_8bit) {
  if (b_zeros.scalar_type() == torch::headeronly::ScalarType::Half) {
    return b_zeros;
  }
  STD_TORCH_CHECK(b_zeros.scalar_type() == torch::headeronly::ScalarType::Int,
              "SM70 Marlin dense zero points must be fp16 or packed int32.");
  const int pack_factor = 32 / num_bits;
  STD_TORCH_CHECK(b_zeros.dim() == 2, "b_zeros rank = ", b_zeros.dim(),
              " is not 2");
  STD_TORCH_CHECK(b_zeros.size(0) == b_scales.size(0),
              "b_zeros dim 0 = ", b_zeros.size(0),
              " is not num_groups = ", b_scales.size(0));
  STD_TORCH_CHECK(b_zeros.size(1) * pack_factor == size_n,
              "packed b_zeros dim 1 = ", b_zeros.size(1),
              " does not match size_n = ", size_n);
  STD_TORCH_CHECK(b_scales.scalar_type() == torch::headeronly::ScalarType::Half,
              "SM70 Marlin packed zero-point adapter expects fp16 scales.");

  auto out = torch::stable::empty(
      {b_scales.size(0), size_n}, b_scales.scalar_type(), std::nullopt,
      b_scales.device());
  const int64_t total = out.numel();
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  sm70_unpack_dense_zp_to_half_kernel<<<blocks, threads, 0,
                                        get_current_cuda_stream()>>>(
      b_zeros.const_data_ptr<int32_t>(),
      reinterpret_cast<const half*>(b_scales.const_data_ptr<half>()),
      out.mutable_data_ptr<half>(), total, size_n,
      b_zeros.size(1), num_bits, is_a_8bit);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace

torch::stable::Tensor sm70_marlin_u4b8_gemm(torch::stable::Tensor& a, torch::stable::Tensor& c,
                                    torch::stable::Tensor& b_q_weight,
                                    torch::stable::Tensor& b_scales, int64_t size_m,
                                    int64_t size_n, int64_t size_k,
                                    int64_t group_size);
torch::stable::Tensor sm70_marlin_u4_gemm(torch::stable::Tensor& a, torch::stable::Tensor& c,
                                  torch::stable::Tensor& b_q_weight,
                                  torch::stable::Tensor& b_scales,
                                  torch::stable::Tensor& b_zeros, int64_t size_m,
                                  int64_t size_n, int64_t size_k,
                                  int64_t group_size);
torch::stable::Tensor sm70_marlin_u8_gemm(torch::stable::Tensor& a, torch::stable::Tensor& c,
                                  torch::stable::Tensor& b_q_weight,
                                  torch::stable::Tensor& b_scales,
                                  torch::stable::Tensor& b_zeros, int64_t size_m,
                                  int64_t size_n, int64_t size_k,
                                  int64_t group_size);
torch::stable::Tensor sm70_marlin_u8b128_gemm(torch::stable::Tensor& a, torch::stable::Tensor& c,
                                      torch::stable::Tensor& b_q_weight,
                                      torch::stable::Tensor& b_scales, int64_t size_m,
                                      int64_t size_n, int64_t size_k,
                                      int64_t group_size);
torch::stable::Tensor sm70_marlin_fp8_gemm(torch::stable::Tensor& a, torch::stable::Tensor& c,
                                   torch::stable::Tensor& b_q_weight,
                                   torch::stable::Tensor& b_scales, int64_t size_m,
                                   int64_t size_n, int64_t size_k,
                                   int64_t group_size);
torch::stable::Tensor sm70_marlin_nvfp4_gemm(torch::stable::Tensor& a, torch::stable::Tensor& c,
                                     torch::stable::Tensor& b_q_weight,
                                     torch::stable::Tensor& b_scales,
                                     torch::stable::Tensor& global_scale,
                                     int64_t size_m, int64_t size_n,
                                     int64_t size_k, int64_t group_size);
torch::stable::Tensor sm70_marlin_mxfp4_gemm(torch::stable::Tensor& a, torch::stable::Tensor& c,
                                     torch::stable::Tensor& b_q_weight,
                                     torch::stable::Tensor& b_scales, int64_t size_m,
                                     int64_t size_n, int64_t size_k,
                                     int64_t group_size);

torch::stable::Tensor marlin_gemm(
    torch::stable::Tensor& a, std::optional<torch::stable::Tensor> c_or_none,
    torch::stable::Tensor& b_q_weight,
    std::optional<torch::stable::Tensor> const& b_bias_or_none, torch::stable::Tensor& b_scales,
    std::optional<torch::stable::Tensor> const& a_scales_or_none,
    std::optional<torch::stable::Tensor> const& global_scale_or_none,
    std::optional<torch::stable::Tensor> const& b_zeros_or_none,
    std::optional<torch::stable::Tensor> const& g_idx_or_none,
    std::optional<torch::stable::Tensor> const& perm_or_none, torch::stable::Tensor& workspace,
    vllm::ScalarTypeId const& b_type_id, int64_t size_m, int64_t size_n,
    int64_t size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
    bool is_zp_float) {
  (void)workspace;
  vllm::ScalarTypeId a_type_id, c_type_id, s_type_id;

  torch::headeronly::ScalarType c_dtype = a.scalar_type();
  if (a.scalar_type() == torch::headeronly::ScalarType::Half) {
    a_type_id = vllm::kFloat16.id();
    c_type_id = vllm::kFloat16.id();
  } else if (a.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    a_type_id = vllm::kBFloat16.id();
    c_type_id = vllm::kBFloat16.id();
  } else {
    c_dtype = b_scales.scalar_type();
    if (b_scales.scalar_type() == torch::headeronly::ScalarType::Half) {
      c_type_id = vllm::kFloat16.id();
    } else if (b_scales.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
      c_type_id = vllm::kBFloat16.id();
    } else {
      c_type_id = vllm::kBFloat16.id();

      STD_TORCH_CHECK(c_or_none.has_value(), "c must be passed for W4A8-FP4");
      torch::stable::Tensor c = c_or_none.value();
      c_dtype = c.scalar_type();

      if (c.scalar_type() == torch::headeronly::ScalarType::Half) {
        c_type_id = vllm::kFloat16.id();
      } else if (c.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
        c_type_id = vllm::kBFloat16.id();
      } else {
        STD_TORCH_CHECK(false, "unsupported c dtype");
      }
    }

    if (a.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn) {
      a_type_id = vllm::kFE4M3fn.id();
    } else if (a.scalar_type() == torch::headeronly::ScalarType::Char) {
      a_type_id = vllm::kS8.id();
    } else {
      STD_TORCH_CHECK(false, "unsupported `a` scalar_type");
    }
  }

  s_type_id = c_type_id;
  if (b_type_id == vllm::kFE2M1f.id()) {
    if (b_scales.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn) {
      s_type_id = vllm::kFE4M3fn.id();
    } else if (b_scales.scalar_type() == torch::headeronly::ScalarType::Float8_e8m0fnu) {
      s_type_id = vllm::kFE8M0fnu.id();
    } else {
      STD_TORCH_CHECK(false,
                  "When b_type = float4_e2m1f on the SM70 build, b_scales "
                  "must be float8_e4m3fn for NVFP4 or float8_e8m0fnu for "
                  "MXFP4 scales.");
    }
  }

  vllm::ScalarType a_type = vllm::ScalarType::from_id(a_type_id);
  vllm::ScalarType b_type = vllm::ScalarType::from_id(b_type_id);
  vllm::ScalarType c_type = vllm::ScalarType::from_id(c_type_id);
  vllm::ScalarType s_type = vllm::ScalarType::from_id(s_type_id);

  STD_TORCH_CHECK(a_type == vllm::kFloat16,
              "SM70 build only supports float16 activations.");
  STD_TORCH_CHECK(c_type == vllm::kFloat16,
              "SM70 build only supports float16 outputs.");
  STD_TORCH_CHECK(s_type == vllm::kFloat16 ||
                  (b_type == vllm::kFE2M1f &&
                   (s_type == vllm::kFE4M3fn || s_type == vllm::kFE8M0fnu)),
              "SM70 build only supports float16 scales, except FP4 uses "
              "float8_e4m3fn scales for NVFP4 or float8_e8m0fnu scales for "
              "MXFP4.");
  STD_TORCH_CHECK(b_type == vllm::kU4 || b_type == vllm::kU4B8 ||
                  b_type == vllm::kU8 || b_type == vllm::kU8B128 ||
                  b_type == vllm::kFE4M3fn || b_type == vllm::kFE2M1f,
              "SM70 Marlin currently implements only uint4, "
              "uint4b8, uint8, uint8b128, fp8_e4m3fn, nvfp4, and "
              "mxfp4 dense weights.");
  (void)use_fp32_reduce;

  int pack_factor = 32 / b_type.size_bits();

  // Verify A
  STD_TORCH_CHECK(a.size(0) == size_m, "Shape mismatch: a.size(0) = ", a.size(0),
              ", size_m = ", size_m);
  STD_TORCH_CHECK(a.size(1) == size_k, "Shape mismatch: a.size(1) = ", a.size(1),
              ", size_k = ", size_k);

  // Verify B
  STD_TORCH_CHECK(
      size_k % MARLIN_NAMESPACE_NAME::tile_size == 0, "size_k = ", size_k,
      " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  STD_TORCH_CHECK((size_k / MARLIN_NAMESPACE_NAME::tile_size) == b_q_weight.size(0),
              "Shape mismatch: b_q_weight.size(0) = ", b_q_weight.size(0),
              ", size_k = ", size_k,
              ", tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  STD_TORCH_CHECK(
      b_q_weight.size(1) % MARLIN_NAMESPACE_NAME::tile_size == 0,
      "b_q_weight.size(1) = ", b_q_weight.size(1),
      " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  int actual_size_n =
      (b_q_weight.size(1) / MARLIN_NAMESPACE_NAME::tile_size) * pack_factor;
  STD_TORCH_CHECK(size_n == actual_size_n, "size_n = ", size_n,
              ", actual_size_n = ", actual_size_n);

  // Verify device and strides
  STD_TORCH_CHECK(a.device().is_cuda(), "A is not on GPU");
  STD_TORCH_CHECK(a.stride(1) == 1, "A.stride(1) is not 1");
  // We use int4 (16 bytes) to load A, so A must aligned to 16 bytes
  STD_TORCH_CHECK(a.stride(0) % 8 == 0, "A.stride(0) must divisible by 8");
  STD_TORCH_CHECK(((uint64_t)a.data_ptr()) % 16 == 0, "A must aligned to 16 bytes");

  STD_TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
  STD_TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");

  STD_TORCH_CHECK(b_scales.device().is_cuda(), "b_scales is not on GPU");
  STD_TORCH_CHECK(b_scales.is_contiguous(), "b_scales is not contiguous");
  STD_TORCH_CHECK(b_scales.scalar_type() == torch::headeronly::ScalarType::Half ||
                  (b_type == vllm::kFE2M1f &&
                   (b_scales.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn ||
                    b_scales.scalar_type() == torch::headeronly::ScalarType::Float8_e8m0fnu)),
              "SM70 build only supports float16 scales, except FP4 uses "
              "float8_e4m3fn scales for NVFP4 or float8_e8m0fnu scales for "
              "MXFP4.");

  torch::stable::Tensor a_scales;

  if (a_scales_or_none.has_value()) {
    a_scales = a_scales_or_none.value();
    STD_TORCH_CHECK(a_type.size_bits() == 8,
                "a_scales can only be used for 8bit activation.");
  } else {
    a_scales = torch::stable::empty({0}, torch::headeronly::ScalarType::Float, std::nullopt, a.device());
    STD_TORCH_CHECK(a_type.size_bits() != 8,
                "the a_scales parameter must be passed for 8bit activation.");
  }

  // Alloc buffers
  const torch::stable::accelerator::DeviceGuard device_guard(a.get_device_index());
  torch::stable::Tensor c;
  if (c_or_none.has_value()) {
    c = c_or_none.value();
    STD_TORCH_CHECK(c.device().is_cuda(), "c is not on GPU");
    STD_TORCH_CHECK(c.is_contiguous(), "c is not contiguous");
    STD_TORCH_CHECK(c.scalar_type() == torch::headeronly::ScalarType::Half,
                "SM70 build only supports float16 outputs.");
    STD_TORCH_CHECK(c.size(0) == size_m, "Shape mismatch: c.size(0) = ", c.size(0),
                ", size_m = ", size_m);
    STD_TORCH_CHECK(c.size(1) == size_n, "Shape mismatch: c.size(1) = ", c.size(1),
                ", size_n = ", size_n);
  } else {
    c = torch::stable::empty({size_m, size_n}, c_dtype, std::nullopt, a.device());
  }
  if (size_m == 0) return c;

  // Detect groupsize and act_order
  int num_groups = -1;
  int group_size = -1;

  int rank = b_scales.sizes().size();
  STD_TORCH_CHECK(rank == 2, "b_scales rank = ", rank, " is not 2");
  STD_TORCH_CHECK(b_scales.size(1) == size_n, "b_scales dim 1 = ", b_scales.size(1),
              " is not size_n = ", size_n);
  num_groups = b_scales.size(0);

  torch::stable::Tensor g_idx, perm;
  if (g_idx_or_none.has_value() && perm_or_none.has_value()) {
    g_idx = g_idx_or_none.value();
    perm = perm_or_none.value();

    STD_TORCH_CHECK(g_idx.device().is_cuda(), "g_idx is not on GPU");
    STD_TORCH_CHECK(g_idx.is_contiguous(), "g_idx is not contiguous");
    STD_TORCH_CHECK(perm.device().is_cuda(), "perm is not on GPU");
    STD_TORCH_CHECK(perm.is_contiguous(), "perm is not contiguous");

    // Verify g_idx and perm
    STD_TORCH_CHECK((g_idx.size(-1) == 0 && perm.size(-1) == 0) ||
                    (g_idx.size(-1) == size_k && perm.size(-1) == size_k),
                "Unexpected g_idx.size(-1) = ", g_idx.size(-1),
                " and perm.size(-1) = ", perm.size(-1),
                ", where size_k = ", size_k);
  } else {
    g_idx = torch::stable::empty({0}, c_dtype, std::nullopt, a.device());
    perm = torch::stable::empty({0}, c_dtype, std::nullopt, a.device());
  }
  bool has_act_order = g_idx.size(-1) > 0 && perm.size(-1) > 0;

  if (has_act_order) {
    if (is_k_full) {
      STD_TORCH_CHECK(num_groups > 1, "For act_order, num_groups must be > 1");
      STD_TORCH_CHECK(size_k % num_groups == 0, "size_k = ", size_k,
                  ", is not divisible by num_groups = ", num_groups);
      group_size = size_k / num_groups;
    } else {
      group_size = 0;
    }

  } else {
    if (num_groups > 1) {
      STD_TORCH_CHECK(
          size_k % num_groups == 0, "size_k = ", size_k,
          ", is not divisible by b_scales.size(0) = ", b_scales.size(0));
      group_size = size_k / num_groups;
    } else {
      group_size = -1;
    }
  }

  torch::stable::Tensor b_scales_kernel = b_scales;
  if (b_scales.scalar_type() == torch::headeronly::ScalarType::Half) {
    b_scales_kernel = sm70_dense_scales_as_logical(b_scales, size_k, group_size,
                                                   a_type.size_bits() == 8);
  }

  torch::stable::Tensor global_scale;
  if (global_scale_or_none.has_value()) {
    global_scale = global_scale_or_none.value();
    STD_TORCH_CHECK(b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn,
                "SM70 Marlin supports global_scale only for "
                "nvfp4 format.");
    STD_TORCH_CHECK(global_scale.device().is_cuda(), "global_scale is not on GPU");
    STD_TORCH_CHECK(global_scale.is_contiguous(), "global_scale is not contiguous");
    STD_TORCH_CHECK(global_scale.scalar_type() == torch::headeronly::ScalarType::Float,
                "SM70 Marlin NVFP4 expects fp32 global_scale.");
    STD_TORCH_CHECK(global_scale.numel() == 1,
                "SM70 Marlin NVFP4 expects a single global_scale "
                "value.");
  } else {
    global_scale = torch::stable::empty({0}, torch::headeronly::ScalarType::Float, std::nullopt, a.device());
    STD_TORCH_CHECK(!(b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn),
                "the global_scale parameter must be passed for nvfp4 format.");
  }

  bool has_bias = b_bias_or_none.has_value();
  torch::stable::Tensor b_bias;
  if (has_bias) {
    b_bias = b_bias_or_none.value();
    STD_TORCH_CHECK(b_bias.device().is_cuda(), "b_bias is not on GPU");
    STD_TORCH_CHECK(b_bias.is_contiguous(), "b_bias is not contiguous");
    STD_TORCH_CHECK(b_bias.size(0) == size_n, "b_bias.size(0) != size_n");
    STD_TORCH_CHECK(b_bias.stride(0) == 1, "b_bias.stride(0) != 1");
  } else {
    b_bias = torch::stable::empty({0}, c_dtype, std::nullopt, a.device());
  }

  torch::stable::Tensor b_zeros;
  if (b_zeros_or_none.has_value()) {
    b_zeros = b_zeros_or_none.value();
    STD_TORCH_CHECK(b_zeros.device().is_cuda(), "b_zeros is not on GPU");
    STD_TORCH_CHECK(b_zeros.is_contiguous(), "b_zeros is not contiguous");
  } else {
    b_zeros = torch::stable::empty({0}, c_dtype, std::nullopt, a.device());
  }
  bool has_zp = b_zeros.size(-1) > 0;
  bool zp_is_float = is_zp_float;

  STD_TORCH_CHECK(!has_zp || b_type == vllm::kU4 || b_type == vllm::kU8,
              "SM70 Marlin does not support zero-point "
              "metadata for this quant type.");

  if (has_zp) {
    if (!is_zp_float && b_zeros.scalar_type() == torch::headeronly::ScalarType::Int) {
      b_zeros = sm70_dense_zp_as_half(b_zeros, b_scales_kernel, size_n,
                                      b_type == vllm::kU4 ? 4 : 8,
                                      a_type.size_bits() == 8);
      zp_is_float = true;
    }
  }

  // Verify fp16 zero-point metadata for dense uint4/uint8.
  if (has_zp) {
    STD_TORCH_CHECK(zp_is_float,
                "SM70 Marlin received unsupported zero-point metadata.");
    int rank = b_zeros.sizes().size();
    STD_TORCH_CHECK(rank == 2, "b_zeros rank = ", rank, " is not 2");
    STD_TORCH_CHECK(b_zeros.scalar_type() == torch::headeronly::ScalarType::Half,
                "SM70 Marlin uint4/uint8 dense path expects fp16 "
                "zero points.");
    STD_TORCH_CHECK(b_zeros.size(1) == size_n, "b_zeros dim 1 = ", b_zeros.size(1),
                " is not size_n = ", size_n);
    STD_TORCH_CHECK(num_groups == b_zeros.size(0),
                "b_zeros dim 0 = ", b_zeros.size(0),
                " is not num_groups = ", num_groups);
  } else {
    STD_TORCH_CHECK(!zp_is_float,
                "is_zp_float is true but b_zeros was not provided.");
  }

  STD_TORCH_CHECK(size_n % MARLIN_NAMESPACE_NAME::min_thread_n == 0,
              "SM70 Marlin requires size_n % 64 == 0. "
              "size_n = ",
              size_n, ", min_thread_n = ", MARLIN_NAMESPACE_NAME::min_thread_n);

  STD_TORCH_CHECK(a_scales.scalar_type() == torch::headeronly::ScalarType::Float,
              "scalar type of a_scales must be float");
  STD_TORCH_CHECK(global_scale.scalar_type() == torch::headeronly::ScalarType::Float,
              "scalar type of global_scale must be float");
  if (a_type.size_bits() == 16) {
    STD_TORCH_CHECK(
        a.scalar_type() == c.scalar_type(),
        "scalar type of a must be the same with c for 16 bit activation");
  }

  STD_TORCH_CHECK(b_q_weight.scalar_type() == torch::headeronly::ScalarType::Int,
              "SM70 Marlin expects int32 packed weights.");
  STD_TORCH_CHECK(!has_act_order,
              "act_order is not supported for the SM70 Marlin path.");
  STD_TORCH_CHECK(
      !has_bias,
      "SM70 Marlin does not support bias. TODO: add epilogue bias fusion.");
  STD_TORCH_CHECK((b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn) ||
                  global_scale.numel() == 0,
              "SM70 Marlin supports global_scale only for "
              "nvfp4 format.");
  STD_TORCH_CHECK(!use_atomic_add,
              "SM70 Marlin does not support atomic-add output.");
  STD_TORCH_CHECK(is_k_full, "SM70 Marlin requires full-K non act-order inputs.");
  STD_TORCH_CHECK(size_k % 32 == 0, "SM70 Marlin requires size_k % 32 == 0.");
  STD_TORCH_CHECK(size_n % 64 == 0, "SM70 Marlin requires size_n % 64 == 0.");
  STD_TORCH_CHECK(group_size == -1 || group_size > 0,
              "SM70 Marlin received invalid group_size = ", group_size);

  if (b_type == vllm::kU4) {
    STD_TORCH_CHECK(size_k % 32 == 0,
                "SM70 Marlin uint4 dense path requires size_k % 32 == 0.");
    STD_TORCH_CHECK(has_zp && zp_is_float,
                "SM70 Marlin uint4 dense path requires fp16 zero points.");
    return sm70_marlin_u4_gemm(a, c, b_q_weight, b_scales_kernel, b_zeros,
                               size_m, size_n, size_k, group_size);
  }

  if (b_type == vllm::kU8) {
    STD_TORCH_CHECK(size_k % 32 == 0,
                "SM70 Marlin uint8 dense path requires size_k % 32 == 0.");
    STD_TORCH_CHECK(has_zp && zp_is_float,
                "SM70 Marlin uint8 dense path requires fp16 zero points.");
    return sm70_marlin_u8_gemm(a, c, b_q_weight, b_scales_kernel, b_zeros,
                               size_m, size_n, size_k, group_size);
  }

  if (b_type == vllm::kU8B128) {
    STD_TORCH_CHECK(size_k % 32 == 0,
                "SM70 Marlin uint8b128 dense path requires size_k % 32 == 0.");
    STD_TORCH_CHECK(!has_zp && !is_zp_float,
                "SM70 Marlin uint8b128 does not support "
                "zero-point metadata.");
    return sm70_marlin_u8b128_gemm(a, c, b_q_weight, b_scales_kernel, size_m,
                                   size_n, size_k, group_size);
  }

  if (b_type == vllm::kFE4M3fn) {
    STD_TORCH_CHECK(size_k % 32 == 0,
                "SM70 Marlin FP8 dense path requires size_k % 32 == 0.");
    STD_TORCH_CHECK(group_size == -1 || group_size == 128,
                "SM70 Marlin FP8 supports only group_size -1 or "
                "128. Got ",
                group_size);
    STD_TORCH_CHECK(!has_zp && !is_zp_float,
                "SM70 Marlin FP8 does not support zero-point "
                "metadata.");
    return sm70_marlin_fp8_gemm(a, c, b_q_weight, b_scales_kernel, size_m,
                                size_n, size_k, group_size);
  }

  if (b_type == vllm::kFE2M1f) {
    STD_TORCH_CHECK(size_k % 32 == 0,
                "SM70 Marlin FP4 dense path requires size_k % 32 == 0.");
    STD_TORCH_CHECK(!has_zp && !is_zp_float,
                "SM70 Marlin FP4 does not support zero-point "
                "metadata.");
    if (s_type == vllm::kFE4M3fn) {
      STD_TORCH_CHECK(
          global_scale.numel() == 1,
          "the global_scale parameter must be passed for nvfp4 format.");
      STD_TORCH_CHECK(group_size == 16,
                  "SM70 Marlin NVFP4 supports only group_size 16. "
                  "Got ",
                  group_size);
      return sm70_marlin_nvfp4_gemm(a, c, b_q_weight, b_scales_kernel,
                                    global_scale, size_m, size_n, size_k,
                                    group_size);
    }

    STD_TORCH_CHECK(s_type == vllm::kFE8M0fnu,
                "SM70 Marlin MXFP4 expects float8_e8m0fnu "
                "MXFP4 scales.");
    STD_TORCH_CHECK(group_size == 32,
                "SM70 Marlin MXFP4 supports only group_size 32 "
                "when global_scale is not provided. Got ",
                group_size);
    return sm70_marlin_mxfp4_gemm(a, c, b_q_weight, b_scales_kernel, size_m,
                                  size_n, size_k, group_size);
  }

  STD_TORCH_CHECK(!has_zp && !is_zp_float,
              "SM70 Marlin uint4b8 does not support zero-point "
              "metadata.");
  return sm70_marlin_u4b8_gemm(a, c, b_q_weight, b_scales_kernel, size_m,
                               size_n, size_k, group_size);
}

STABLE_TORCH_LIBRARY_IMPL(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("marlin_gemm", TORCH_BOX(&marlin_gemm));
}
