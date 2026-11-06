// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Defines the typed data_ptr template specializations that the stable ABI
// headers declare but only partially define (tensor_inl.h generates them
// for integral/float/double/complex + uint16/32/64, leaving half-family,
// char and std::byte unresolved at link time).

#include <cstddef>
#include <type_traits>

#include <cuda_fp16.h>

#include <cutlass/bfloat16.h>
#include <cutlass/half.h>

#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

namespace torch {
namespace stable {

#define VLLM_STABLE_DATA_PTR_SHIM_CAST(T, NAME)                                \
  template <>                                                                    \
  T* Tensor::mutable_data_ptr<T>() const {                                       \
    auto stype = scalar_type();                                                 \
    STD_TORCH_CHECK(                                                            \
        stype == torch::headeronly::ScalarType::NAME,                           \
        "expected scalar type " #NAME " but found ",                              \
        torch::headeronly::toString(stype));                                     \
    return static_cast<T*>(mutable_data_ptr());                                  \
  }                                                                               \
  template <>                                                                     \
  const T* Tensor::const_data_ptr<T, 0>() const {                                 \
    auto stype = scalar_type();                                                  \
    STD_TORCH_CHECK(                                                             \
        stype == torch::headeronly::ScalarType::NAME,                            \
        "expected scalar type " #NAME " but found ",                               \
        torch::headeronly::toString(stype));                                      \
    return static_cast<const T*>(const_data_ptr());                               \
  }

VLLM_STABLE_DATA_PTR_SHIM_CAST(__half, Half)
VLLM_STABLE_DATA_PTR_SHIM_CAST(cutlass::half_t, Half)
VLLM_STABLE_DATA_PTR_SHIM_CAST(cutlass::bfloat16_t, BFloat16)
VLLM_STABLE_DATA_PTR_SHIM_CAST(char, Char)
VLLM_STABLE_DATA_PTR_SHIM_CAST(std::byte, Byte)

#undef VLLM_STABLE_DATA_PTR_SHIM_CAST

}  // namespace stable
}  // namespace torch
