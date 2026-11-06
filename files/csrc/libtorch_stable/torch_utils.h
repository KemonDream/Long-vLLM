#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/HeaderOnlyArrayRef.h>
#include <torch/headeronly/util/shim_utils.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <array>
#include <cstddef>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

// Stable ABI equivalent of TORCH_CHECK_NOT_IMPLEMENTED.
#define STD_TORCH_CHECK_NOT_IMPLEMENTED(cond, ...) \
  STD_TORCH_CHECK(cond, "NotImplementedError: ", __VA_ARGS__)

// Device properties cache for stable ABI compatibility.
// Uses raw CUDA/HIP APIs instead of ATen functions.
// Using inline ensures a single instance across all translation units.
inline std::deque<std::once_flag> device_flags;
inline std::vector<cudaDeviceProp> device_properties;
inline std::once_flag vectors_init_flag;

inline void do_init_device_vectors() {
  int device_count;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    STD_TORCH_CHECK(false, "cudaGetDeviceCount failed: " +
                               std::string(cudaGetErrorString(err)));
  }
  device_flags.resize(device_count);
  device_properties.resize(device_count);
}

inline void initDeviceVectors() {
  std::call_once(vectors_init_flag, do_init_device_vectors);
}

inline void initDeviceProperty(int device_index) {
  cudaDeviceProp device_prop{};
  cudaError_t err = cudaGetDeviceProperties(&device_prop, device_index);
  if (err != cudaSuccess) {
    STD_TORCH_CHECK(false, "cudaGetDeviceProperties failed: " +
                               std::string(cudaGetErrorString(err)));
  }
  device_properties[device_index] = device_prop;
}

// Get device properties using raw CUDA/HIP APIs (stable ABI compatible).
// Caches results per device so cudaGetDeviceProperties is called at most once
// per device.
inline cudaDeviceProp* get_device_prop() {
  initDeviceVectors();
  int device_index;
  cudaError_t err = cudaGetDevice(&device_index);
  if (err != cudaSuccess) {
    STD_TORCH_CHECK(
        false, "cudaGetDevice failed: " + std::string(cudaGetErrorString(err)));
  }
  STD_TORCH_CHECK(device_index >= 0 && static_cast<size_t>(device_index) <
                                           device_properties.size(),
                  "CUDA device index " + std::to_string(device_index) +
                      " out of range [0, " +
                      std::to_string(device_properties.size()) + ")");

  std::call_once(device_flags[device_index], initDeviceProperty, device_index);
  return &device_properties[device_index];
}

// Utility to get the current CUDA stream for a given device using stable APIs.
// Returns a cudaStream_t for use in kernel launches.
inline cudaStream_t get_current_cuda_stream(int32_t device_index = -1) {
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device_index, &stream_ptr));
  return reinterpret_cast<cudaStream_t>(stream_ptr);
}

// Utility to get the current cuBLAS handle using stable APIs.
inline cublasHandle_t get_current_cuda_blas_handle() {
  void* blas_handle_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(torch_get_current_cuda_blas_handle(&blas_handle_ptr));
  return reinterpret_cast<cublasHandle_t>(blas_handle_ptr);
}

// Element-wise equality for IntHeaderOnlyArrayRef (no built-in operator==).
inline bool stable_array_ref_eq(const c10::HeaderOnlyArrayRef<int64_t>& a,
                                const c10::HeaderOnlyArrayRef<int64_t>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

// Tensor-shape equality via sizes().
inline bool stable_tensor_sizes_equal(const torch::stable::Tensor& a,
                                       const torch::stable::Tensor& b) {
  return stable_array_ref_eq(a.sizes(), b.sizes());
}

// Shape-literal equality: compare tensor sizes against a static int64 list.
template <size_t N>
inline bool stable_tensor_sizes_match(const torch::stable::Tensor& t,
                                       const std::array<int64_t, N>& shape) {
  auto ts = t.sizes();
  if (ts.size() != N) {
    return false;
  }
  for (size_t i = 0; i < N; ++i) {
    if (ts[i] != shape[i]) {
      return false;
    }
  }
  return true;
}
