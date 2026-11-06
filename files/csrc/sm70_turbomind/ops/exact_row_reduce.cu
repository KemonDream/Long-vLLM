// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <libtorch_stable/torch_utils.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/HeaderOnlyArrayRef.h>

#ifndef C10_CUDA_CHECK
#define C10_CUDA_CHECK(call)                                                       \
  do {                                                                              \
    cudaError_t _cke_err = (call);                                                  \
    if (_cke_err != cudaSuccess) {                                                   \
      fprintf(stderr, "[sm70] CUDA error: %s\n", cudaGetErrorString(_cke_err));   \
      std::abort();                                                                 \
    }                                                                                \
  } while (0)
#endif
#ifndef C10_CUDA_KERNEL_LAUNCH_CHECK
#define C10_CUDA_KERNEL_LAUNCH_CHECK()                                              \
  do {                                                                               \
    cudaError_t _cklc_err = cudaGetLastError();                                      \
    if (_cklc_err != cudaSuccess) {                                                 \
      fprintf(stderr, "[sm70] kernel launch error: %s\n",                           \
              cudaGetErrorString(_cklc_err));                                        \
      std::abort();                                                                  \
    }                                                                                \
  } while (0)
#endif

struct Peers {
  float* data[4];
  unsigned* flags[4];
};
constexpr int GRID = 80;
__device__ __forceinline__ void publish(unsigned* p, unsigned value) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(value)
               : "memory");
}
__device__ __forceinline__ unsigned acquire(unsigned* p) {
  unsigned value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(p)
               : "memory");
  return value;
}
__device__ __forceinline__ void barrier(Peers peers, int rank, unsigned epoch,
                                        int phase) {
  __syncthreads();
  if (threadIdx.x < 4) {
    int peer = threadIdx.x;
    int location = phase * GRID * 4 + blockIdx.x * 4;
    publish(peers.flags[peer] + location + rank, epoch);
    while (acquire(peers.flags[rank] + location + peer) != epoch) {
    }
  }
  __syncthreads();
}
__device__ __forceinline__ float tree(unsigned code, float x0, float x1,
                                      float x2, float x3) {
  switch (code) {
    case 0:
      return __fadd_rn(x0, __fadd_rn(x1, __fadd_rn(x2, x3)));
    case 1:
      return __fadd_rn(x0, __fadd_rn(__fadd_rn(x1, x2), x3));
    case 2:
      return __fadd_rn(x0, __fadd_rn(__fadd_rn(x1, x3), x2));
    case 3:
      return __fadd_rn(__fadd_rn(x0, x1), __fadd_rn(x2, x3));
    case 4:
      return __fadd_rn(__fadd_rn(x0, x2), __fadd_rn(x1, x3));
    case 5:
      return __fadd_rn(__fadd_rn(x0, x3), __fadd_rn(x1, x2));
    case 6:
      return __fadd_rn(__fadd_rn(x0, __fadd_rn(x1, x2)), x3);
    case 7:
      return __fadd_rn(__fadd_rn(__fadd_rn(x0, x1), x2), x3);
    case 8:
      return __fadd_rn(__fadd_rn(__fadd_rn(x0, x2), x1), x3);
    case 9:
      return __fadd_rn(__fadd_rn(x0, __fadd_rn(x1, x3)), x2);
    case 10:
      return __fadd_rn(__fadd_rn(__fadd_rn(x0, x1), x3), x2);
    case 11:
      return __fadd_rn(__fadd_rn(__fadd_rn(x0, x3), x1), x2);
    case 12:
      return __fadd_rn(__fadd_rn(x0, __fadd_rn(x2, x3)), x1);
    case 13:
      return __fadd_rn(__fadd_rn(__fadd_rn(x0, x2), x3), x1);
    case 14:
      return __fadd_rn(__fadd_rn(__fadd_rn(x0, x3), x2), x1);
    default:
      return __int_as_float(0x7fffffff);
  }
}
__global__ void reduce_rows(Peers peers, const uint8_t* codes, float* output,
                            int64_t count, int64_t offset, int rank,
                            unsigned epoch) {
  barrier(peers, rank, epoch, 0);
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += int64_t(gridDim.x) * blockDim.x) {
    int64_t at = offset + i;
    float x0 = __ldcg(peers.data[0] + at);
    float x1 = __ldcg(peers.data[1] + at);
    float x2 = __ldcg(peers.data[2] + at);
    float x3 = __ldcg(peers.data[3] + at);
    output[i] = tree(codes[i], x0, x1, x2, x3);
  }
  barrier(peers, rank, epoch, 1);
}
std::tuple<int64_t, pybind11::bytes> allocate(int64_t bytes) {
  STD_TORCH_CHECK(bytes > 0);
  void* ptr = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&ptr, bytes));
  cudaIpcMemHandle_t handle;
  try {
    C10_CUDA_CHECK(cudaIpcGetMemHandle(&handle, ptr));
    C10_CUDA_CHECK(cudaMemset(ptr, 0, bytes));
  } catch (...) {
    cudaFree(ptr);
    throw;
  }
  return {reinterpret_cast<int64_t>(ptr),
          pybind11::bytes(reinterpret_cast<char*>(&handle), sizeof(handle))};
}
int64_t open_handle(pybind11::bytes bytes) {
  std::string value = bytes;
  STD_TORCH_CHECK(value.size() == sizeof(cudaIpcMemHandle_t));
  cudaIpcMemHandle_t handle;
  std::memcpy(&handle, value.data(), sizeof(handle));
  void* ptr = nullptr;
  C10_CUDA_CHECK(
      cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<int64_t>(ptr);
}
void release(int64_t ptr, bool owner) {
  if (owner)
    C10_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(ptr)));
  else
    C10_CUDA_CHECK(cudaIpcCloseMemHandle(reinterpret_cast<void*>(ptr)));
}
void run(torch::Tensor input, torch::Tensor codes, torch::Tensor output,
         std::vector<int64_t> pointers, std::vector<int64_t> flags, int rank,
         unsigned epoch) {
  STD_TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::headeronly::ScalarType::Float &&
              input.is_contiguous());
  STD_TORCH_CHECK(codes.is_cuda() && codes.scalar_type() == torch::headeronly::ScalarType::Byte &&
              codes.is_contiguous());
  STD_TORCH_CHECK(output.is_cuda() && output.scalar_type() == torch::headeronly::ScalarType::Float &&
              output.is_contiguous());
  STD_TORCH_CHECK(input.device() == codes.device() &&
              input.device() == output.device());
  STD_TORCH_CHECK(rank >= 0 && rank < 4 && pointers.size() == 4 &&
              flags.size() == 4 && epoch > 0);
  STD_TORCH_CHECK(input.numel() == output.numel() * 4 &&
              codes.numel() == output.numel());
  const torch::stable::accelerator::DeviceGuard guard(input.device().index());
  auto stream = get_current_cuda_stream();
  cudaDeviceProp _cb_props_properties;
  int _cb_dev_properties = input.device().index();
cudaGetDeviceProperties(&_cb_props_properties, _cb_dev_properties);
const auto* properties = &_cb_props_properties;
  STD_TORCH_CHECK(
      properties->major == 7 && properties->minor == 0 &&
          properties->multiProcessorCount >= GRID,
      "Exact peer reduction requires an SM70 device with at least 80 SMs");
  Peers peers;
  for (int i = 0; i < 4; ++i) {
    STD_TORCH_CHECK(pointers[i] && flags[i]);
    peers.data[i] = reinterpret_cast<float*>(pointers[i]);
    peers.flags[i] = reinterpret_cast<unsigned*>(flags[i]);
  }
  C10_CUDA_CHECK(cudaMemcpyAsync(peers.data[rank], input.data_ptr(),
                                 input.nbytes(), cudaMemcpyDeviceToDevice,
                                 stream));
  reduce_rows<<<GRID, 256, 0, stream>>>(
      peers, codes.data_ptr<uint8_t>(), output.data_ptr<float>(),
      output.numel(), output.numel() * rank, rank, epoch);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("allocate", &allocate);
  m.def("open_handle", &open_handle);
  m.def("release", &release);
  m.def("run", &run);
}
