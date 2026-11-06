#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/headeronly/version.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <vector>

namespace {
// This translation unit cannot use a from_blob deleter: the torch 2.10 stable
// ABI's from_blob takes no deleter argument (the overload that does is newer),
// which is also why get_cuda_view_from_cpu_tensor cannot hand the CPU tensor's
// lifetime to the view it returns. Both the pinned CPU tensors and the staging
// buffers allocated here are therefore kept alive until process exit.
//
// Two consequences, accepted rather than accidental:
//  * memory held this way is never released before exit;
//  * the registry is touched from every thread that offloads KV, so it is
//    mutex-guarded (the earlier version mutated a plain static vector).
struct MappedCpuViewRegistry {
  std::mutex mu;
  std::vector<torch::stable::Tensor> keep_alive;  // pinned CPU tensors
  std::vector<void*> buffers;                     // cudaHostAlloc'd staging
};

MappedCpuViewRegistry& mapped_cpu_view_registry() {
  // Deliberately never destroyed: running torch tensor destructors during
  // process teardown is riskier than letting the OS reclaim the memory. Only
  // the host buffers below are freed explicitly, and that happens in an atexit
  // handler that makes no torch calls.
  static MappedCpuViewRegistry* registry = new MappedCpuViewRegistry();
  return *registry;
}

void free_mapped_host_buffers() {
  for (void* p : mapped_cpu_view_registry().buffers) {
    cudaFreeHost(p);
  }
}

void register_host_buffer_atexit_once() {
  static bool registered = [] {
    std::atexit(free_mapped_host_buffers);
    return true;
  }();
  (void)registered;
}
}  // namespace

// This function assumes that `cpu_tensor` is a CPU tensor,
// and that UVA (Unified Virtual Addressing) is enabled.
torch::stable::Tensor get_cuda_view_from_cpu_tensor(
    torch::stable::Tensor& cpu_tensor) {
  STD_TORCH_CHECK(cpu_tensor.device().is_cpu(), "Input tensor must be on CPU");

  const auto dtype = cpu_tensor.scalar_type();
  const auto layout = torch::headeronly::Layout::Strided;
  const torch::stable::Device cuda_dev(torch::headeronly::DeviceType::CUDA);

  // handle empty tensor
  if (cpu_tensor.numel() == 0) {
    return torch::stable::empty(cpu_tensor.sizes(), dtype, layout, cuda_dev);
  }

  std::array<StableIValue, 2> is_pinned_stack{
      torch::stable::detail::from(cpu_tensor),
      torch::stable::detail::from(std::nullopt)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::is_pinned", "", is_pinned_stack.data(), TORCH_ABI_VERSION));
  if (torch::stable::detail::to<bool>(is_pinned_stack[0])) {
    // If CPU tensor is pinned, directly get the device pointer.
    void* host_ptr = const_cast<void*>(cpu_tensor.mutable_data_ptr());
    void* device_ptr = nullptr;
    cudaError_t err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
    STD_TORCH_CHECK(err == cudaSuccess, "cudaHostGetDevicePointer failed: ",
                    cudaGetErrorString(err));

    // Keep the pinned tensor alive for as long as the view can be used: with no
    // deleter available, the registry is the only place that can hold it.
    {
      auto& reg = mapped_cpu_view_registry();
      std::lock_guard<std::mutex> lock(reg.mu);
      reg.keep_alive.push_back(cpu_tensor);
    }

    return torch::stable::from_blob(
        device_ptr, cpu_tensor.sizes(), cpu_tensor.strides(), cuda_dev, dtype);
  }

  // If CPU tensor is not pinned, allocate a new pinned memory buffer.
  torch::stable::Tensor contiguous_cpu = torch::stable::contiguous(cpu_tensor);
  size_t nbytes = contiguous_cpu.numel() * contiguous_cpu.element_size();

  void* host_ptr = nullptr;
  cudaError_t err = cudaHostAlloc(&host_ptr, nbytes, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    STD_TORCH_CHECK(false, "cudaHostAlloc failed: ", cudaGetErrorString(err));
  }

  err = cudaMemcpy(host_ptr, contiguous_cpu.const_data_ptr(), nbytes,
                   cudaMemcpyDefault);
  if (err != cudaSuccess) {
    cudaFreeHost(host_ptr);
    STD_TORCH_CHECK(false, "cudaMemcpy failed: ", cudaGetErrorString(err));
  }

  void* device_ptr = nullptr;
  err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
  if (err != cudaSuccess) {
    cudaFreeHost(host_ptr);
    STD_TORCH_CHECK(
        false, "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));
  }

  // No deleter on this torch version, so the staging buffer is freed at exit.
  // cudaFreeHost is a real free (unlike the pinned-tensor case, where we only
  // need the owner to outlive the view).
  {
    register_host_buffer_atexit_once();
    auto& reg = mapped_cpu_view_registry();
    std::lock_guard<std::mutex> lock(reg.mu);
    reg.buffers.push_back(host_ptr);
  }

  return torch::stable::from_blob(device_ptr, contiguous_cpu.sizes(),
                                  contiguous_cpu.strides(), cuda_dev,
                                  contiguous_cpu.scalar_type());
}
