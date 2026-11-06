// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/tuner/cache_utils.h"
#include <libtorch_stable/torch_utils.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/HeaderOnlyArrayRef.h>

namespace turbomind::gemm {

CacheFlushing::CacheFlushing()
{
    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, 0);

    size_ = props.l2CacheSize;

    cudaMalloc(&buffer_, size_);
}

void CacheFlushing::flush(cudaStream_t stream)
{
    thread_local CacheFlushing inst{};
    inst(stream);
}

void CacheFlushing::operator()(cudaStream_t stream) const
{
    cudaMemsetAsync(buffer_, 0, size_, stream);
}

}  // namespace turbomind::gemm
