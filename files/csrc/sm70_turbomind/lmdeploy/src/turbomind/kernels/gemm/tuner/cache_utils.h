// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstdint>
#include <libtorch_stable/torch_utils.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/HeaderOnlyArrayRef.h>

namespace turbomind::gemm {

class CacheFlushing {
public:
    static void flush(cudaStream_t stream = {});

private:
    CacheFlushing();
    void operator()(cudaStream_t stream) const;

    uint32_t* buffer_;
    size_t    size_;
};

}  // namespace turbomind::gemm
