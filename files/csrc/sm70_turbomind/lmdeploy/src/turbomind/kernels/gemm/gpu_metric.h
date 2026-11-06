// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/types.h"
#include <libtorch_stable/torch_utils.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/HeaderOnlyArrayRef.h>

namespace turbomind::gemm {

// bytes / second
float MeasureL2CacheThroughput();

// fused multiply-add / second
float MeasureMmaThroughput(int proble_size = 16384);

}  // namespace turbomind::gemm
