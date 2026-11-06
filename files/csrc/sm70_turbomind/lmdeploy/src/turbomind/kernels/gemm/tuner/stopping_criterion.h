// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/tuner/stats.h"
#include <memory>
#include <libtorch_stable/torch_utils.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/HeaderOnlyArrayRef.h>

namespace turbomind::gemm {

class StoppingCriterion {
public:
    virtual ~StoppingCriterion()                 = default;
    virtual bool should_stop(const Stats& stats) = 0;
};

std::unique_ptr<StoppingCriterion> CreateStoppingCriterion(int min_iter, int max_iter, float max_ms);

}  // namespace turbomind::gemm
