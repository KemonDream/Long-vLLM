
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

/// TODO: remove this dependency
#include "src/turbomind/utils/cuda_utils.h"
#include <libtorch_stable/torch_utils.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/HeaderOnlyArrayRef.h>

namespace turbomind::core {

class Allocator;
class Buffer;
class Stream;
class Event;
class Context;

using std::shared_ptr;
using std::vector;

using ssize_t = std::ptrdiff_t;

}  // namespace turbomind::core
