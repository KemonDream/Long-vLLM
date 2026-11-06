// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

#include <cublasLt.h>
#include <libtorch_stable/torch_utils.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/HeaderOnlyArrayRef.h>

namespace {

void check_cublaslt(cublasStatus_t status, const char* operation) {
  STD_TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, operation,
              " failed with cuBLAS status ", static_cast<int>(status));
}

struct MatmulResources {
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t input = nullptr;
  cublasLtMatrixLayout_t weight = nullptr;
  cublasLtMatrixLayout_t output = nullptr;

  ~MatmulResources() {
    if (input != nullptr) {
      cublasLtMatrixLayoutDestroy(input);
    }
    if (weight != nullptr) {
      cublasLtMatrixLayoutDestroy(weight);
    }
    if (output != nullptr) {
      cublasLtMatrixLayoutDestroy(output);
    }
    if (operation != nullptr) {
      cublasLtMatmulDescDestroy(operation);
    }
  }
};

void create_row_major_layout(cublasLtMatrixLayout_t* layout, int64_t rows,
                             int64_t columns) {
  check_cublaslt(
      cublasLtMatrixLayoutCreate(layout, CUDA_R_16F, rows, columns, columns),
      "cublasLtMatrixLayoutCreate");
  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  check_cublaslt(
      cublasLtMatrixLayoutSetAttribute(*layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                       &order, sizeof(order)),
      "set row-major matrix layout");
}

template <typename T>
void set_algorithm_attribute(cublasLtMatmulAlgo_t* algorithm,
                             cublasLtMatmulAlgoConfigAttributes_t attribute,
                             T value) {
  check_cublaslt(cublasLtMatmulAlgoConfigSetAttribute(algorithm, attribute,
                                                      &value, sizeof(value)),
                 "cublasLtMatmulAlgoConfigSetAttribute");
}

}  // namespace

void sm70_glm53_tp8_cublaslt_out(torch::stable::Tensor out, torch::stable::Tensor input,
                                 torch::stable::Tensor weight) {
  constexpr int64_t kTokens = 8;
  STD_TORCH_CHECK(out.is_cuda() && input.is_cuda() && weight.is_cuda(),
              "sm70_glm53_tp8_cublaslt_out: tensors must be CUDA tensors");
  STD_TORCH_CHECK(out.scalar_type() == torch::headeronly::ScalarType::Half &&
                  input.scalar_type() == torch::headeronly::ScalarType::Half &&
                  weight.scalar_type() == torch::headeronly::ScalarType::Half,
              "sm70_glm53_tp8_cublaslt_out: tensors must be float16");
  STD_TORCH_CHECK(
      out.is_contiguous() && input.is_contiguous() && weight.is_contiguous(),
      "sm70_glm53_tp8_cublaslt_out: tensors must be contiguous");
  STD_TORCH_CHECK(out.dim() == 2 && input.dim() == 2 && weight.dim() == 2,
              "sm70_glm53_tp8_cublaslt_out: tensors must be rank two");
  STD_TORCH_CHECK(input.get_device_index() == out.get_device() &&
                  input.get_device_index() == weight.get_device(),
              "sm70_glm53_tp8_cublaslt_out: tensors must share one device");

  const int64_t m = input.size(0);
  const int64_t k = input.size(1);
  const int64_t n = weight.size(0);
  const bool in_projection = m == kTokens && k == 4096 && n == 3336 &&
                             stable_array_ref_eq(weight.sizes(), torch::headeronly::IntHeaderOnlyArrayRef({3336, 4096}));
  const bool out_projection =
      m == kTokens && k == 1024 && n == 4096 &&
      stable_array_ref_eq(weight.sizes(), torch::headeronly::IntHeaderOnlyArrayRef({4096, 1024}));
  STD_TORCH_CHECK(in_projection || out_projection,
              "sm70_glm53_tp8_cublaslt_out: unsupported GLM-5.3 TP8 shape");
  STD_TORCH_CHECK(stable_array_ref_eq(out.sizes(), torch::headeronly::IntHeaderOnlyArrayRef({m, n})),
              "sm70_glm53_tp8_cublaslt_out: output shape mismatch");

  const torch::stable::accelerator::DeviceGuard device_guard(input.get_device_index());
  cudaDeviceProp cb_lt_props_local;
int cb_lt_dev = input.get_device_index();
cudaGetDeviceProperties(&cb_lt_props_local, cb_lt_dev);
const auto* properties = &cb_lt_props_local;
  STD_TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "sm70_glm53_tp8_cublaslt_out: requires SM70");
  static cublasLtHandle_t cb_lt_handle = [] {
    cublasLtHandle_t h = nullptr;
    if (cublasLtCreate(&h) != CUBLAS_STATUS_SUCCESS) {
      h = nullptr;
    }
    return h;
  }();
  STD_TORCH_CHECK(cb_lt_handle != nullptr,
              "sm70_glm53_tp8_cublaslt_out: cuBLASLt handle creation failed");
  auto handle = cb_lt_handle;

  MatmulResources resources;
  check_cublaslt(cublasLtMatmulDescCreate(&resources.operation,
                                          CUBLAS_COMPUTE_32F, CUDA_R_32F),
                 "cublasLtMatmulDescCreate");
  cublasOperation_t transpose_a = CUBLAS_OP_N;
  cublasOperation_t transpose_b = CUBLAS_OP_T;
  check_cublaslt(cublasLtMatmulDescSetAttribute(
                     resources.operation, CUBLASLT_MATMUL_DESC_TRANSA,
                     &transpose_a, sizeof(transpose_a)),
                 "set transpose A");
  check_cublaslt(cublasLtMatmulDescSetAttribute(
                     resources.operation, CUBLASLT_MATMUL_DESC_TRANSB,
                     &transpose_b, sizeof(transpose_b)),
                 "set transpose B");
  create_row_major_layout(&resources.input, m, k);
  create_row_major_layout(&resources.weight, n, k);
  create_row_major_layout(&resources.output, m, n);

  cublasLtMatmulAlgo_t algorithm{};
  check_cublaslt(cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,
                                        CUDA_R_16F, CUDA_R_16F, CUDA_R_16F,
                                        CUDA_R_16F, 21, &algorithm),
                 "cublasLtMatmulAlgoInit");
  set_algorithm_attribute(&algorithm, CUBLASLT_ALGO_CONFIG_TILE_ID,
                          uint32_t{5});
  set_algorithm_attribute(&algorithm, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                          uint32_t{20});
  set_algorithm_attribute(&algorithm, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                          uint32_t{0});
  set_algorithm_attribute(&algorithm, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
                          uint32_t{0});

  cublasLtMatmulHeuristicResult_t checked{};
  checked.algo = algorithm;
  check_cublaslt(
      cublasLtMatmulAlgoCheck(handle, resources.operation, resources.input,
                              resources.weight, resources.output,
                              resources.output, &checked.algo, &checked),
      "cublasLtMatmulAlgoCheck");
  STD_TORCH_CHECK(
      checked.state == CUBLAS_STATUS_SUCCESS && checked.workspaceSize == 0,
      "sm70_glm53_tp8_cublaslt_out: pinned algorithm is unavailable");

  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  check_cublaslt(
      cublasLtMatmul(handle, resources.operation, &alpha, input.data_ptr(),
                     resources.input, weight.data_ptr(), resources.weight,
                     &beta, out.data_ptr(), resources.output, out.data_ptr(),
                     resources.output, &checked.algo, nullptr, 0,
                     get_current_cuda_stream(input.get_device_index())),
      "cublasLtMatmul");
}
