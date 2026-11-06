// Provides torch::Tensor for ops.h (previously included transitively via
// cache.h, which is no longer included here after cache ops moved to
// _C_stable_libtorch).
#include <torch/all.h>
#include "ops.h"
#include "core/registration.h"
#include <torch/library.h>
#include <torch/version.h>

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

#ifndef USE_ROCM
TORCH_LIBRARY_FRAGMENT(CONCAT(TORCH_EXTENSION_NAME, _custom_ar), custom_ar) {
  // Custom all-reduce kernels (SM70 / V100 support, ported from 1Cat-vLLM)
  custom_ar.def(
      "init_custom_ar(int[] ipc_tensors, Tensor rank_data, "
      "int rank, bool fully_connected) -> int");
  custom_ar.impl("init_custom_ar", torch::kCUDA, &init_custom_ar);
  custom_ar.def(
      "all_reduce(int fa, Tensor inp, Tensor! out, int reg_buffer, "
      "int reg_buffer_sz_bytes) -> ()");
  custom_ar.impl("all_reduce", torch::kCUDA, &all_reduce);
  custom_ar.def(
      "sm70_tp2_all_reduce_gemma_rms_norm(int fa, Tensor inp, Tensor "
      "residual, Tensor weight, Tensor! normalized_out, Tensor! residual_out, "
      "int reg_buffer, int reg_buffer_sz_bytes, float epsilon) -> ()");
  custom_ar.impl("sm70_tp2_all_reduce_gemma_rms_norm", torch::kCUDA,
                 &sm70_tp2_all_reduce_gemma_rms_norm);
  custom_ar.def(
      "sm70_tp4_all_reduce_gemma_rms_norm(int fa, Tensor inp, Tensor "
      "residual, Tensor weight, Tensor! normalized_out, Tensor! residual_out, "
      "int reg_buffer, int reg_buffer_sz_bytes, float epsilon) -> ()");
  custom_ar.impl("sm70_tp4_all_reduce_gemma_rms_norm", torch::kCUDA,
                 &sm70_tp4_all_reduce_gemma_rms_norm);
  custom_ar.def(
      "sm70_tp4_reduce_scatter_gemma_rms_norm_all_gather(int fa, Tensor inp, "
      "Tensor residual, Tensor weight, Tensor! normalized_out, Tensor! "
      "residual_out, int reg_input_buffer, int reg_output_buffer, int "
      "reg_buffer_sz_bytes, float epsilon) -> ()");
  custom_ar.impl("sm70_tp4_reduce_scatter_gemma_rms_norm_all_gather",
                 torch::kCUDA,
                 &sm70_tp4_reduce_scatter_gemma_rms_norm_all_gather);
  custom_ar.def(
      "all_reduce_sum2(int fa, Tensor inp_a, Tensor inp_b, Tensor! out) -> ()");
  custom_ar.impl("all_reduce_sum2", torch::kCUDA, &all_reduce_sum2);
  custom_ar.def(
      "sm70_qwen38_hc_down_allgather(int fa, Tensor inp, Tensor! out) -> ()");
  custom_ar.impl("sm70_qwen38_hc_down_allgather", torch::kCUDA,
                 &sm70_qwen38_hc_down_allgather);
  custom_ar.def(
      "sm70_qwen38_hc_gate_mix(int fa, Tensor local_gate, Tensor branches, "
      "Tensor! out) -> ()");
  custom_ar.impl("sm70_qwen38_hc_gate_mix", torch::kCUDA,
                 &sm70_qwen38_hc_gate_mix);
  custom_ar.def(
      "sm70_qwen38_hc_output_allgather(int fa, Tensor local_block, "
      "Tensor! out) -> ()");
  custom_ar.impl("sm70_qwen38_hc_output_allgather", torch::kCUDA,
                 &sm70_qwen38_hc_output_allgather);
  custom_ar.def(
      "sm70_qwen38_hc_up_mix_allgather(int fa, Tensor lora, Tensor weight, "
      "Tensor branches, Tensor! out) -> ()");
  custom_ar.impl("sm70_qwen38_hc_up_mix_allgather", torch::kCUDA,
                 &sm70_qwen38_hc_up_mix_allgather);
  custom_ar.def(
      "top1_argmax(int fa, Tensor input_pair, Tensor! output, int reg_buffer, "
      "int reg_buffer_sz_bytes) -> ()");
  custom_ar.impl("top1_argmax", torch::kCUDA, &top1_argmax);
  custom_ar.def(
      "tile_runtime_all_reduce(int fa, Tensor inp, Tensor! out, int "
      "reg_buffer, int reg_buffer_sz_bytes, int tile_numel, int "
      "engine_blocks, int compute_iters) -> ()");
  custom_ar.impl("tile_runtime_all_reduce", torch::kCUDA,
                 &tile_runtime_all_reduce);
  custom_ar.def(
      "tile_runtime_all_reduce_engine(int fa, Tensor inp, Tensor! out, int "
      "reg_buffer, int reg_buffer_sz_bytes, int tile_numel, int "
      "producer_blocks, int reducer_blocks, int compute_iters) -> ()");
  custom_ar.impl("tile_runtime_all_reduce_engine", torch::kCUDA,
                 &tile_runtime_all_reduce_engine);
  custom_ar.def(
      "tile_runtime_wait_reduce(int fa, Tensor staging, Tensor! out, "
      "int tile_numel, int reducer_blocks) -> ()");
  custom_ar.impl("tile_runtime_wait_reduce", torch::kCUDA,
                 &tile_runtime_wait_reduce);
  custom_ar.def("dispose", &dispose);
  custom_ar.def("meta_size", &meta_size);
  custom_ar.def("sm70_tp4_push_allreduce_buffer_size",
                &sm70_tp4_push_allreduce_buffer_size);
  custom_ar.def("sm70_tp8_hierarchical_push_allreduce_buffer_size",
                &sm70_tp8_hierarchical_push_allreduce_buffer_size);
  custom_ar.def("register_buffer", &register_buffer);
  custom_ar.def("register_sm70_tp4_push_allreduce_buffer",
                &register_sm70_tp4_push_allreduce_buffer);
  custom_ar.def("register_sm70_tp8_hierarchical_push_allreduce_buffer",
                &register_sm70_tp8_hierarchical_push_allreduce_buffer);
  custom_ar.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  custom_ar.def("register_graph_buffers", &register_graph_buffers);
  custom_ar.def("allocate_shared_buffer_and_handle",
                &allocate_shared_buffer_and_handle);
  custom_ar.def("open_mem_handle(Tensor mem_handle) -> int", &open_mem_handle);
  custom_ar.impl("open_mem_handle", torch::kCPU, &open_mem_handle);
  custom_ar.def("free_shared_buffer", &free_shared_buffer);
}
#endif

#ifdef USE_ROCM
TORCH_LIBRARY_FRAGMENT(CONCAT(TORCH_EXTENSION_NAME, _custom_ar), custom_ar) {
  // Quick Reduce all-reduce kernels (ROCm-only; stays on legacy _C).
  custom_ar.def(
      "qr_all_reduce(int fa, Tensor inp, Tensor out, int quant_level, bool "
      "cast_bf2half) -> ()");
  custom_ar.impl("qr_all_reduce", torch::kCUDA, &qr_all_reduce);

  custom_ar.def("init_custom_qr", &init_custom_qr);
  custom_ar.def("qr_destroy", &qr_destroy);
  custom_ar.def("qr_get_handle", &qr_get_handle);

  custom_ar.def("qr_open_handles(int _fa, Tensor[](b!) handles) -> ()");
  custom_ar.impl("qr_open_handles", torch::kCPU, &qr_open_handles);

  custom_ar.def("qr_max_size", &qr_max_size);
}
#endif

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
