# FA2 (fused multi-head attention) extension for SM70/V100, ported from
# 1Cat-vLLM (zhinianqin/flash-attention-v100 @ c2eda5e).
if (VLLM_GPU_LANG STREQUAL "CUDA")
  list(APPEND VLLM_FA_GPU_FLAGS --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math)
  if (NVCC_THREADS AND NVCC_THREADS GREATER 1)
    list(APPEND VLLM_FA_GPU_FLAGS "--threads=${NVCC_THREADS}")
  endif()
  # Replace instead of appending; nvcc dislikes duplicate -O flags.
  string(REPLACE "-O2" "-O3" CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}")

  file(GLOB FA2_GEN_SRCS
    "csrc/flash_attn/src/flash_fwd_split_hdim*.cu"
    "csrc/flash_attn/src/flash_fwd_hdim*.cu")

  set(FA2_ARCHS "7.0")
  message(STATUS "[vllm-flash-attn] FA2_ARCHS: ${FA2_ARCHS}")
  set_gencode_flags_for_srcs(
    SRCS "${FA2_GEN_SRCS}"
    CUDA_ARCHS "${FA2_ARCHS}")

  define_extension_target(
    _vllm_fa2_C
    DESTINATION vllm_flash_attn
    LANGUAGE ${VLLM_GPU_LANG}
    SOURCES
      csrc/flash_attn/flash_api.cpp
      csrc/flash_attn/flash_api_torch_lib.cpp
      ${FA2_GEN_SRCS}
    COMPILE_FLAGS ${VLLM_FA_GPU_FLAGS}
    ARCHITECTURES "70"
    USE_SABI 3
    WITH_SOABI)

  target_include_directories(_vllm_fa2_C PRIVATE
    csrc/flash_attn
    csrc/flash_attn/src
    csrc/common
    ${cutlass_SOURCE_DIR}/include)

  target_compile_definitions(_vllm_fa2_C PRIVATE
    FLASHATTENTION_DISABLE_BACKWARD
    FLASHATTENTION_DISABLE_DROPOUT
    FLASHATTENTION_DISABLE_PYBIND)

  target_link_libraries(_vllm_fa2_C PRIVATE "${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so")
endif()
