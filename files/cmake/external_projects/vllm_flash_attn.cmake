# vLLM flash attention requires VLLM_GPU_ARCHES to contain the set of target
# arches in the CMake syntax (75-real, 89-virtual, etc), since we clear the
# arches in the CUDA case (and instead set the gencodes on a per file basis)
# we need to manually set VLLM_GPU_ARCHES here.
if(VLLM_GPU_LANG STREQUAL "CUDA")
  foreach(_ARCH ${CUDA_ARCHS})
    string(REPLACE "." "" _ARCH "${_ARCH}")
    list(APPEND VLLM_GPU_ARCHES "${_ARCH}-real")
  endforeach()
endif()

#
# Build vLLM flash attention from source
#
# IMPORTANT: This has to be the last thing we do, because vllm-flash-attn uses the same macros/functions as vLLM.
# Because functions all belong to the global scope, vllm-flash-attn's functions overwrite vLLMs.
# They should be identical but if they aren't, this is a massive footgun.
#
# The vllm-flash-attn install rules are nested under vllm to make sure the library gets installed in the correct place.
# To only install vllm-flash-attn, use --component _vllm_fa2_C (for FA2), --component _vllm_fa3_C (for FA3),
# or --component _vllm_fa4_cutedsl_C (for FA4 CuteDSL Python files).
# If no component is specified, vllm-flash-attn is still installed.

# If VLLM_FLASH_ATTN_SRC_DIR is set, vllm-flash-attn is installed from that directory instead of downloading.
# This is to enable local development of vllm-flash-attn within vLLM.
# It can be set as an environment variable or passed as a cmake argument.
# The environment variable takes precedence.
if (DEFINED ENV{VLLM_FLASH_ATTN_SRC_DIR})
  set(VLLM_FLASH_ATTN_SRC_DIR $ENV{VLLM_FLASH_ATTN_SRC_DIR})
endif()

if(VLLM_FLASH_ATTN_SRC_DIR)
  FetchContent_Declare(
          vllm-flash-attn SOURCE_DIR
          ${VLLM_FLASH_ATTN_SRC_DIR}
          BINARY_DIR ${CMAKE_BINARY_DIR}/vllm-flash-attn
  )
else()
  FetchContent_Declare(
          vllm-flash-attn
          GIT_REPOSITORY https://github.com/vllm-project/flash-attention.git
          GIT_TAG 9cd61de38763d712bb6ce56e2a02cc2bf718c89f
          GIT_PROGRESS TRUE
          # Don't share the vllm-flash-attn build between build types
          BINARY_DIR ${CMAKE_BINARY_DIR}/vllm-flash-attn
  )
endif()

# Make sure vllm-flash-attn install rules are nested under vllm/
# ALL_COMPONENTS ensures the save/modify/restore runs exactly once regardless
# of how many components are being installed, avoiding double-append of /vllm/.
install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY FALSE)" ALL_COMPONENTS)
install(CODE "set(OLD_CMAKE_INSTALL_PREFIX \"\${CMAKE_INSTALL_PREFIX}\")" ALL_COMPONENTS)
install(CODE "set(CMAKE_INSTALL_PREFIX \"\${CMAKE_INSTALL_PREFIX}/vllm/\")" ALL_COMPONENTS)

# Fetch the vllm-flash-attn library
FetchContent_MakeAvailable(vllm-flash-attn)
message(STATUS "vllm-flash-attn is available at ${vllm-flash-attn_SOURCE_DIR}")

# Keep the precision-qualified SM70 prefill route in the parent repository.
# Its private CUTLASS visitors have distinct types and do not modify the
# legacy FA2 headers or operators, which remain available for rollback.
if(VLLM_FLASH_ATTN_SM70 AND TARGET _vllm_fa2_C)
  set(SM70_V37_DIR "${CMAKE_CURRENT_LIST_DIR}/../../csrc/attention/sm70_v37")
  set(SM70_V37_CUDA_SRCS
    "${SM70_V37_DIR}/prefill.cu"
    "${SM70_V37_DIR}/tail.cu"
    "${SM70_V37_DIR}/bridge.cu")
   # FA2 is created in a child directory. Source properties must be visible in
   # that target's scope; setting them only in the parent silently loses SM70.
    set_source_files_properties(${SM70_V37_CUDA_SRCS}
      TARGET_DIRECTORY _vllm_fa2_C
      PROPERTIES COMPILE_OPTIONS "-gencode=arch=compute_70,code=sm_70;-I${vllm-flash-attn_SOURCE_DIR}/csrc/cutlass/examples/35_gemm_softmax"
      COMPILE_DEFINITIONS "PREFIX_TORCH_EXTENSION")
   target_sources(_vllm_fa2_C PRIVATE
     ${SM70_V37_CUDA_SRCS}
     "${SM70_V37_DIR}/register.cpp")
    # SM70 sources reach into the parent csrc tree (libtorch_stable helpers).
    target_include_directories(_vllm_fa2_C PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../../csrc")
    # The missing typed data_ptr specializations (half family, char, byte)
    # must exist in every extension that uses them.
    set(STABLE_DATA_PTR_SHIM_SRC
      "${CMAKE_CURRENT_LIST_DIR}/../../csrc/libtorch_stable/stable_data_ptr_shim.cu")
     set_source_files_properties(${STABLE_DATA_PTR_SHIM_SRC}
       TARGET_DIRECTORY _vllm_fa2_C
       PROPERTIES COMPILE_DEFINITIONS "PREFIX_TORCH_EXTENSION")
    target_sources(_vllm_fa2_C PRIVATE ${STABLE_DATA_PTR_SHIM_SRC})
endif()

# The grouped E4M3 FP32 long-context route ships inside the same extension, so
# the accelerated path is available without an externally built DSO.
if(VLLM_FLASH_ATTN_SM70 AND TARGET _vllm_fa2_C)
  set(SM70_GROUPED_LONG_DIR
      "${CMAKE_CURRENT_LIST_DIR}/../../csrc/attention/sm70_grouped_long")
  set(SM70_GROUPED_LONG_SRC
      "${SM70_GROUPED_LONG_DIR}/kernel/grouped-attention.cu"
      "${SM70_GROUPED_LONG_DIR}/kernel/scalar-attention.cu")
  # Flags mirror the manifest the operator was qualified with. As with v37, the
  # properties must be set in the target scope or SM70 silently loses them.
  set_source_files_properties(${SM70_GROUPED_LONG_SRC}
    TARGET_DIRECTORY _vllm_fa2_C
    PROPERTIES COMPILE_OPTIONS
      "-gencode=arch=compute_70,code=sm_70;-O3;-std=c++17;--use_fast_math;--expt-relaxed-constexpr;--expt-extended-lambda;-U__CUDA_NO_HALF_OPERATORS__;-U__CUDA_NO_HALF_CONVERSIONS__;-U__CUDA_NO_HALF2_OPERATORS__")
  target_include_directories(_vllm_fa2_C PRIVATE "${SM70_GROUPED_LONG_DIR}/include")
  target_sources(_vllm_fa2_C PRIVATE ${SM70_GROUPED_LONG_SRC})
endif()

# SM70 long-context prefill (79t). Guarded: this module is shared with every
# non-SM70 build, and sm70_79t.cmake must not define targets for them.
if(VLLM_FLASH_ATTN_SM70)
  include("${CMAKE_CURRENT_LIST_DIR}/../sm70_79t.cmake")
endif()

# Restore the install prefix after FA's install rules
install(CODE "set(CMAKE_INSTALL_PREFIX \"\${OLD_CMAKE_INSTALL_PREFIX}\")" ALL_COMPONENTS)
install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" ALL_COMPONENTS)

# Install shared Python files for both FA2 and FA3 components
foreach(_FA_COMPONENT _vllm_fa2_C _vllm_fa3_C)
  # Ensure the vllm/vllm_flash_attn directory exists before installation
  install(CODE "file(MAKE_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/vllm/vllm_flash_attn\")"
    COMPONENT ${_FA_COMPONENT})

  # Copy vllm_flash_attn python files (except __init__.py and flash_attn_interface.py
  # which are source-controlled in vllm)
  install(
    DIRECTORY ${vllm-flash-attn_SOURCE_DIR}/vllm_flash_attn/
    DESTINATION vllm/vllm_flash_attn
    COMPONENT ${_FA_COMPONENT}
    FILES_MATCHING PATTERN "*.py"
    PATTERN "__init__.py" EXCLUDE
    PATTERN "flash_attn_interface.py" EXCLUDE
  )

endforeach()

#
# FA4 CuteDSL component
# This is a Python-only component that copies the flash_attn/cute directory
# and transforms imports to match our package structure.
#
add_custom_target(_vllm_fa4_cutedsl_C)

# Install flash_attn/cute directory (needed for FA4).
# When using a local source dir (VLLM_FLASH_ATTN_SRC_DIR), create a symlink
# so edits to cute-dsl Python files take effect immediately without rebuilding.
# Otherwise, copy files and transform flash_attn.cute imports to
# vllm.vllm_flash_attn.cute to match our package structure.
if(VLLM_FLASH_ATTN_SRC_DIR)
  install(CODE "
    set(LINK_TARGET \"${vllm-flash-attn_SOURCE_DIR}/flash_attn/cute\")
    set(LINK_NAME \"\${CMAKE_INSTALL_PREFIX}/vllm/vllm_flash_attn/cute\")
    file(MAKE_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/vllm/vllm_flash_attn\")
    file(REMOVE_RECURSE \"\${LINK_NAME}\")
    file(CREATE_LINK \"\${LINK_TARGET}\" \"\${LINK_NAME}\" SYMBOLIC)
  " COMPONENT _vllm_fa4_cutedsl_C)
else()
  install(CODE "
    file(GLOB_RECURSE CUTE_PY_FILES \"${vllm-flash-attn_SOURCE_DIR}/flash_attn/cute/*.py\")
    foreach(SRC_FILE \${CUTE_PY_FILES})
      file(RELATIVE_PATH REL_PATH \"${vllm-flash-attn_SOURCE_DIR}/flash_attn/cute\" \${SRC_FILE})
      set(DST_FILE \"\${CMAKE_INSTALL_PREFIX}/vllm/vllm_flash_attn/cute/\${REL_PATH}\")
      get_filename_component(DST_DIR \${DST_FILE} DIRECTORY)
      file(MAKE_DIRECTORY \${DST_DIR})
      file(READ \${SRC_FILE} FILE_CONTENTS)
      string(REPLACE \"flash_attn.cute\" \"vllm.vllm_flash_attn.cute\" FILE_CONTENTS \"\${FILE_CONTENTS}\")
      file(WRITE \${DST_FILE} \"\${FILE_CONTENTS}\")
    endforeach()
  " COMPONENT _vllm_fa4_cutedsl_C)
endif()
