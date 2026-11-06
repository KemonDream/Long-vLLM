#pragma once
// Compatibility shims for SM70 attention translation units built in
// stable-only mode (TORCH_TARGET_VERSION defined), where the classic
// ATen/c10 assertion macros are unavailable.
#include <cstdio>
#include <sstream>
#include <type_traits>
#include <torch/csrc/stable/macros.h>
#include <torch/headeronly/util/Exception.h>

#ifndef TORCH_CHECK
#define TORCH_CHECK(cond, ...) STD_TORCH_CHECK(cond, ##__VA_ARGS__)
#endif

namespace sm70_stable_compat_detail {

template <typename T>
void AppendArg(std::ostream& os, const T& v) {
  using U = std::remove_cv_t<std::remove_reference_t<T>>;
  os << ' ';
  if constexpr (std::is_arithmetic_v<U> || std::is_enum_v<U>) {
    os << v;
  } else {
    os << "(object)";
  }
}

template <typename... Args>
[[maybe_unused]] void WarnImpl(const char* head, Args... rest) {
  std::ostringstream oss;
  oss << head;
  (void)std::initializer_list<int>{(AppendArg(oss, rest), 0)...};
  fprintf(stderr, "[WARN] %s\n", oss.str().c_str());
  fflush(stderr);
}

}  // namespace sm70_stable_compat_detail

#if !defined(TORCH_WARN)
#define TORCH_WARN(head, ...) ::sm70_stable_compat_detail::WarnImpl(head, ##__VA_ARGS__)
#endif

#ifndef C10_CUDA_CHECK
#define C10_CUDA_CHECK(EXPR) STD_CUDA_CHECK(EXPR)
#endif

#ifndef C10_CUDA_KERNEL_LAUNCH_CHECK
#define C10_CUDA_KERNEL_LAUNCH_CHECK() STD_CUDA_KERNEL_LAUNCH_CHECK()
#endif
