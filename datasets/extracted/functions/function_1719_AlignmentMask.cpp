#ifndef THIRD_PARTY_CEL_CPP_INTERNAL_ALIGN_H_
#define THIRD_PARTY_CEL_CPP_INTERNAL_ALIGN_H_
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "absl/base/casts.h"
#include "absl/base/config.h"
#include "absl/base/macros.h"
#include "absl/numeric/bits.h"
namespace cel::internal {
template <typename T>
constexpr std::enable_if_t<
    std::conjunction_v<std::is_integral<T>, std::is_unsigned<T>>, T>
AlignmentMask(T alignment) {
  ABSL_ASSERT(absl::has_single_bit(alignment));
  return alignment - T{1};
}
template <typename T>
std::enable_if_t<std::conjunction_v<std::is_integral<T>, std::is_unsigned<T>>,
                 T>
AlignDown(T x, size_t alignment) {
  ABSL_ASSERT(absl::has_single_bit(alignment));
#if ABSL_HAVE_BUILTIN(__builtin_align_up)
  return __builtin_align_down(x, alignment);
#else
  using C = std::common_type_t<T, size_t>;
  return static_cast<T>(static_cast<C>(x) &
                        ~AlignmentMask(static_cast<C>(alignment)));
#endif
}
template <typename T>
std::enable_if_t<std::is_pointer_v<T>, T> AlignDown(T x, size_t alignment) {
  return absl::bit_cast<T>(AlignDown(absl::bit_cast<uintptr_t>(x), alignment));
}
template <typename T>
std::enable_if_t<std::conjunction_v<std::is_integral<T>, std::is_unsigned<T>>,
                 T>
AlignUp(T x, size_t alignment) {
  ABSL_ASSERT(absl::has_single_bit(alignment));
#if ABSL_HAVE_BUILTIN(__builtin_align_up)
  return __builtin_align_up(x, alignment);
#else
  using C = std::common_type_t<T, size_t>;
  return static_cast<T>(AlignDown(
      static_cast<C>(x) + AlignmentMask(static_cast<C>(alignment)), alignment));
#endif
}
template <typename T>
std::enable_if_t<std::is_pointer_v<T>, T> AlignUp(T x, size_t alignment) {
  return absl::bit_cast<T>(AlignUp(absl::bit_cast<uintptr_t>(x), alignment));
}
template <typename T>
constexpr std::enable_if_t<
    std::conjunction_v<std::is_integral<T>, std::is_unsigned<T>>, bool>
IsAligned(T x, size_t alignment) {
  ABSL_ASSERT(absl::has_single_bit(alignment));
#if ABSL_HAVE_BUILTIN(__builtin_is_aligned)
  return __builtin_is_aligned(x, alignment);
#else
  using C = std::common_type_t<T, size_t>;
  return (static_cast<C>(x) & AlignmentMask(static_cast<C>(alignment))) == C{0};
#endif
}
template <typename T>
std::enable_if_t<std::is_pointer_v<T>, bool> IsAligned(T x, size_t alignment) {
  return IsAligned(absl::bit_cast<uintptr_t>(x), alignment);
}
}  
#endif  