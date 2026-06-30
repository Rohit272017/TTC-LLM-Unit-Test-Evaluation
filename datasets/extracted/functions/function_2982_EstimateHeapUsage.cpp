#ifndef TENSORSTORE_INTERNAL_ESTIMATE_HEAP_USAGE_ESTIMATE_HEAP_USAGE_H_
#define TENSORSTORE_INTERNAL_ESTIMATE_HEAP_USAGE_ESTIMATE_HEAP_USAGE_H_
#include <stddef.h>
#include <memory>
#include <string>
#include <type_traits>
#include "absl/strings/cord.h"
#include "tensorstore/util/apply_members/apply_members.h"
namespace tensorstore {
namespace internal {
template <typename T, typename SFINAE = void>
struct HeapUsageEstimator;
template <typename T, typename SFINAE = void>
constexpr inline bool MayUseHeapMemory = true;
template <typename T>
constexpr inline bool MayUseHeapMemory<
    T, std::enable_if_t<
           !std::is_trivially_destructible_v<T>,
           std::void_t<decltype(&HeapUsageEstimator<T>::MayUseHeapMemory)>>> =
    HeapUsageEstimator<T>::MayUseHeapMemory();
template <typename T>
constexpr inline bool
    MayUseHeapMemory<T, std::enable_if_t<std::is_trivially_destructible_v<T>>> =
        false;
template <typename T>
size_t EstimateHeapUsage(const T& x, size_t max_depth = -1) {
  if constexpr (!MayUseHeapMemory<T>) {
    return 0;
  } else {
    return HeapUsageEstimator<T>::EstimateHeapUsage(x, max_depth);
  }
}
struct MayAnyUseHeapMemory {
  template <typename... T>
  constexpr auto operator()(const T&... arg) const {
    return std::integral_constant<bool, (MayUseHeapMemory<T> || ...)>{};
  }
};
template <typename T>
struct HeapUsageEstimator<T, std::enable_if_t<SupportsApplyMembers<T>>> {
  static size_t EstimateHeapUsage(const T& v, size_t max_depth) {
    return ApplyMembers<T>::Apply(v, [&](auto&&... x) {
      return (internal::EstimateHeapUsage(x, max_depth) + ... +
              static_cast<size_t>(0));
    });
  }
  static constexpr bool MayUseHeapMemory() {
    return decltype(ApplyMembers<T>::Apply(std::declval<const T&>(),
                                           MayAnyUseHeapMemory{}))::value;
  }
};
template <>
struct HeapUsageEstimator<std::string> {
  static size_t EstimateHeapUsage(const std::string& x, size_t max_depth) {
    return x.capacity();
  }
};
template <>
struct HeapUsageEstimator<absl::Cord> {
  static size_t EstimateHeapUsage(const absl::Cord& x, size_t max_depth) {
    return x.size();
  }
};
template <typename T>
struct PointerHeapUsageEstimator {
  static size_t EstimateHeapUsage(const T& x, size_t max_depth) {
    if (!x) return 0;
    size_t total = sizeof(*x);
    if (max_depth > 0) {
      total += internal::EstimateHeapUsage(*x);
    }
    return total;
  }
};
template <typename T>
struct HeapUsageEstimator<std::shared_ptr<T>>
    : public PointerHeapUsageEstimator<std::shared_ptr<T>> {};
template <typename T>
struct HeapUsageEstimator<std::unique_ptr<T>>
    : public PointerHeapUsageEstimator<std::unique_ptr<T>> {};
template <typename T, typename R>
class IntrusivePtr;
template <typename T, typename R>
struct HeapUsageEstimator<IntrusivePtr<T, R>>
    : public PointerHeapUsageEstimator<IntrusivePtr<T, R>> {};
}  
}  
#endif  