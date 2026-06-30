#ifndef TENSORSTORE_INTERNAL_ESTIMATE_HEAP_USAGE_STD_OPTIONAL_H_
#define TENSORSTORE_INTERNAL_ESTIMATE_HEAP_USAGE_STD_OPTIONAL_H_
#include <optional>
#include "tensorstore/internal/estimate_heap_usage/estimate_heap_usage.h"
namespace tensorstore {
namespace internal {
template <typename T>
struct HeapUsageEstimator<std::optional<T>> {
  static size_t EstimateHeapUsage(const std::optional<T>& v, size_t max_depth) {
    if (!v) return 0;
    return internal::EstimateHeapUsage(*v, max_depth);
  }
  static constexpr bool MayUseHeapMemory() {
    return internal::MayUseHeapMemory<T>;
  }
};
}  
}  
#endif  