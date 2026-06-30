#ifndef TENSORFLOW_TSL_PROFILER_LIB_SCOPED_ANNOTATION_H_
#define TENSORFLOW_TSL_PROFILER_LIB_SCOPED_ANNOTATION_H_
#include <stddef.h>
#include <atomic>
#include <string>
#include <string_view>
#include <utility>
#include "tsl/platform/macros.h"
#include "tsl/platform/platform.h"  
#include "tsl/profiler/lib/nvtx_utils.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "xla/tsl/profiler/backends/cpu/annotation_stack.h"
#endif
namespace tsl::profiler {
template <typename T>
void PushAnnotation(const T& generator) {
  if (auto domain = DefaultProfilerDomain();
      TF_PREDICT_FALSE(domain != nullptr)) {
    RangePush(domain, generator());
    return;
  }
#if !defined(IS_MOBILE_PLATFORM)
  if (TF_PREDICT_FALSE(AnnotationStack::IsEnabled())) {
    AnnotationStack::PushAnnotation(static_cast<std::string_view>(generator()));
  }
#endif
}
inline void PushAnnotation(const char* name) {
  PushAnnotation([&] { return name; });
}
inline void PushAnnotation(const std::string& name) {
  PushAnnotation([&] { return name; });
}
inline void PopAnnotation() {
  std::atomic_thread_fence(std::memory_order_acquire);
  if (auto domain = DefaultProfilerDomain();
      TF_PREDICT_FALSE(domain != nullptr)) {
    RangePop(domain);
    return;
  }
#if !defined(IS_MOBILE_PLATFORM)
  if (TF_PREDICT_FALSE(AnnotationStack::IsEnabled())) {
    AnnotationStack::PopAnnotation();
  }
#endif
}
class ScopedAnnotation {
 public:
  template <typename T>
  explicit ScopedAnnotation(T&& annotation) {
    PushAnnotation(std::forward<T>(annotation));
  }
  ~ScopedAnnotation() { PopAnnotation(); }
  static bool IsEnabled() {
#if !defined(IS_MOBILE_PLATFORM)
    return AnnotationStack::IsEnabled();
#else
    return false;
#endif
  }
 private:
  ScopedAnnotation(const ScopedAnnotation&) = delete;
  ScopedAnnotation& operator=(const ScopedAnnotation&) = delete;
};
}  
#endif  