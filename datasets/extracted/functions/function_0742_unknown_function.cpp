#ifndef TENSORFLOW_CORE_PROFILER_LIB_SCOPED_ANNOTATION_H_
#define TENSORFLOW_CORE_PROFILER_LIB_SCOPED_ANNOTATION_H_
#include <stddef.h>
#include <atomic>
#include <string>
#include <string_view>
#include <utility>
#include "absl/base/macros.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "xla/tsl/profiler/backends/cpu/annotation_stack.h"
#endif
#ifndef ABSL_DEPRECATE_AND_INLINE
#define ABSL_DEPRECATE_AND_INLINE()
#endif
namespace tensorflow {
namespace profiler {
using ScopedAnnotation ABSL_DEPRECATE_AND_INLINE() =
    tsl::profiler::ScopedAnnotation;  
}  
}  
#endif  