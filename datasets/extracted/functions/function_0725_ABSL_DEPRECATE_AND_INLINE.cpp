#ifndef TENSORFLOW_CORE_PROFILER_LIB_TRACEME_ENCODE_H_
#define TENSORFLOW_CORE_PROFILER_LIB_TRACEME_ENCODE_H_
#include <string.h>
#include <initializer_list>
#include <string>
#include <utility>
#include "absl/base/macros.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tsl/profiler/lib/traceme_encode.h"
#ifndef ABSL_DEPRECATE_AND_INLINE
#define ABSL_DEPRECATE_AND_INLINE()
#endif
namespace tensorflow {
namespace profiler {
using TraceMeArg ABSL_DEPRECATE_AND_INLINE() =
    tsl::profiler::TraceMeArg;  
ABSL_DEPRECATE_AND_INLINE()
inline std::string TraceMeEncode(
    std::string name, std::initializer_list<tsl::profiler::TraceMeArg> args) {
  return tsl::profiler::TraceMeEncode(std::move(name), args);
}
ABSL_DEPRECATE_AND_INLINE()
inline std::string TraceMeEncode(
    absl::string_view name,
    std::initializer_list<tsl::profiler::TraceMeArg> args) {
  return tsl::profiler::TraceMeEncode(name, args);
}
ABSL_DEPRECATE_AND_INLINE()
inline std::string TraceMeEncode(
    const char* name, std::initializer_list<tsl::profiler::TraceMeArg> args) {
  return tsl::profiler::TraceMeEncode(name, args);
}
ABSL_DEPRECATE_AND_INLINE()
inline std::string TraceMeEncode(
    std::initializer_list<tsl::profiler::TraceMeArg> args) {
  return tsl::profiler::TraceMeEncode(args);
}
ABSL_DEPRECATE_AND_INLINE()
inline std::string TraceMeOp(absl::string_view op_name,
                             absl::string_view op_type) {
  return tsl::profiler::TraceMeOp(op_name, op_type);
}
ABSL_DEPRECATE_AND_INLINE()
inline std::string TraceMeOp(const char* op_name, const char* op_type) {
  return tsl::profiler::TraceMeOp(op_name, op_type);
}
ABSL_DEPRECATE_AND_INLINE()
inline std::string TraceMeOp(std::string&& op_name, absl::string_view op_type) {
  return tsl::profiler::TraceMeOp(op_name, op_type);
}
ABSL_DEPRECATE_AND_INLINE()
inline std::string TraceMeOpOverride(absl::string_view op_name,
                                     absl::string_view op_type) {
  return tsl::profiler::TraceMeOpOverride(op_name, op_type);
}
ABSL_DEPRECATE_AND_INLINE()
inline std::string TraceMeOpOverride(const char* op_name, const char* op_type) {
  return tsl::profiler::TraceMeOpOverride(op_name, op_type);
}
}  
}  
#endif  