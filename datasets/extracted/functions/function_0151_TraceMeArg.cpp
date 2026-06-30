#ifndef TENSORFLOW_TSL_PROFILER_LIB_TRACEME_ENCODE_H_
#define TENSORFLOW_TSL_PROFILER_LIB_TRACEME_ENCODE_H_
#include <string.h>
#include <initializer_list>
#include <string>
#include "absl/base/attributes.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/macros.h"
namespace tsl {
namespace profiler {
struct TraceMeArg {
  TraceMeArg(absl::string_view k,
             const absl::AlphaNum& v ABSL_ATTRIBUTE_LIFETIME_BOUND)
      : key(k), value(v.Piece()) {}
  TraceMeArg(const TraceMeArg&) = delete;
  void operator=(const TraceMeArg&) = delete;
  absl::string_view key;
  absl::string_view value;
};
namespace traceme_internal {
TF_ATTRIBUTE_ALWAYS_INLINE inline char* Append(char* out,
                                               absl::string_view str) {
  DCHECK(!absl::StrContains(str, '#'))
      << "'#' is not a valid character in TraceMeEncode";
  const size_t str_size = str.size();
  if (TF_PREDICT_TRUE(str_size > 0)) {
    memcpy(out, str.data(), str_size);
    out += str_size;
  }
  return out;
}
TF_ATTRIBUTE_ALWAYS_INLINE inline std::string AppendArgs(
    std::string name, std::initializer_list<TraceMeArg> args) {
  if (TF_PREDICT_TRUE(args.size() > 0)) {
    const auto old_size = name.size();
    auto new_size = old_size + args.size() * 2 + 1;
    for (const auto& arg : args) {
      new_size += arg.key.size() + arg.value.size();
    }
    name.resize(new_size);
    char* const begin = &name[0];
    char* out = begin + old_size;
    *out++ = '#';
    for (const auto& arg : args) {
      out = Append(out, arg.key);
      *out++ = '=';
      out = Append(out, arg.value);
      *out++ = ',';
    }
    *(out - 1) = '#';
    DCHECK_EQ(out, begin + new_size);
  }
  return name;
}
TF_ATTRIBUTE_ALWAYS_INLINE inline void AppendMetadata(
    std::string* name, absl::string_view new_metadata) {
  if (!TF_PREDICT_FALSE(new_metadata.empty())) {
    if (!name->empty() && name->back() == '#') {  
      name->back() = ',';
      if (TF_PREDICT_TRUE(new_metadata.front() == '#')) {
        new_metadata.remove_prefix(1);
      }
    }
    name->append(new_metadata.data(), new_metadata.size());
  }
}
}  
TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeEncode(
    std::string name, std::initializer_list<TraceMeArg> args) {
  return traceme_internal::AppendArgs(std::move(name), args);
}
TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeEncode(
    absl::string_view name, std::initializer_list<TraceMeArg> args) {
  return traceme_internal::AppendArgs(std::string(name), args);
}
TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeEncode(
    const char* name, std::initializer_list<TraceMeArg> args) {
  return traceme_internal::AppendArgs(std::string(name), args);
}
TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeEncode(
    std::initializer_list<TraceMeArg> args) {
  return traceme_internal::AppendArgs(std::string(), args);
}
TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeOp(
    absl::string_view op_name, absl::string_view op_type) {
  return absl::StrCat(op_name, ":", op_type);
}
TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeOp(const char* op_name,
                                                        const char* op_type) {
  return absl::StrCat(op_name, ":", op_type);
}
TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeOp(
    std::string&& op_name, absl::string_view op_type) {
  absl::StrAppend(&op_name, ":", op_type);
  return op_name;
}
TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeOpOverride(
    absl::string_view op_name, absl::string_view op_type) {
  return absl::StrCat("#tf_op=", op_name, ":", op_type, "#");
}
TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeOpOverride(
    const char* op_name, const char* op_type) {
  return absl::StrCat("#tf_op=", op_name, ":", op_type, "#");
}
}  
}  
#endif  