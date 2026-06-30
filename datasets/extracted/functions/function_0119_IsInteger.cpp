#include "xla/tsl/profiler/utils/tf_op_utils.h"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "tsl/platform/regexp.h"
namespace tsl {
namespace profiler {
namespace {
const absl::string_view kIterator = "Iterator";
const absl::string_view kSeparator = "::";
constexpr char kNameScopeSeparator = '/';
constexpr char kOpNameSuffixSeparator = '_';
bool IsInteger(absl::string_view str) {
  int64_t unused;
  return absl::SimpleAtoi(str, &unused);
}
absl::string_view DeriveOpType(absl::string_view full_op_name) {
  std::vector<absl::string_view> name_scopes_and_op_name =
      absl::StrSplit(full_op_name, kNameScopeSeparator);
  absl::string_view op_name = name_scopes_and_op_name.back();
  std::vector<absl::string_view> op_type_and_maybe_suffix =
      absl::StrSplit(op_name, kOpNameSuffixSeparator);
  absl::string_view maybe_suffix = op_type_and_maybe_suffix.back();
  absl::string_view op_type = op_name;
  if (IsInteger(maybe_suffix)) {
    op_type = op_name.substr(0, op_name.size() - maybe_suffix.size() - 1);
  }
  return op_type;
}
std::optional<TfOp> GetMemcpyOp(absl::string_view tf_op_fullname) {
  TfOp tf_op;
  tf_op.name = tf_op_fullname;
  if (absl::StartsWithIgnoreCase(tf_op_fullname, "MEMCPYHToD")) {
    tf_op.category = Category::kMemcpyHToD;
    tf_op.type = kMemcpyHToDOp;
    return tf_op;
  }
  if (absl::StartsWithIgnoreCase(tf_op_fullname, "MEMCPYDToH")) {
    tf_op.category = Category::kMemcpyDToH;
    tf_op.type = kMemcpyDToHOp;
    return tf_op;
  }
  if (absl::StartsWithIgnoreCase(tf_op_fullname, "MEMCPYDToD")) {
    tf_op.category = Category::kMemcpyDToD;
    tf_op.type = kMemcpyDToDOp;
    return tf_op;
  } else if (absl::StartsWithIgnoreCase(tf_op_fullname, "MEMCPYHToH")) {
    tf_op.category = Category::kMemcpyHToH;
    tf_op.type = kMemcpyHToHOp;
    return tf_op;
  }
  return std::nullopt;
}
}  
const absl::string_view kUnknownOp = "";  
const absl::string_view kDatasetOp = "Dataset";
const absl::string_view kMemcpyHToDOp = "MemcpyHToD";
const absl::string_view kMemcpyDToHOp = "MemcpyDToH";
const absl::string_view kMemcpyDToDOp = "MemcpyDToD";
const absl::string_view kMemcpyHToHOp = "MemcpyHToH";
bool IsTfOpName(absl::string_view op_name) {
  static const LazyRE2 kTfOpNameRegEx = {"[A-Za-z0-9.][A-Za-z0-9_.\\/>-]*"};
  return RE2::FullMatch(op_name, *kTfOpNameRegEx);
}
bool IsTfOpType(absl::string_view op_type) {
  static const LazyRE2 kTfOpTypeRegEx = {"[A-Z_][a-zA-Z0-9_]*"};
  return RE2::FullMatch(op_type, *kTfOpTypeRegEx);
}
bool IsJaxOpType(absl::string_view op_type) {
  static const LazyRE2 kJaxOpTypeRegEx = {"[a-z_][a-z0-9_]*(\\[.*\\])?"};
  return RE2::FullMatch(op_type, *kJaxOpTypeRegEx);
}
bool IsJaxOpNameAndType(absl::string_view op_name, absl::string_view op_type) {
  if (op_name.empty() || !IsJaxOpType(op_type)) return false;
  std::vector<absl::string_view> split_result =
      absl::StrSplit(op_name, kNameScopeSeparator);
  return absl::StrContains(split_result.back(), op_type);
}
TfOp ParseTfOpFullname(absl::string_view tf_op_fullname) {
  TfOp tf_op = {Category::kUnknown, tf_op_fullname, kUnknownOp};
  std::vector<absl::string_view> parts =
      absl::StrSplit(tf_op_fullname, absl::MaxSplits(':', 1));
  if (parts.size() != 2) {
    if (std::optional<TfOp> tfop = GetMemcpyOp(parts[0]); tfop.has_value()) {
      return *tfop;
    }
    return tf_op;
  }
  if (parts[0] == kIterator) {
    tf_op.category = Category::kTfData;
    tf_op.type = kDatasetOp;
    return tf_op;
  }
  if (IsTfOpName(parts[0]) && IsTfOpType(parts[1])) {
    tf_op.category = Category::kTensorFlow;
    tf_op.name = parts[0];
    tf_op.type = parts[1];
    return tf_op;
  }
  absl::string_view op_type =
      parts[1].empty() ? DeriveOpType(parts[0]) : parts[1];
  if (IsJaxOpType(op_type)) {
    tf_op.category = Category::kJax;
    tf_op.name = parts[0];
    tf_op.type = op_type.substr(0, op_type.find('['));
    return tf_op;
  }
  if (parts[1].empty()) {
    tf_op.category = Category::kTensorFlow;
    tf_op.name = parts[0];
    tf_op.type = op_type;
    return tf_op;
  }
  return tf_op;
}
std::vector<absl::string_view> ParseTfNameScopes(absl::string_view tf_op_name) {
  std::vector<absl::string_view> name_scopes =
      absl::StrSplit(tf_op_name, kNameScopeSeparator);
  if (!name_scopes.empty()) name_scopes.pop_back();
  return name_scopes;
}
std::vector<absl::string_view> ParseTfNameScopes(const TfOp& tf_op) {
  return ParseTfNameScopes(tf_op.name);
}
std::string TfOpEventName(const TfOp& tf_op) {
  std::string event_name;
  if (tf_op.category == Category::kUnknown) {
    event_name = std::string(absl::StripTrailingAsciiWhitespace(tf_op.name));
  } else if (tf_op.category == Category::kTfData) {
    event_name = DatasetOpEventName(tf_op.name);
  } else {
    event_name = std::string(tf_op.type);
  }
  return event_name;
}
std::string TfOpEventName(absl::string_view tf_op_fullname) {
  return TfOpEventName(ParseTfOpFullname(tf_op_fullname));
}
std::string DatasetOpEventName(absl::string_view full_name) {
  std::vector<absl::string_view> split_result =
      absl::StrSplit(full_name, kSeparator);
  return absl::StrCat(kIterator, kSeparator, split_result.back());
}
std::string IteratorName(absl::string_view full_name) {
  std::vector<absl::string_view> split_result =
      absl::StrSplit(full_name, kSeparator);
  return std::string(split_result.back());
}
std::vector<absl::string_view> ParseTensorShapes(
    absl::string_view tensor_shapes) {
  absl::ConsumePrefix(&tensor_shapes, "(");
  absl::ConsumeSuffix(&tensor_shapes, ")");
  return absl::StrSplit(tensor_shapes, ';');
}
}  
}  