#include "extensions/strings.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "common/casting.h"
#include "common/type.h"
#include "common/value.h"
#include "common/value_manager.h"
#include "eval/public/cel_function_registry.h"
#include "eval/public/cel_options.h"
#include "internal/status_macros.h"
#include "internal/utf8.h"
#include "runtime/function_adapter.h"
#include "runtime/function_registry.h"
#include "runtime/internal/errors.h"
#include "runtime/runtime_options.h"
namespace cel::extensions {
namespace {
struct AppendToStringVisitor {
  std::string& append_to;
  void operator()(absl::string_view string) const { append_to.append(string); }
  void operator()(const absl::Cord& cord) const {
    append_to.append(static_cast<std::string>(cord));
  }
};
absl::StatusOr<Value> Join2(ValueManager& value_manager, const ListValue& value,
                            const StringValue& separator) {
  std::string result;
  CEL_ASSIGN_OR_RETURN(auto iterator, value.NewIterator(value_manager));
  Value element;
  if (iterator->HasNext()) {
    CEL_RETURN_IF_ERROR(iterator->Next(value_manager, element));
    if (auto string_element = As<StringValue>(element); string_element) {
      string_element->NativeValue(AppendToStringVisitor{result});
    } else {
      return ErrorValue{
          runtime_internal::CreateNoMatchingOverloadError("join")};
    }
  }
  std::string separator_scratch;
  absl::string_view separator_view = separator.NativeString(separator_scratch);
  while (iterator->HasNext()) {
    result.append(separator_view);
    CEL_RETURN_IF_ERROR(iterator->Next(value_manager, element));
    if (auto string_element = As<StringValue>(element); string_element) {
      string_element->NativeValue(AppendToStringVisitor{result});
    } else {
      return ErrorValue{
          runtime_internal::CreateNoMatchingOverloadError("join")};
    }
  }
  result.shrink_to_fit();
  return value_manager.CreateUncheckedStringValue(std::move(result));
}
absl::StatusOr<Value> Join1(ValueManager& value_manager,
                            const ListValue& value) {
  return Join2(value_manager, value, StringValue{});
}
struct SplitWithEmptyDelimiter {
  ValueManager& value_manager;
  int64_t& limit;
  ListValueBuilder& builder;
  absl::StatusOr<Value> operator()(absl::string_view string) const {
    char32_t rune;
    size_t count;
    std::string buffer;
    buffer.reserve(4);
    while (!string.empty() && limit > 1) {
      std::tie(rune, count) = internal::Utf8Decode(string);
      buffer.clear();
      internal::Utf8Encode(buffer, rune);
      CEL_RETURN_IF_ERROR(builder.Add(
          value_manager.CreateUncheckedStringValue(absl::string_view(buffer))));
      --limit;
      string.remove_prefix(count);
    }
    if (!string.empty()) {
      CEL_RETURN_IF_ERROR(
          builder.Add(value_manager.CreateUncheckedStringValue(string)));
    }
    return std::move(builder).Build();
  }
  absl::StatusOr<Value> operator()(const absl::Cord& string) const {
    auto begin = string.char_begin();
    auto end = string.char_end();
    char32_t rune;
    size_t count;
    std::string buffer;
    while (begin != end && limit > 1) {
      std::tie(rune, count) = internal::Utf8Decode(begin);
      buffer.clear();
      internal::Utf8Encode(buffer, rune);
      CEL_RETURN_IF_ERROR(builder.Add(
          value_manager.CreateUncheckedStringValue(absl::string_view(buffer))));
      --limit;
      absl::Cord::Advance(&begin, count);
    }
    if (begin != end) {
      buffer.clear();
      while (begin != end) {
        auto chunk = absl::Cord::ChunkRemaining(begin);
        buffer.append(chunk);
        absl::Cord::Advance(&begin, chunk.size());
      }
      buffer.shrink_to_fit();
      CEL_RETURN_IF_ERROR(builder.Add(
          value_manager.CreateUncheckedStringValue(std::move(buffer))));
    }
    return std::move(builder).Build();
  }
};
absl::StatusOr<Value> Split3(ValueManager& value_manager,
                             const StringValue& string,
                             const StringValue& delimiter, int64_t limit) {
  if (limit == 0) {
    return ListValue{};
  }
  if (limit < 0) {
    limit = std::numeric_limits<int64_t>::max();
  }
  CEL_ASSIGN_OR_RETURN(auto builder,
                       value_manager.NewListValueBuilder(ListType{}));
  if (string.IsEmpty()) {
    builder->Reserve(1);
    CEL_RETURN_IF_ERROR(builder->Add(StringValue{}));
    return std::move(*builder).Build();
  }
  if (delimiter.IsEmpty()) {
    return string.NativeValue(
        SplitWithEmptyDelimiter{value_manager, limit, *builder});
  }
  std::string delimiter_scratch;
  absl::string_view delimiter_view = delimiter.NativeString(delimiter_scratch);
  std::string content_scratch;
  absl::string_view content_view = string.NativeString(content_scratch);
  while (limit > 1 && !content_view.empty()) {
    auto pos = content_view.find(delimiter_view);
    if (pos == absl::string_view::npos) {
      break;
    }
    CEL_RETURN_IF_ERROR(builder->Add(
        value_manager.CreateUncheckedStringValue(content_view.substr(0, pos))));
    --limit;
    content_view.remove_prefix(pos + delimiter_view.size());
    if (content_view.empty()) {
      CEL_RETURN_IF_ERROR(builder->Add(StringValue{}));
      return std::move(*builder).Build();
    }
  }
  CEL_RETURN_IF_ERROR(
      builder->Add(value_manager.CreateUncheckedStringValue(content_view)));
  return std::move(*builder).Build();
}
absl::StatusOr<Value> Split2(ValueManager& value_manager,
                             const StringValue& string,
                             const StringValue& delimiter) {
  return Split3(value_manager, string, delimiter, -1);
}
absl::StatusOr<Value> LowerAscii(ValueManager& value_manager,
                                 const StringValue& string) {
  std::string content = string.NativeString();
  absl::AsciiStrToLower(&content);
  return value_manager.CreateUncheckedStringValue(std::move(content));
}
}  
absl::Status RegisterStringsFunctions(FunctionRegistry& registry,
                                      const RuntimeOptions& options) {
  CEL_RETURN_IF_ERROR(registry.Register(
      UnaryFunctionAdapter<absl::StatusOr<Value>, ListValue>::CreateDescriptor(
          "join", true),
      UnaryFunctionAdapter<absl::StatusOr<Value>, ListValue>::WrapFunction(
          Join1)));
  CEL_RETURN_IF_ERROR(registry.Register(
      BinaryFunctionAdapter<absl::StatusOr<Value>, ListValue, StringValue>::
          CreateDescriptor("join", true),
      BinaryFunctionAdapter<absl::StatusOr<Value>, ListValue,
                            StringValue>::WrapFunction(Join2)));
  CEL_RETURN_IF_ERROR(registry.Register(
      BinaryFunctionAdapter<absl::StatusOr<Value>, StringValue, StringValue>::
          CreateDescriptor("split", true),
      BinaryFunctionAdapter<absl::StatusOr<Value>, StringValue,
                            StringValue>::WrapFunction(Split2)));
  CEL_RETURN_IF_ERROR(registry.Register(
      VariadicFunctionAdapter<
          absl::StatusOr<Value>, StringValue, StringValue,
          int64_t>::CreateDescriptor("split", true),
      VariadicFunctionAdapter<absl::StatusOr<Value>, StringValue, StringValue,
                              int64_t>::WrapFunction(Split3)));
  CEL_RETURN_IF_ERROR(registry.Register(
      UnaryFunctionAdapter<absl::StatusOr<Value>, StringValue>::
          CreateDescriptor("lowerAscii", true),
      UnaryFunctionAdapter<absl::StatusOr<Value>, StringValue>::WrapFunction(
          LowerAscii)));
  return absl::OkStatus();
}
absl::Status RegisterStringsFunctions(
    google::api::expr::runtime::CelFunctionRegistry* registry,
    const google::api::expr::runtime::InterpreterOptions& options) {
  return RegisterStringsFunctions(
      registry->InternalGetRegistry(),
      google::api::expr::runtime::ConvertToRuntimeOptions(options));
}
}  