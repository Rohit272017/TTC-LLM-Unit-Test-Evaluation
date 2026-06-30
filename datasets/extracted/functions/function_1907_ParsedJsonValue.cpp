#include "common/values/parsed_json_value.h"
#include <string>
#include <utility>
#include "absl/base/attributes.h"
#include "absl/functional/overload.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/variant.h"
#include "common/allocator.h"
#include "common/memory.h"
#include "common/value.h"
#include "internal/well_known_types.h"
#include "google/protobuf/message.h"
namespace cel::common_internal {
namespace {
using ::cel::well_known_types::AsVariant;
using ::cel::well_known_types::GetValueReflectionOrDie;
}  
Value ParsedJsonValue(Allocator<> allocator,
                      Borrowed<const google::protobuf::Message> message) {
  const auto reflection = GetValueReflectionOrDie(message->GetDescriptor());
  const auto kind_case = reflection.GetKindCase(*message);
  switch (kind_case) {
    case google::protobuf::Value::KIND_NOT_SET:
      ABSL_FALLTHROUGH_INTENDED;
    case google::protobuf::Value::kNullValue:
      return NullValue();
    case google::protobuf::Value::kBoolValue:
      return BoolValue(reflection.GetBoolValue(*message));
    case google::protobuf::Value::kNumberValue:
      return DoubleValue(reflection.GetNumberValue(*message));
    case google::protobuf::Value::kStringValue: {
      std::string scratch;
      return absl::visit(
          absl::Overload(
              [&](absl::string_view string) -> StringValue {
                if (string.empty()) {
                  return StringValue();
                }
                if (string.data() == scratch.data() &&
                    string.size() == scratch.size()) {
                  return StringValue(allocator, std::move(scratch));
                } else {
                  return StringValue(message, string);
                }
              },
              [&](absl::Cord&& cord) -> StringValue {
                if (cord.empty()) {
                  return StringValue();
                }
                return StringValue(std::move(cord));
              }),
          AsVariant(reflection.GetStringValue(*message, scratch)));
    }
    case google::protobuf::Value::kListValue:
      return ParsedJsonListValue(Owned<const google::protobuf::Message>(
          Owner(message), &reflection.GetListValue(*message)));
    case google::protobuf::Value::kStructValue:
      return ParsedJsonMapValue(Owned<const google::protobuf::Message>(
          Owner(message), &reflection.GetStructValue(*message)));
    default:
      return ErrorValue(absl::InvalidArgumentError(
          absl::StrCat("unexpected value kind case: ", kind_case)));
  }
}
}  