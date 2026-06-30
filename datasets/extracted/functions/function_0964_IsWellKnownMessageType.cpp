#include <array>
#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "common/type.h"
namespace cel {
bool IsWellKnownMessageType(absl::string_view name) {
  static constexpr absl::string_view kPrefix = "google.protobuf.";
  static constexpr std::array<absl::string_view, 15> kNames = {
      "Any",
      "BoolValue",
      "BytesValue",
      "DoubleValue",
      "Duration",
      "FloatValue",
      "Int32Value",
      "Int64Value",
      "ListValue",
      "StringValue",
      "Struct",
      "Timestamp",
      "UInt32Value",
      "UInt64Value",
      "Value",
  };
  if (!absl::ConsumePrefix(&name, kPrefix)) {
    return false;
  }
  return absl::c_binary_search(kNames, name);
}
}  