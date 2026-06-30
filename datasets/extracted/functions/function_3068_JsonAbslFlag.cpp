#ifndef TENSORSTORE_UTIL_JSON_ABSL_FLAG_H_
#define TENSORSTORE_UTIL_JSON_ABSL_FLAG_H_
#include <string>
#include <string_view>
#include <type_traits>
#include "absl/flags/marshalling.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/util/result.h"
namespace tensorstore {
template <typename T>
struct JsonAbslFlag {
  T value;
  JsonAbslFlag() = default;
  template <typename... U,
            typename = std::enable_if_t<std::is_constructible_v<T, U&&...>>>
  JsonAbslFlag(U&&... arg) : value(std::forward<U>(arg)...) {}
  friend std::string AbslUnparseFlag(const JsonAbslFlag& json_flag) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto j, internal_json_binding::ToJson(json_flag.value), "");
    if (j.is_discarded()) return {};
    return absl::UnparseFlag(j.dump());
  }
  friend bool AbslParseFlag(std::string_view in, JsonAbslFlag* out,
                            std::string* error) {
    if (in.empty()) {
      out->value = {};
      return true;
    }
    ::nlohmann::json j = ::nlohmann::json::parse(in, nullptr, false);
    if (j.is_discarded()) {
      *error = absl::StrFormat("Failed to parse JSON: '%s'", in);
      return false;
    }
    T new_value = {};
    absl::Status status = internal_json_binding::DefaultBinder<>(
        std::true_type{}, internal_json_binding::NoOptions{}, &new_value, &j);
    if (!status.ok()) {
      *error = absl::StrFormat("Failed to bind JSON: %s", status.message());
      return false;
    }
    out->value = std::move(new_value);
    return true;
  }
};
}  
#endif  