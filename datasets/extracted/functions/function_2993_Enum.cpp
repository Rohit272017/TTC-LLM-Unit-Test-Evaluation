#ifndef TENSORSTORE_INTERNAL_JSON_BINDING_ENUM_H_
#define TENSORSTORE_INTERNAL_JSON_BINDING_ENUM_H_
#include <stddef.h>
#include <string>
#include <utility>
#include <variant>
#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include <nlohmann/json_fwd.hpp>
#include "tensorstore/internal/json/same.h"
#include "tensorstore/internal/json/value_as.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/util/str_cat.h"
namespace tensorstore {
namespace internal_json_binding {
template <typename EnumValue, typename JsonValue, size_t N>
constexpr auto Enum(const std::pair<EnumValue, JsonValue> (&values)[N]) {
  return [=](auto is_loading, const auto& options, auto* obj,
             auto* j) -> absl::Status {
    for (const auto& p : values) {
      if constexpr (is_loading) {
        if (internal_json::JsonSame(p.second, *j)) {
          *obj = p.first;
          return absl::OkStatus();
        }
      } else {
        if (p.first == *obj) {
          *j = p.second;
          return absl::OkStatus();
        }
      }
    }
    if constexpr (is_loading) {
      return internal_json::ExpectedError(
          *j,
          tensorstore::StrCat(
              "one of ",
              absl::StrJoin(values, ", ", [](std::string* out, const auto& p) {
                *out += ::nlohmann::json(p.second).dump();
              })));
    } else {
      ABSL_UNREACHABLE();  
    }
  };
}
template <typename Binder, typename... Value, typename... JsonValue>
constexpr auto MapValue(Binder binder, std::pair<Value, JsonValue>... pairs) {
  constexpr size_t N = sizeof...(pairs);
  static_assert(N > 0);
  return [=](auto is_loading, const auto& options, auto* obj,
             auto* j) -> absl::Status {
    if constexpr (is_loading) {
      if (((internal_json::JsonSame(*j, pairs.second) &&
            (static_cast<void>(*obj = pairs.first), true)) ||
           ...))
        return absl::OkStatus();
    } else {
      if ((((*obj == pairs.first) &&
            (static_cast<void>(*j = pairs.second), true)) ||
           ...))
        return absl::OkStatus();
    }
    return binder(is_loading, options, obj, j);
  };
}
}  
}  
#endif  