#include "tensorstore/internal/json_binding/unit.h"
#include <cmath>
#include <string>
#include "absl/status/status.h"
#include <nlohmann/json_fwd.hpp>
#include "tensorstore/internal/json/value_as.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_tuple.h"
#include "tensorstore/util/unit.h"
namespace tensorstore {
namespace internal_json_binding {
TENSORSTORE_DEFINE_JSON_BINDER(
    UnitJsonBinder,
    [](auto is_loading, const auto& options, auto* obj,
       ::nlohmann::json* j) -> absl::Status {
      if constexpr (is_loading) {
        if (auto* s = j->get_ptr<const std::string*>()) {
          *obj = Unit(*s);
          return absl::OkStatus();
        } else if (j->is_number()) {
          *obj = Unit(j->get<double>(), "");
          return absl::OkStatus();
        }
      }
      return HeterogeneousArray(
          Projection<&Unit::multiplier>(Validate([](const auto& options,
                                                    auto* num) {
            if (*num > 0 && std::isfinite(*num)) return absl::OkStatus();
            return internal_json::ExpectedError(*num, "finite positive number");
          })),
          Projection<&Unit::base_unit>())(is_loading, options, obj, j);
    });
TENSORSTORE_DEFINE_JSON_BINDER(
    StringOnlyUnitJsonBinder,
    Compose<std::string>([](auto is_loading, const auto& options, auto* obj,
                            std::string* j) -> absl::Status {
      if constexpr (is_loading) {
        *obj = Unit(*j);
      } else {
        *j = obj->to_string();
      }
      return absl::OkStatus();
    }));
}  
}  