#include "tensorstore/internal/json_binding/staleness_bound.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json/value_as.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
namespace tensorstore {
namespace internal {
TENSORSTORE_DEFINE_JSON_BINDER(
    StalenessBoundJsonBinder,
    [](auto is_loading, const auto& options, auto* obj,
       ::nlohmann::json* j) -> absl::Status {
      if constexpr (is_loading) {
        if (const auto* b = j->get_ptr<const bool*>()) {
          *obj = *b ? absl::InfiniteFuture() : absl::InfinitePast();
        } else if (j->is_number()) {
          const double t = static_cast<double>(*j);
          *obj = absl::UnixEpoch() + absl::Seconds(t);
        } else if (*j == "open") {
          obj->time = absl::InfiniteFuture();
          obj->bounded_by_open_time = true;
        } else {
          return internal_json::ExpectedError(*j,
                                              "boolean, number, or \"open\"");
        }
      } else {
        if (obj->bounded_by_open_time) {
          *j = "open";
        } else {
          const absl::Time& t = obj->time;
          if (t == absl::InfiniteFuture()) {
            *j = true;
          } else if (t == absl::InfinitePast()) {
            *j = false;
          } else {
            *j = absl::ToDoubleSeconds(t - absl::UnixEpoch());
          }
        }
      }
      return absl::OkStatus();
    })
}  
}  