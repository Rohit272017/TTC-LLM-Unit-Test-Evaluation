#ifndef TENSORSTORE_INTERNAL_JSON_BINDING_OPTIONAL_OBJECT_H_
#define TENSORSTORE_INTERNAL_JSON_BINDING_OPTIONAL_OBJECT_H_
#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json/value_as.h"
#include "tensorstore/internal/json_binding/json_binding.h"
namespace tensorstore {
namespace internal_json_binding {
template <typename ObjectBinder>
constexpr auto OptionalObject(ObjectBinder binder) {
  return [binder = std::move(binder)](auto is_loading, const auto& options,
                                      auto* obj, auto* j) -> absl::Status {
    ::nlohmann::json::object_t json_obj;
    if constexpr (is_loading) {
      if (!j->is_discarded()) {
        if (auto* ptr = j->template get_ptr<::nlohmann::json::object_t*>();
            ptr) {
          json_obj = std::move(*ptr);
        } else {
          return internal_json::ExpectedError(*j, "object");
        }
      }
    }
    if (auto status = internal_json_binding::Object(binder)(is_loading, options,
                                                            obj, &json_obj);
        !status.ok()) {
      return status;
    }
    if constexpr (!is_loading) {
      if (!json_obj.empty()) {
        *j = std::move(json_obj);
      } else {
        *j = ::nlohmann::json::value_t::discarded;
      }
    }
    return absl::OkStatus();
  };
}
}  
}  
#endif  