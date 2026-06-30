#include "tensorflow/core/common_runtime/request_cost_accessor_registry.h"
#include <string>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/logging.h"
namespace tensorflow {
namespace {
using RegistrationMap =
    absl::flat_hash_map<std::string, RequestCostAccessorRegistry::Creator>;
RegistrationMap* GetRegistrationMap() {
  static RegistrationMap* registered_request_cost_accessors =
      new RegistrationMap;
  return registered_request_cost_accessors;
}
}  
std::unique_ptr<RequestCostAccessor>
RequestCostAccessorRegistry::CreateByNameOrNull(absl::string_view name) {
  const auto it = GetRegistrationMap()->find(name);
  if (it == GetRegistrationMap()->end()) return nullptr;
  return std::unique_ptr<RequestCostAccessor>(it->second());
}
void RequestCostAccessorRegistry::RegisterRequestCostAccessor(
    absl::string_view name, Creator creator) {
  const auto it = GetRegistrationMap()->find(name);
  CHECK(it == GetRegistrationMap()->end())  
      << "RequestCostAccessor " << name << " is registered twice.";
  GetRegistrationMap()->emplace(name, std::move(creator));
}
}  