#include "xla/ffi/type_id_registry.h"
#include <atomic>
#include <cstdint>
#include <string>
#include <string_view>
#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/util.h"
namespace xla::ffi {
ABSL_CONST_INIT absl::Mutex type_registry_mutex(absl::kConstInit);
using ExternalTypeIdRegistry =
    absl::flat_hash_map<std::string, TypeIdRegistry::TypeId>;
static ExternalTypeIdRegistry& StaticExternalTypeIdRegistry() {
  static auto* registry = new ExternalTypeIdRegistry();
  return *registry;
}
TypeIdRegistry::TypeId TypeIdRegistry::GetNextTypeId() {
  static auto* counter = new std::atomic<int64_t>(1);
  return TypeId(counter->fetch_add(1));
}
absl::StatusOr<TypeIdRegistry::TypeId> TypeIdRegistry::RegisterExternalTypeId(
    std::string_view name) {
  absl::MutexLock lock(&type_registry_mutex);
  auto& registry = StaticExternalTypeIdRegistry();
  auto emplaced = registry.emplace(name, TypeId(0));
  if (!emplaced.second) {
    return Internal("Type id %d already registered for type name %s",
                    emplaced.first->second.value(), name);
  }
  return emplaced.first->second = GetNextTypeId();
}
}  