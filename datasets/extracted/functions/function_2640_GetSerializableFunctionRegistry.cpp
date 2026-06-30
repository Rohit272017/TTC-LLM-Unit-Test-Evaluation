#include "tensorstore/serialization/function.h"
#include <string_view>
#include <typeinfo>
#include "absl/base/no_destructor.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "tensorstore/internal/container/heterogeneous_container.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/util/garbage_collection/garbage_collection.h"
#include "tensorstore/util/str_cat.h"
namespace tensorstore {
namespace serialization {
namespace internal_serialization {
bool NonSerializableFunctionBase::Encode(EncodeSink& sink) const {
  sink.Fail(internal_serialization::NonSerializableError());
  return false;
}
void NonSerializableFunctionBase::GarbageCollectionVisit(
    garbage_collection::GarbageCollectionVisitor& visitor) const {
}
using SerializableFunctionRegistry =
    internal::HeterogeneousHashSet<const RegisteredSerializableFunction*,
                                   RegisteredSerializableFunction::Key,
                                   &RegisteredSerializableFunction::key>;
SerializableFunctionRegistry& GetSerializableFunctionRegistry() {
  static absl::NoDestructor<SerializableFunctionRegistry> registry;
  return *registry;
}
void RegisterSerializableFunction(const RegisteredSerializableFunction& r) {
  if (!GetSerializableFunctionRegistry().insert(&r).second) {
    ABSL_LOG(FATAL) << "Duplicate SerializableFunction registration: id="
                    << r.id << ", signature=" << r.signature->name();
  }
}
SerializableFunctionBase::~SerializableFunctionBase() = default;
bool DecodeSerializableFunction(DecodeSource& source,
                                SerializableFunctionBase::Ptr& value,
                                const std::type_info& signature) {
  std::string_view id;
  if (!serialization::Decode(source, id)) return false;
  auto& registry = GetSerializableFunctionRegistry();
  auto it = registry.find(RegisteredSerializableFunction::Key(signature, id));
  if (it == registry.end()) {
    source.Fail(absl::DataLossError(
        tensorstore::StrCat("SerializableFunction not registered: ", id)));
    return false;
  }
  return (*it)->decode(source, value);
}
}  
}  
}  