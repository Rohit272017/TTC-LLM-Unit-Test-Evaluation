#include "eval/public/structs/protobuf_descriptor_type_provider.h"
#include <memory>
#include <utility>
#include "google/protobuf/descriptor.h"
#include "absl/synchronization/mutex.h"
#include "eval/public/structs/proto_message_type_adapter.h"
namespace google::api::expr::runtime {
absl::optional<LegacyTypeAdapter> ProtobufDescriptorProvider::ProvideLegacyType(
    absl::string_view name) const {
  const ProtoMessageTypeAdapter* result = GetTypeAdapter(name);
  if (result == nullptr) {
    return absl::nullopt;
  }
  return LegacyTypeAdapter(result, result);
}
absl::optional<const LegacyTypeInfoApis*>
ProtobufDescriptorProvider::ProvideLegacyTypeInfo(
    absl::string_view name) const {
  const ProtoMessageTypeAdapter* result = GetTypeAdapter(name);
  if (result == nullptr) {
    return absl::nullopt;
  }
  return result;
}
std::unique_ptr<ProtoMessageTypeAdapter>
ProtobufDescriptorProvider::CreateTypeAdapter(absl::string_view name) const {
  const google::protobuf::Descriptor* descriptor =
      descriptor_pool_->FindMessageTypeByName(name);
  if (descriptor == nullptr) {
    return nullptr;
  }
  return std::make_unique<ProtoMessageTypeAdapter>(descriptor,
                                                   message_factory_);
}
const ProtoMessageTypeAdapter* ProtobufDescriptorProvider::GetTypeAdapter(
    absl::string_view name) const {
  absl::MutexLock lock(&mu_);
  auto it = type_cache_.find(name);
  if (it != type_cache_.end()) {
    return it->second.get();
  }
  auto type_provider = CreateTypeAdapter(name);
  const ProtoMessageTypeAdapter* result = type_provider.get();
  type_cache_[name] = std::move(type_provider);
  return result;
}
}  