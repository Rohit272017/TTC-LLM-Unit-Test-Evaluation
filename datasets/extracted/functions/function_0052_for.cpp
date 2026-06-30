#include "runtime/function_registry.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "base/function.h"
#include "base/function_descriptor.h"
#include "base/kind.h"
#include "runtime/activation_interface.h"
#include "runtime/function_overload_reference.h"
#include "runtime/function_provider.h"
namespace cel {
namespace {
class ActivationFunctionProviderImpl
    : public cel::runtime_internal::FunctionProvider {
 public:
  ActivationFunctionProviderImpl() = default;
  absl::StatusOr<absl::optional<cel::FunctionOverloadReference>> GetFunction(
      const cel::FunctionDescriptor& descriptor,
      const cel::ActivationInterface& activation) const override {
    std::vector<cel::FunctionOverloadReference> overloads =
        activation.FindFunctionOverloads(descriptor.name());
    absl::optional<cel::FunctionOverloadReference> matching_overload =
        absl::nullopt;
    for (const auto& overload : overloads) {
      if (overload.descriptor.ShapeMatches(descriptor)) {
        if (matching_overload.has_value()) {
          return absl::Status(absl::StatusCode::kInvalidArgument,
                              "Couldn't resolve function.");
        }
        matching_overload.emplace(overload);
      }
    }
    return matching_overload;
  }
};
std::unique_ptr<cel::runtime_internal::FunctionProvider>
CreateActivationFunctionProvider() {
  return std::make_unique<ActivationFunctionProviderImpl>();
}
}  
absl::Status FunctionRegistry::Register(
    const cel::FunctionDescriptor& descriptor,
    std::unique_ptr<cel::Function> implementation) {
  if (DescriptorRegistered(descriptor)) {
    return absl::Status(
        absl::StatusCode::kAlreadyExists,
        "CelFunction with specified parameters already registered");
  }
  if (!ValidateNonStrictOverload(descriptor)) {
    return absl::Status(absl::StatusCode::kAlreadyExists,
                        "Only one overload is allowed for non-strict function");
  }
  auto& overloads = functions_[descriptor.name()];
  overloads.static_overloads.push_back(
      StaticFunctionEntry(descriptor, std::move(implementation)));
  return absl::OkStatus();
}
absl::Status FunctionRegistry::RegisterLazyFunction(
    const cel::FunctionDescriptor& descriptor) {
  if (DescriptorRegistered(descriptor)) {
    return absl::Status(
        absl::StatusCode::kAlreadyExists,
        "CelFunction with specified parameters already registered");
  }
  if (!ValidateNonStrictOverload(descriptor)) {
    return absl::Status(absl::StatusCode::kAlreadyExists,
                        "Only one overload is allowed for non-strict function");
  }
  auto& overloads = functions_[descriptor.name()];
  overloads.lazy_overloads.push_back(
      LazyFunctionEntry(descriptor, CreateActivationFunctionProvider()));
  return absl::OkStatus();
}
std::vector<cel::FunctionOverloadReference>
FunctionRegistry::FindStaticOverloads(absl::string_view name,
                                      bool receiver_style,
                                      absl::Span<const cel::Kind> types) const {
  std::vector<cel::FunctionOverloadReference> matched_funcs;
  auto overloads = functions_.find(name);
  if (overloads == functions_.end()) {
    return matched_funcs;
  }
  for (const auto& overload : overloads->second.static_overloads) {
    if (overload.descriptor->ShapeMatches(receiver_style, types)) {
      matched_funcs.push_back({*overload.descriptor, *overload.implementation});
    }
  }
  return matched_funcs;
}
std::vector<FunctionRegistry::LazyOverload> FunctionRegistry::FindLazyOverloads(
    absl::string_view name, bool receiver_style,
    absl::Span<const cel::Kind> types) const {
  std::vector<FunctionRegistry::LazyOverload> matched_funcs;
  auto overloads = functions_.find(name);
  if (overloads == functions_.end()) {
    return matched_funcs;
  }
  for (const auto& entry : overloads->second.lazy_overloads) {
    if (entry.descriptor->ShapeMatches(receiver_style, types)) {
      matched_funcs.push_back({*entry.descriptor, *entry.function_provider});
    }
  }
  return matched_funcs;
}
absl::node_hash_map<std::string, std::vector<const cel::FunctionDescriptor*>>
FunctionRegistry::ListFunctions() const {
  absl::node_hash_map<std::string, std::vector<const cel::FunctionDescriptor*>>
      descriptor_map;
  for (const auto& entry : functions_) {
    std::vector<const cel::FunctionDescriptor*> descriptors;
    const RegistryEntry& function_entry = entry.second;
    descriptors.reserve(function_entry.static_overloads.size() +
                        function_entry.lazy_overloads.size());
    for (const auto& entry : function_entry.static_overloads) {
      descriptors.push_back(entry.descriptor.get());
    }
    for (const auto& entry : function_entry.lazy_overloads) {
      descriptors.push_back(entry.descriptor.get());
    }
    descriptor_map[entry.first] = std::move(descriptors);
  }
  return descriptor_map;
}
bool FunctionRegistry::DescriptorRegistered(
    const cel::FunctionDescriptor& descriptor) const {
  return !(FindStaticOverloads(descriptor.name(), descriptor.receiver_style(),
                               descriptor.types())
               .empty()) ||
         !(FindLazyOverloads(descriptor.name(), descriptor.receiver_style(),
                             descriptor.types())
               .empty());
}
bool FunctionRegistry::ValidateNonStrictOverload(
    const cel::FunctionDescriptor& descriptor) const {
  auto overloads = functions_.find(descriptor.name());
  if (overloads == functions_.end()) {
    return true;
  }
  const RegistryEntry& entry = overloads->second;
  if (!descriptor.is_strict()) {
    return false;
  }
  return (entry.static_overloads.empty() ||
          entry.static_overloads[0].descriptor->is_strict()) &&
         (entry.lazy_overloads.empty() ||
          entry.lazy_overloads[0].descriptor->is_strict());
}
}  