#include "tensorflow/core/tfrt/mlrt/interpreter/context.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/executable.h"
namespace mlrt {
namespace context_internal {
UserContextBase::~UserContextBase() = default;
}
void KernelRegistry::Register(absl::string_view name,
                              KernelImplementation kernel) {
  map_.emplace(name, kernel);
}
KernelImplementation KernelRegistry::Get(absl::string_view name) const {
  DCHECK(map_.contains(name)) << "Missing kernel in registry: " << name;
  return map_.at(name);
}
void KernelRegistry::Merge(const KernelRegistry& other) {
  map_.insert(other.map_.begin(), other.map_.end());
}
LoadedExecutable::LoadedExecutable(bc::Executable executable,
                                   const KernelRegistry& kernel_registry)
    : executable_(executable) {
  kernels_.reserve(executable_.kernel_names().size());
  for (auto kernel_name : executable_.kernel_names()) {
    kernels_.push_back(kernel_registry.Get(kernel_name));
  }
  functions_.reserve(executable_.functions().size());
  for (auto function : executable_.functions()) {
    functions_[function.name().Get()] = function;
  }
}
}  