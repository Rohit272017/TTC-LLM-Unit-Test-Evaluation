#include <string>
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/proto/descriptor_pool_registry.h"
namespace tensorflow {
DescriptorPoolRegistry* DescriptorPoolRegistry::Global() {
  static DescriptorPoolRegistry* registry = new DescriptorPoolRegistry;
  return registry;
}
DescriptorPoolRegistry::DescriptorPoolFn* DescriptorPoolRegistry::Get(
    const string& source) {
  auto found = fns_.find(source);
  if (found == fns_.end()) return nullptr;
  return &found->second;
}
void DescriptorPoolRegistry::Register(
    const string& source,
    const DescriptorPoolRegistry::DescriptorPoolFn& pool_fn) {
  auto existing = Get(source);
  CHECK_EQ(existing, nullptr)
      << "descriptor pool for source: " << source << " already registered";
  fns_.insert(std::pair<const string&, DescriptorPoolFn>(source, pool_fn));
}
}  