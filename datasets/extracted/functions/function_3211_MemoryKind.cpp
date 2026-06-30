#include "xla/python/ifrt/memory.h"
#include <optional>
#include <string>
#include <utility>
#include "absl/base/thread_annotations.h"
#include "absl/container/node_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/python/ifrt/device.h"
namespace xla {
namespace ifrt {
namespace {
struct MemoryKindsSet {
  absl::Mutex mu;
  absl::node_hash_set<std::string> memory_kinds_set ABSL_GUARDED_BY(mu);
};
}  
MemoryKind::MemoryKind(std::optional<absl::string_view> memory_kind) {
  static auto* const global_set = new MemoryKindsSet();
  if (!memory_kind.has_value()) {
    return;
  }
  absl::MutexLock lock(&global_set->mu);
  auto it = global_set->memory_kinds_set.find(*memory_kind);
  if (it == global_set->memory_kinds_set.end()) {
    memory_kind_ =
        *global_set->memory_kinds_set.insert(std::string(*memory_kind)).first;
  } else {
    memory_kind_ = *it;
  }
}
std::string MemoryKind::ToString() const {
  if (memory_kind_.has_value()) {
    return std::string(*memory_kind_);
  }
  return "(default)";
}
MemoryKind CanonicalizeMemoryKind(MemoryKind memory_kind, Device* device) {
  if (memory_kind.memory_kind().has_value()) {
    return memory_kind;
  }
  auto default_memory = device->DefaultMemory();
  if (default_memory.ok()) {
    return (*default_memory)->Kind();
  }
  return MemoryKind();
}
char Memory::ID = 0;
}  
}  