#include "extensions/protobuf/memory_manager.h"
#include "absl/base/nullability.h"
#include "common/memory.h"
#include "google/protobuf/arena.h"
namespace cel {
namespace extensions {
MemoryManagerRef ProtoMemoryManager(google::protobuf::Arena* arena) {
  return arena != nullptr ? MemoryManagerRef::Pooling(arena)
                          : MemoryManagerRef::ReferenceCounting();
}
absl::Nullable<google::protobuf::Arena*> ProtoMemoryManagerArena(
    MemoryManager memory_manager) {
  return memory_manager.arena();
}
}  
}  