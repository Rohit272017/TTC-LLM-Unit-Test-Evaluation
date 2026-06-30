#include "xla/runtime/buffer_use.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
namespace xla {
BufferUse::ReadWriteSet::ReadWriteSet() = default;
void BufferUse::ReadWriteSet::Add(BufferUse use) {
  switch (use.access()) {
    case BufferUse::kRead:
      AddRead(use.slice());
      break;
    case BufferUse::kWrite:
      AddWrite(use.slice());
      break;
  }
}
void BufferUse::ReadWriteSet::AddRead(BufferAllocation::Slice slice) {
  read_.insert(slice);
}
void BufferUse::ReadWriteSet::AddWrite(BufferAllocation::Slice slice) {
  write_.insert(slice);
}
void BufferUse::ReadWriteSet::AddAll(absl::Span<const BufferUse> uses) {
  for (const auto& use : uses) Add(use);
}
bool BufferUse::ReadWriteSet::HasConflicts(const BufferUse& use) const {
  auto overlaps = [](const absl::flat_hash_set<BufferAllocation::Slice>& set,
                     const BufferUse& use) {
    return set.contains(use.slice()) ||
           absl::c_any_of(set, [&](const BufferAllocation::Slice& slice) {
             return slice.OverlapsWith(use.slice());
           });
  };
  return use.access() == MemoryAccess::kWrite
             ? overlaps(write_, use) || overlaps(read_, use)
             : overlaps(write_, use);
}
bool BufferUse::ReadWriteSet::HasConflicts(const ReadWriteSet& other) {
  return absl::c_any_of(other.read_,
                        [&](const BufferAllocation::Slice& slice) {
                          return HasConflicts(BufferUse::Read(slice));
                        }) ||
         absl::c_any_of(other.write_,
                        [&](const BufferAllocation::Slice& slice) {
                          return HasConflicts(BufferUse::Write(slice));
                        });
}
}  