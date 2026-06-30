#ifndef AROLLA_UTIL_MEMORY_ALLOCATION_H_
#define AROLLA_UTIL_MEMORY_ALLOCATION_H_
#include <utility>
#include "absl/log/check.h"
#include "arolla/memory/frame.h"
#include "arolla/util/memory.h"
namespace arolla {
class MemoryAllocation {
 public:
  MemoryAllocation() = default;
  explicit MemoryAllocation(const FrameLayout* layout)
      : layout_(layout),
        alloc_(AlignedAlloc(layout->AllocAlignment(), layout->AllocSize())) {
    layout_->InitializeAlignedAlloc(alloc_.get());
  }
  MemoryAllocation(const MemoryAllocation&) = delete;
  MemoryAllocation& operator=(const MemoryAllocation&) = delete;
  MemoryAllocation(MemoryAllocation&&) = default;
  MemoryAllocation& operator=(MemoryAllocation&& other) {
    if (alloc_ != nullptr) {
      layout_->DestroyAlloc(alloc_.get());
    }
    layout_ = other.layout_;
    alloc_ = std::move(other.alloc_);
    return *this;
  }
  ~MemoryAllocation() {
    if (alloc_ != nullptr) {
      layout_->DestroyAlloc(alloc_.get());
    }
  }
  bool IsValid() const { return alloc_ != nullptr; }
  FramePtr frame() {
    DCHECK(IsValid());
    return FramePtr(alloc_.get(), layout_);
  }
  ConstFramePtr frame() const {
    DCHECK(IsValid());
    return ConstFramePtr(alloc_.get(), layout_);
  }
 private:
  const FrameLayout* layout_ = nullptr;
  MallocPtr alloc_ = nullptr;
};
}  
#endif  