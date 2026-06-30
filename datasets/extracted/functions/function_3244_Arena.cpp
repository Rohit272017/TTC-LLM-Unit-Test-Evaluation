#include "tensorflow/core/lib/core/arena.h"
#include <assert.h>
#include <algorithm>
#include <vector>
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mem.h"
namespace tensorflow {
namespace core {
Arena::Arena(const size_t block_size)
    : remaining_(0),
      block_size_(block_size),
      freestart_(nullptr),  
      blocks_alloced_(1),
      overflow_blocks_(nullptr) {
  assert(block_size > kDefaultAlignment);
  first_blocks_[0].mem =
      reinterpret_cast<char*>(port::AlignedMalloc(block_size_, sizeof(void*)));
  first_blocks_[0].size = block_size_;
  Reset();
}
Arena::~Arena() {
  FreeBlocks();
  assert(overflow_blocks_ == nullptr);  
  for (size_t i = 0; i < blocks_alloced_; ++i) {
    port::AlignedFree(first_blocks_[i].mem);
  }
}
bool Arena::SatisfyAlignment(size_t alignment) {
  const size_t overage = reinterpret_cast<size_t>(freestart_) & (alignment - 1);
  if (overage > 0) {
    const size_t waste = alignment - overage;
    if (waste >= remaining_) {
      return false;
    }
    freestart_ += waste;
    remaining_ -= waste;
  }
  DCHECK_EQ(size_t{0}, reinterpret_cast<size_t>(freestart_) & (alignment - 1));
  return true;
}
void Arena::Reset() {
  FreeBlocks();
  freestart_ = first_blocks_[0].mem;
  remaining_ = first_blocks_[0].size;
  CHECK(SatisfyAlignment(kDefaultAlignment));
  freestart_when_empty_ = freestart_;
}
void Arena::MakeNewBlock(const uint32 alignment) {
  AllocatedBlock* block = AllocNewBlock(block_size_, alignment);
  freestart_ = block->mem;
  remaining_ = block->size;
  CHECK(SatisfyAlignment(alignment));
}
static uint32 LeastCommonMultiple(uint32 a, uint32 b) {
  if (a > b) {
    return (a / MathUtil::GCD<uint32>(a, b)) * b;
  } else if (a < b) {
    return (b / MathUtil::GCD<uint32>(b, a)) * a;
  } else {
    return a;
  }
}
Arena::AllocatedBlock* Arena::AllocNewBlock(const size_t block_size,
                                            const uint32 alignment) {
  AllocatedBlock* block;
  if (blocks_alloced_ < TF_ARRAYSIZE(first_blocks_)) {
    block = &first_blocks_[blocks_alloced_++];
  } else {  
    if (overflow_blocks_ == nullptr)
      overflow_blocks_ = new std::vector<AllocatedBlock>;
    overflow_blocks_->resize(overflow_blocks_->size() + 1);
    block = &overflow_blocks_->back();
  }
  uint32 adjusted_alignment =
      (alignment > 1 ? LeastCommonMultiple(alignment, kDefaultAlignment) : 1);
  adjusted_alignment =
      std::max(adjusted_alignment, static_cast<uint32>(sizeof(void*)));
  CHECK_LE(adjusted_alignment, static_cast<uint32>(1 << 20))
      << "Alignment on boundaries greater than 1MB not supported.";
  size_t adjusted_block_size = block_size;
  if (adjusted_block_size > adjusted_alignment) {
    const uint32 excess = adjusted_block_size % adjusted_alignment;
    adjusted_block_size += (excess > 0 ? adjusted_alignment - excess : 0);
  }
  block->mem = reinterpret_cast<char*>(
      port::AlignedMalloc(adjusted_block_size, adjusted_alignment));
  block->size = adjusted_block_size;
  CHECK(nullptr != block->mem) << "block_size=" << block_size
                               << " adjusted_block_size=" << adjusted_block_size
                               << " alignment=" << alignment
                               << " adjusted_alignment=" << adjusted_alignment;
  return block;
}
void* Arena::GetMemoryFallback(const size_t size, const int alignment) {
  if (0 == size) {
    return nullptr;  
  }
  CHECK(alignment > 0 && 0 == (alignment & (alignment - 1)));
  if (block_size_ == 0 || size > block_size_ / 4) {
    return AllocNewBlock(size, alignment)->mem;
  }
  if (!SatisfyAlignment(alignment) || size > remaining_) {
    MakeNewBlock(alignment);
  }
  CHECK_LE(size, remaining_);
  remaining_ -= size;
  void* result = freestart_;
  freestart_ += size;
  return result;
}
void Arena::FreeBlocks() {
  for (size_t i = 1; i < blocks_alloced_; ++i) {  
    port::AlignedFree(first_blocks_[i].mem);
    first_blocks_[i].mem = nullptr;
    first_blocks_[i].size = 0;
  }
  blocks_alloced_ = 1;
  if (overflow_blocks_ != nullptr) {
    std::vector<AllocatedBlock>::iterator it;
    for (it = overflow_blocks_->begin(); it != overflow_blocks_->end(); ++it) {
      port::AlignedFree(it->mem);
    }
    delete overflow_blocks_;  
    overflow_blocks_ = nullptr;
  }
}
}  
}  