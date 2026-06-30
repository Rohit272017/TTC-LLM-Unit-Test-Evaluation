#include "tensorflow/core/common_runtime/pool_allocator.h"
#include <errno.h>
#ifndef _MSC_VER
#include <strings.h>
#include <sys/mman.h>  
#endif
#include <map>
#include <utility>
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/numa.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/profiler/lib/traceme.h"
namespace tensorflow {
PoolAllocator::PoolAllocator(size_t pool_size_limit, bool auto_resize,
                             SubAllocator* allocator,
                             RoundUpInterface* size_rounder, string name)
    : name_(std::move(name)),
      has_size_limit_(pool_size_limit > 0),
      auto_resize_(auto_resize),
      pool_size_limit_(pool_size_limit),
      allocator_(allocator),
      size_rounder_(size_rounder) {
  if (auto_resize) {
    CHECK_LT(size_t{0}, pool_size_limit)
        << "size limit must be > 0 if auto_resize is true.";
  }
}
PoolAllocator::~PoolAllocator() { Clear(); }
namespace {
struct ChunkPrefix {
  size_t num_bytes;
  void* chunk_ptr;
};
static const int kPoolAlignment = sizeof(ChunkPrefix);
void* PrepareChunk(void* chunk, size_t alignment, size_t num_bytes) {
  ChunkPrefix* cp = reinterpret_cast<ChunkPrefix*>(chunk);
  cp->num_bytes = num_bytes;
  cp->chunk_ptr = chunk;
  void* user_ptr = reinterpret_cast<void*>(cp + 1);
  if (alignment > kPoolAlignment) {
    size_t aligned_ptr = reinterpret_cast<size_t>(user_ptr) + alignment;
    user_ptr = reinterpret_cast<void*>(aligned_ptr & ~(alignment - 1));
    (reinterpret_cast<ChunkPrefix*>(user_ptr) - 1)->chunk_ptr = chunk;
  }
  CHECK_GE(user_ptr, reinterpret_cast<ChunkPrefix*>(chunk) + 1);
  return user_ptr;
}
ChunkPrefix* FindPrefix(void* user_ptr) {
  ChunkPrefix* cp = reinterpret_cast<ChunkPrefix*>(user_ptr) - 1;
  return reinterpret_cast<ChunkPrefix*>(cp->chunk_ptr);
}
}  
void* PoolAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  if (num_bytes == 0) return nullptr;
  if (alignment > kPoolAlignment) {
    num_bytes += alignment;
  }
  num_bytes += sizeof(ChunkPrefix);
  num_bytes = size_rounder_->RoundUp(num_bytes);
  PtrRecord* pr = nullptr;
  if (has_size_limit_) {
    {
      mutex_lock lock(mutex_);
      auto iter = pool_.find(num_bytes);
      if (iter == pool_.end()) {
        allocated_count_++;
      } else {
        get_from_pool_count_++;
        pr = iter->second;
        RemoveFromList(pr);
        pool_.erase(iter);
      }
    }
  }
  if (pr != nullptr) {
    void* r = pr->ptr;
    delete pr;
    return PrepareChunk(r, alignment, num_bytes);
  } else {
    size_t bytes_received;
    void* ptr = allocator_->Alloc(kPoolAlignment, num_bytes, &bytes_received);
    return PrepareChunk(ptr, alignment, bytes_received);
  }
}
void PoolAllocator::DeallocateRaw(void* ptr) {
  if (ptr == nullptr) return;
  ChunkPrefix* cp = FindPrefix(ptr);
  CHECK_LE((void*)cp, (void*)ptr);
  if (!has_size_limit_ && !auto_resize_) {
    allocator_->Free(cp, cp->num_bytes);
  } else {
    mutex_lock lock(mutex_);
    ++put_count_;
    while (pool_.size() >= pool_size_limit_) {
      EvictOne();
    }
    PtrRecord* pr = new PtrRecord;
    pr->num_bytes = cp->num_bytes;
    pr->ptr = cp;
    AddToList(pr);
    pool_.insert(std::make_pair(cp->num_bytes, pr));
  }
}
void PoolAllocator::Clear() {
  if (has_size_limit_) {
    mutex_lock lock(mutex_);
    for (auto iter : pool_) {
      PtrRecord* pr = iter.second;
      allocator_->Free(pr->ptr, pr->num_bytes);
      delete pr;
    }
    pool_.clear();
    get_from_pool_count_ = 0;
    put_count_ = 0;
    allocated_count_ = 0;
    evicted_count_ = 0;
    lru_head_ = nullptr;
    lru_tail_ = nullptr;
  }
}
void PoolAllocator::RemoveFromList(PtrRecord* pr) {
  if (pr->prev == nullptr) {
    DCHECK_EQ(lru_head_, pr);
    lru_head_ = nullptr;
  } else {
    pr->prev->next = pr->next;
  }
  if (pr->next == nullptr) {
    DCHECK_EQ(lru_tail_, pr);
    lru_tail_ = pr->prev;
  } else {
    pr->next->prev = pr->prev;
    if (lru_head_ == nullptr) {
      lru_head_ = pr->next;
    }
  }
}
void PoolAllocator::AddToList(PtrRecord* pr) {
  pr->prev = nullptr;
  if (lru_head_ == nullptr) {
    CHECK(lru_tail_ == nullptr);
    lru_tail_ = pr;
    pr->next = nullptr;
  } else {
    pr->next = lru_head_;
    pr->next->prev = pr;
  }
  lru_head_ = pr;
}
void PoolAllocator::EvictOne() {
  DCHECK(lru_tail_ != nullptr);
  PtrRecord* prec = lru_tail_;
  RemoveFromList(prec);
  auto iter = pool_.find(prec->num_bytes);
  while (iter->second != prec) {
    ++iter;
    DCHECK(iter != pool_.end());
  }
  pool_.erase(iter);
  allocator_->Free(prec->ptr, prec->num_bytes);
  delete prec;
  ++evicted_count_;
  static const double kTolerable = 2e-3;
  static const int kCheckInterval = 1000;
  static const double kIncreaseFactor = 1.1;
  static const int kMinPoolSize = 100;
  if (0 == evicted_count_ % kCheckInterval) {
    const double eviction_rate =
        evicted_count_ / static_cast<double>(put_count_);
    const int64_t alloc_request_count = allocated_count_ + get_from_pool_count_;
    const double alloc_rate =
        (alloc_request_count == 0)
            ? 0.0
            : allocated_count_ / static_cast<double>(alloc_request_count);
    const bool kShouldLog = false;
    if (kShouldLog) {
      LOG(INFO) << "PoolAllocator: After " << alloc_request_count
                << " get requests, put_count=" << put_count_
                << " evicted_count=" << evicted_count_
                << " eviction_rate=" << eviction_rate
                << " and unsatisfied allocation rate=" << alloc_rate;
    }
    if (auto_resize_ && (eviction_rate > kTolerable) &&
        (alloc_rate > kTolerable)) {
      size_t new_size_limit = (pool_size_limit_ < kMinPoolSize)
                                  ? kMinPoolSize
                                  : (kIncreaseFactor * pool_size_limit_);
      if (kShouldLog) {
        LOG(INFO) << "Raising pool_size_limit_ from " << pool_size_limit_
                  << " to " << new_size_limit;
      }
      pool_size_limit_ = new_size_limit;
      put_count_ = 0;
      allocated_count_ = 0;
      evicted_count_ = 0;
      get_from_pool_count_ = 0;
    }
  }
}
void* BasicCPUAllocator::Alloc(size_t alignment, size_t num_bytes,
                               size_t* bytes_received) {
  tsl::profiler::TraceMe traceme("BasicCPUAllocator::Alloc");
  void* ptr = nullptr;
  *bytes_received = num_bytes;
  if (num_bytes > 0) {
    if (numa_node_ == port::kNUMANoAffinity) {
      ptr = port::AlignedMalloc(num_bytes, static_cast<int>(alignment));
    } else {
      ptr =
          port::NUMAMalloc(numa_node_, num_bytes, static_cast<int>(alignment));
    }
    VisitAlloc(ptr, numa_node_, num_bytes);
  }
  return ptr;
}
void BasicCPUAllocator::Free(void* ptr, size_t num_bytes) {
  tsl::profiler::TraceMe traceme("BasicCPUAllocator::Free");
  if (num_bytes > 0) {
    VisitFree(ptr, numa_node_, num_bytes);
    if (numa_node_ == port::kNUMANoAffinity) {
      port::AlignedFree(ptr);
    } else {
      port::NUMAFree(ptr, num_bytes);
    }
  }
}
}  