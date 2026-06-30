#ifndef XLA_TSL_CONCURRENCY_CONCURRENT_VECTOR_H_
#define XLA_TSL_CONCURRENCY_CONCURRENT_VECTOR_H_
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tsl/platform/logging.h"
namespace tsl {
namespace internal {
template <typename T>
class ConcurrentVector {
 public:
  explicit ConcurrentVector(size_t initial_capacity) : state_(0ull) {
    auto& v = all_allocated_elements_[0];
    v.reserve(std::max(static_cast<size_t>(1), initial_capacity));
  }
  const T& operator[](size_t index) const {
    auto state = State::Decode(state_.load(std::memory_order_acquire));
    DCHECK_LT(index, state.size);
    return all_allocated_elements_.data()[state.last_allocated].data()[index];
  }
  absl::Span<const T> ToConstSpan() const {
    auto state = State::Decode(state_.load(std::memory_order_acquire));
    auto& storage = all_allocated_elements_[state.last_allocated];
    return absl::MakeConstSpan(storage.data(), state.size);
  }
  size_t size() const {
    return State::Decode(state_.load(std::memory_order_relaxed)).size;
  }
  template <typename... Args>
  size_t emplace_back(Args&&... args) {
    absl::MutexLock lock(&mutex_);
    auto state = State::Decode(state_.load(std::memory_order_relaxed));
    auto& last = all_allocated_elements_[state.last_allocated];
    if (last.size() < last.capacity()) {
      last.emplace_back(std::forward<Args>(args)...);
      state.size += 1;
      state_.store(state.Encode(), std::memory_order_release);
      return state.size - 1;  
    }
    auto& new_last = all_allocated_elements_[state.last_allocated + 1];
    new_last.reserve(last.capacity() * 2);
    DCHECK_EQ(last.size(), last.capacity());
    new_last.insert(new_last.begin(), last.begin(), last.end());
    new_last.emplace_back(std::forward<Args>(args)...);
    state.last_allocated += 1;
    state.size += 1;
    state_.store(state.Encode(), std::memory_order_release);
    return state.size - 1;  
  }
 private:
  static constexpr uint64_t kLastAllocatedMask = (1ull << 32) - 1;
  static constexpr uint64_t kSizeMask = ((1ull << 32) - 1) << 32;
  struct State {
    uint64_t last_allocated;  
    uint64_t size;            
    static State Decode(uint64_t state) {
      uint64_t last_allocated = (state & kLastAllocatedMask);
      uint64_t size = (state & kSizeMask) >> 32;
      return {last_allocated, size};
    }
    uint64_t Encode() const { return (size << 32) | last_allocated; }
  };
  std::atomic<uint64_t> state_;
  absl::Mutex mutex_;
  std::array<std::vector<T>, 64> all_allocated_elements_;
};
}  
}  
#endif  