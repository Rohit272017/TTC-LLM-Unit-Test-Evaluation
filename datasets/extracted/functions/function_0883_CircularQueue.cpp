#ifndef TENSORSTORE_INTERNAL_THREAD_CIRCULAR_QUEUE_H_
#define TENSORSTORE_INTERNAL_THREAD_CIRCULAR_QUEUE_H_
#include <stddef.h>
#include <algorithm>
#include <cassert>
#include <memory>
#include <type_traits>
#include "absl/base/attributes.h"
#include "absl/log/absl_check.h"
#include "tensorstore/internal/container/item_traits.h"
namespace tensorstore {
namespace internal_container {
template <typename T, typename Allocator = std::allocator<T>>
class CircularQueue {
  using TransferTraits = ItemTraits<T>;
  using Storage = typename std::aligned_storage<sizeof(T), alignof(T)>::type;
  static_assert(sizeof(T) == sizeof(Storage));
  using StorageAllocator =
      typename std::allocator_traits<Allocator>::template rebind_alloc<Storage>;
  using StorageAllocatorTraits = std::allocator_traits<StorageAllocator>;
  static constexpr bool kDestroyIsTrivial =
      TransferTraits::template destroy_is_trivial<Allocator>();
 public:
  explicit CircularQueue(size_t n) : CircularQueue(n, Allocator()) {}
  CircularQueue(size_t n, Allocator alloc)
      : allocator_(std::move(alloc)),
        begin_(0),
        end_(0),
        mask_(0),
        buffer_(nullptr) {
    ABSL_CHECK_EQ(n & (n - 1), 0);
    internal_resize(n);
  }
  ~CircularQueue() {
    clear();
    if (buffer_) {
      StorageAllocator storage_alloc(allocator_);
      StorageAllocatorTraits::deallocate(
          storage_alloc, reinterpret_cast<Storage*>(buffer_), mask_ + 1);
    }
  }
  CircularQueue(const CircularQueue&) = delete;
  CircularQueue& operator=(const CircularQueue&) = delete;
  size_t capacity() const { return mask_ + 1; }
  size_t size() const { return end_ - begin_; }
  bool empty() const { return !size(); }
  T& front() {
    ABSL_CHECK(!empty());
    return buffer_[begin_ & mask_];
  }
  const T& front() const {
    ABSL_CHECK(!empty());
    return buffer_[begin_ & mask_];
  }
  T& back() {
    ABSL_CHECK(!empty());
    return buffer_[(end_ - 1) & mask_];
  }
  const T& back() const {
    ABSL_CHECK(!empty());
    return buffer_[(end_ - 1) & mask_];
  }
  T& operator[](size_t i) {
    ABSL_CHECK_LT(i, size());
    return buffer_[(begin_ + i) & mask_];
  }
  const T& operator[](size_t i) const {
    ABSL_CHECK_LT(i, size());
    return buffer_[(begin_ + i) & mask_];
  }
  void push_back(const T& val) { emplace_back(val); }
  void push_back(T&& val) { emplace_back(std::move(val)); }
  template <typename... A>
  T& emplace_back(A&&... args) {
    auto* storage = emplace_back_raw();
    TransferTraits::construct(&allocator_, storage, std::forward<A>(args)...);
    return *storage;
  }
  void pop_front() {
    ABSL_CHECK(!empty());
    auto x = begin_++;
    if constexpr (!kDestroyIsTrivial) {
      TransferTraits::destroy(&allocator_, buffer_ + (x & mask_));
    }
  }
  void clear() {
    if constexpr (!kDestroyIsTrivial) {
      for (size_t i = begin_; i < end_; i++) {
        TransferTraits::destroy(&allocator_, buffer_ + (i & mask_));
      }
    }
    begin_ = 0;
    end_ = 0;
  }
 private:
  T* emplace_back_raw() {
    if (size() == capacity()) {
      internal_resize((mask_ + 1) * 2);
    }
    return buffer_ + (end_++ & mask_);
  }
  void internal_resize(size_t c) {
    ABSL_CHECK_EQ(c & (c - 1), 0);
    ABSL_CHECK_GT(c, mask_ + 1);
    StorageAllocator storage_alloc(allocator_);
    T* new_buffer = std::launder(reinterpret_cast<T*>(
        StorageAllocatorTraits::allocate(storage_alloc, c)));
    size_t j = 0;
    for (size_t i = begin_; i < end_; i++) {
      auto* storage = buffer_ + (i & mask_);
      TransferTraits::transfer(&allocator_, new_buffer + j++, storage);
    }
    if (buffer_) {
      StorageAllocatorTraits::deallocate(
          storage_alloc, reinterpret_cast<Storage*>(buffer_), mask_ + 1);
    }
    begin_ = 0;
    end_ = j;
    mask_ = c - 1;
    buffer_ = new_buffer;
  }
  ABSL_ATTRIBUTE_NO_UNIQUE_ADDRESS Allocator allocator_;
  size_t begin_;
  size_t end_;
  size_t mask_;
  T* buffer_;
};
}  
}  
#endif  