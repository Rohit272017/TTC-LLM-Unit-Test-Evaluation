#ifndef TENSORSTORE_INTERNAL_THREAD_SINGLE_PRODUCER_QUEUE_H_
#define TENSORSTORE_INTERNAL_THREAD_SINGLE_PRODUCER_QUEUE_H_
#include <stdint.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>
#include <type_traits>
#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/log/absl_check.h"
namespace tensorstore {
namespace internal_container {
template <typename T, bool kCanResize = true,
          typename Allocator = std::allocator<T>>
class SingleProducerQueue;
template <typename T, typename Allocator>
class SPQArray {
 private:
  static_assert(std::is_trivially_destructible_v<T>);
  using ArrayAllocator = typename std::allocator_traits<
      Allocator>::template rebind_alloc<SPQArray>;
  using ByteAllocator =
      typename std::allocator_traits<Allocator>::template rebind_alloc<char>;
  constexpr static ptrdiff_t start_offset() {
    struct X {
      SPQArray array;
      std::atomic<T> item[1];
    };
    return offsetof(X, item);
  }
  constexpr static size_t alloc_size(int64_t c) {
    struct X {
      SPQArray array;
      std::atomic<T> item[1];
    };
    return sizeof(X) + (c - 1) * sizeof(std::atomic<T>);
  }
  struct private_t {
   private:
    friend class SPQArray;
    private_t() = default;
  };
 public:
  static SPQArray* New(int64_t c, SPQArray* retired, Allocator* alloc) {
    size_t allocation_bytes = alloc_size(c);
    ByteAllocator byte_alloc(*alloc);
    void* mem = std::allocator_traits<ByteAllocator>::allocate(
        byte_alloc, allocation_bytes);
    auto* as_array = static_cast<SPQArray*>(mem);
    ArrayAllocator array_alloc(*alloc);
    std::allocator_traits<ArrayAllocator>::construct(array_alloc, as_array,
                                                     private_t{}, c, retired);
    return as_array;
  }
  static void Delete(SPQArray* ptr, Allocator* alloc) {
    const size_t allocation_bytes = alloc_size(ptr->capacity);
    void* mem = ptr;
    ByteAllocator byte_alloc(*alloc);
    std::allocator_traits<ByteAllocator>::deallocate(
        byte_alloc, static_cast<char*>(mem), allocation_bytes);
  }
  SPQArray(private_t, int64_t c, SPQArray* retired)
      : capacity(c), mask(c - 1), retired(retired) {}
  SPQArray* resize(int64_t b, int64_t t, Allocator* alloc) {
    auto* a = SPQArray::New(2 * capacity, this, alloc);
    for (int64_t i = t; i != b; ++i) {
      a->item(i).store(  
          item(i).load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    return a;
  }
  std::atomic<T>* buffer() {
    return reinterpret_cast<std::atomic<T>*>(reinterpret_cast<char*>(this) +
                                             start_offset());
  }
  std::atomic<T>& item(int64_t i) { return buffer()[i & mask]; }
  int64_t capacity;
  int64_t mask;
  SPQArray* retired;
};
template <typename T, bool kCanResize, typename Allocator>
class SingleProducerQueue {
  static_assert(std::is_trivially_destructible_v<T>);
  std::nullopt_t missing(std::false_type) { return std::nullopt; }
  std::nullptr_t missing(std::true_type) { return nullptr; }
  using Array = SPQArray<T, Allocator>;
 public:
  using optional_t =
      std::conditional_t<std::is_pointer_v<T>, T, std::optional<T>>;
  SingleProducerQueue(int64_t n, Allocator alloc)
      : top_(0),
        bottom_(0),
        allocator_(alloc),
        array_(Array::New(n, nullptr, &allocator_)) {
    ABSL_CHECK_EQ(n & (n - 1), 0);
  }
  explicit SingleProducerQueue(int64_t n)
      : SingleProducerQueue(n, Allocator()) {}
  ~SingleProducerQueue() {
    Array* a = array_.load(std::memory_order_relaxed);
    while (a) {
      Array* b = a->retired;
      a->retired = nullptr;
      Array::Delete(a, &allocator_);
      a = b;
    }
  }
  int64_t capacity() const {
    return array_.load(std::memory_order_relaxed)->capacity;
  }
  size_t size() const {
    int64_t b = bottom_.load(std::memory_order_relaxed);
    int64_t t = top_.load(std::memory_order_relaxed);
    return static_cast<size_t>(b > t ? b - t : 0);
  }
  bool empty() const { return !size(); }
  bool push(T x) {
    auto b = bottom_.load(std::memory_order_relaxed);
    auto t = top_.load(std::memory_order_acquire);
    Array* a = array_.load(std::memory_order_relaxed);
    if (a->capacity < (b - t) + 1) {
      if (!kCanResize) return false;
      a = a->resize(b, t, &allocator_);
      array_.store(a, std::memory_order_release);
    }
    a->item(b).store(std::move(x), std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    bottom_.store(b + 1, std::memory_order_relaxed);
    return true;
  }
  optional_t try_pop() {
    auto b = bottom_.load(std::memory_order_relaxed) - 1;
    Array* a = array_.load(std::memory_order_relaxed);
    bottom_.store(b, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto t = top_.load(std::memory_order_relaxed);
    if (t > b) {
      bottom_.store(b + 1, std::memory_order_relaxed);
      return missing(std::is_pointer<T>{});
    }
    if (t == b) {
      if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst,
                                        std::memory_order_relaxed)) {
        bottom_.store(b + 1, std::memory_order_relaxed);
        return missing(std::is_pointer<T>{});
      }
      bottom_.store(b + 1, std::memory_order_relaxed);
    }
    return a->item(b).load(std::memory_order_relaxed);
  }
  optional_t try_steal() {
    auto t = top_.load(std::memory_order_acquire);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto b = bottom_.load(std::memory_order_acquire);
    if (t >= b) {
      return missing(std::is_pointer<T>{});
    }
    Array* a = array_.load(std::memory_order_consume);
    T x = a->item(t).load(std::memory_order_relaxed);
    if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst,
                                      std::memory_order_relaxed)) {
      return missing(std::is_pointer<T>{});
    }
    return x;
  }
 private:
  ABSL_CACHELINE_ALIGNED std::atomic<int64_t> top_;
  std::atomic<int64_t> bottom_;
  ABSL_ATTRIBUTE_NO_UNIQUE_ADDRESS Allocator allocator_;
  std::atomic<Array*> array_;
};
}  
}  
#endif  