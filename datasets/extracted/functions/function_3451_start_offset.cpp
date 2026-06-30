#ifndef TENSORSTORE_INTERNAL_THREAD_BLOCK_QUEUE_H_
#define TENSORSTORE_INTERNAL_THREAD_BLOCK_QUEUE_H_
#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include <cassert>
#include <memory>
#include <utility>
#include "absl/base/attributes.h"
#include "absl/log/absl_check.h"
#include "tensorstore/internal/container/item_traits.h"
namespace tensorstore {
namespace internal_container {
template <typename T, size_t kMin = 1024, size_t kMax = 1024,
          typename Allocator = std::allocator<T>>
class BlockQueue;
template <typename T, typename Allocator>
class SQBlock {
 private:
  using BlockAllocator =
      typename std::allocator_traits<Allocator>::template rebind_alloc<SQBlock>;
  using ByteAllocator =
      typename std::allocator_traits<Allocator>::template rebind_alloc<char>;
  constexpr static ptrdiff_t start_offset() {
    struct X {
      SQBlock array;
      T item[1];
    };
    return offsetof(X, item);
  }
  constexpr static size_t start_items() {
    return (start_offset() + sizeof(T) - 1) / sizeof(T);
  }
  struct private_t {
   private:
    friend class SQBlock;
    private_t() = default;
  };
 public:
  static SQBlock* New(int64_t c, Allocator* alloc) {
    size_t allocation_bytes =
        (c <= start_items() + 2)
            ? (start_offset() + 2 * sizeof(T))
            : (c -= start_items(), ((c + start_items()) * sizeof(T)));
    ByteAllocator byte_alloc(*alloc);
    void* mem = std::allocator_traits<ByteAllocator>::allocate(
        byte_alloc, allocation_bytes);
    auto* as_array = static_cast<SQBlock*>(mem);
    BlockAllocator array_alloc(*alloc);
    std::allocator_traits<BlockAllocator>::construct(array_alloc, as_array,
                                                     private_t{}, c);
    return as_array;
  }
  static void Delete(SQBlock* ptr, Allocator* alloc) {
    const size_t allocation_bytes =
        (ptr->capacity() == 2) ? (start_offset() + 2 * sizeof(T))
                               : (start_items() + ptr->capacity()) * sizeof(T);
    BlockAllocator block_alloc(*alloc);
    std::allocator_traits<BlockAllocator>::destroy(block_alloc, ptr);
    void* mem = ptr;
    ByteAllocator byte_alloc(*alloc);
    std::allocator_traits<ByteAllocator>::deallocate(
        byte_alloc, static_cast<char*>(mem), allocation_bytes);
  }
  SQBlock(private_t, size_t c) : end_(begin() + c), next_(nullptr) {}
  SQBlock* next() const { return next_; }
  void set_next(SQBlock* b) { next_ = b; }
  T* begin() {
    return reinterpret_cast<T*>(reinterpret_cast<char*>(this) + start_offset());
  }
  T* end() { return end_; }
  size_t capacity() { return end() - begin(); }
 private:
  T* end_;
  SQBlock* next_;
};
template <typename T, size_t kMin, size_t kMax, typename Allocator>
class BlockQueue {
  using Block = SQBlock<T, Allocator>;
  using TransferTraits = ItemTraits<T>;
  static constexpr bool kDestroyIsTrivial =
      TransferTraits::template destroy_is_trivial<Allocator>();
  static_assert(kMin > 0);
  static_assert(kMin <= kMax);
  struct Cursor {
    Cursor(Block* b) : block(b), ptr(b->begin()), end(b->end()) {}
    Cursor() : block(nullptr), ptr(nullptr), end(nullptr) {}
    Block* block;
    T* ptr;
    T* end;
  };
 public:
  BlockQueue() : BlockQueue(Allocator()) {}
  explicit BlockQueue(Allocator alloc)
      : allocator_(std::move(alloc)), head_(), tail_(), size_(0) {}
  ~BlockQueue() {
    Block* b = head_.block;
    while (b) {
      Block* next = b->next();
      ClearBlock(b);
      Block::Delete(b, &allocator_);
      b = next;
    }
  }
  BlockQueue(const BlockQueue&) = delete;
  BlockQueue& operator=(const BlockQueue&) = delete;
  size_t size() const { return size_; }
  bool empty() const { return !size(); }
  T& front() {
    ABSL_CHECK(!empty());
    return *head_.ptr;
  }
  const T& front() const {
    ABSL_CHECK(!empty());
    return *head_.ptr;
  }
  T& back() {
    ABSL_CHECK(!empty());
    return *((tail_.ptr) - 1);
  }
  const T& back() const {
    ABSL_CHECK(!empty());
    return *((tail_.ptr) - 1);
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
    ABSL_CHECK(head_.block);
    TransferTraits::destroy(&allocator_, head_.ptr);
    ++head_.ptr;
    --size_;
    if (empty()) {
      ABSL_CHECK_EQ(head_.block, tail_.block);
      head_.ptr = tail_.ptr = head_.block->begin();
      return;
    }
    if (head_.ptr == head_.end) {
      Block* n = head_.block->next();
      Block::Delete(head_.block, &allocator_);
      head_ = Cursor(n);
    }
  }
  void clear() {
    Block* b = head_.block;
    if (!b) {
      ABSL_CHECK(empty());
      return;
    }
    while (b) {
      Block* next = b->next();
      ClearBlock(b);
      if (head_.block != b) {
        Block::Delete(b, &allocator_);
      }
      b = next;
    }
    b = head_.block;
    b->set_next(nullptr);
    head_ = tail_ = Cursor(b);
    size_ = 0;
  }
 private:
  T* emplace_back_raw() {
    if (tail_.ptr == tail_.end) {
      size_t capacity = kMin;
      if (tail_.block) {
        capacity = 2 * tail_.block->capacity();
        if (capacity > kMax) capacity = kMax;
      }
      auto* b = Block::New(capacity, &allocator_);
      if (!head_.block) {
        ABSL_CHECK(tail_.block == nullptr);
        head_ = Cursor(b);
      } else {
        ABSL_CHECK(head_.block != nullptr);
        tail_.block->set_next(b);
      }
      tail_ = Cursor(b);
    }
    ++size_;
    return tail_.ptr++;
  }
  void ClearBlock(Block* b) {
    auto* begin = b == head_.block ? head_.ptr : b->begin();
    auto* end = b == tail_.block ? tail_.ptr : b->end();
    if constexpr (!kDestroyIsTrivial) {
      for (; begin != end; ++begin) {
        TransferTraits::destroy(&allocator_, begin);
      }
    }
  }
  ABSL_ATTRIBUTE_NO_UNIQUE_ADDRESS Allocator allocator_;
  Cursor head_;
  Cursor tail_;
  size_t size_;
};
}  
}  
#endif  