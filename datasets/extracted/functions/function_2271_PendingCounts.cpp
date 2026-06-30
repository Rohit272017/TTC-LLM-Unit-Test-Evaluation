#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_PENDING_COUNTS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_PENDING_COUNTS_H_
#include <atomic>
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/port.h"
namespace tensorflow {
class PendingCounts {
 public:
  enum NodeState {
    PENDING_NOTREADY,
    PENDING_READY,
    STARTED,
    COMPLETED
  };
  class Handle;
  class Layout {
   public:
    Handle CreateHandle(size_t max_pending_count, size_t max_dead_count);
   private:
    friend class PendingCounts;
    int next_offset_ = 0;  
  };
  explicit PendingCounts(Layout layout)
      : num_bytes_(layout.next_offset_), bytes_(new char[num_bytes_]()) {
    if (num_bytes_ >= sizeof(LargeCounts)) {
      CHECK_EQ(uintptr_t(bytes_) % alignof(LargeCounts), 0);
    }
  }
  explicit PendingCounts(const PendingCounts& other)
      : num_bytes_(other.num_bytes_), bytes_(new char[num_bytes_]) {
    if (num_bytes_ >= sizeof(LargeCounts)) {
      CHECK_EQ(uintptr_t(bytes_) % alignof(LargeCounts), 0);
    }
    memcpy(bytes_, other.bytes_, other.num_bytes_);
  }
  ~PendingCounts() { delete[] bytes_; }
  void set_initial_count(Handle h, size_t pending_count) {
    if (h.is_large_) {
      std::atomic<LargeCounts>* c_ptr = Large(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      c.pending = pending_count;
      c.dead_count = 0;
      c.has_started = 0;
      c_ptr->store(c, std::memory_order_relaxed);
    } else {
      DCHECK_LE(pending_count, kMaxCountForPackedCounts);
      std::atomic<PackedCounts>* c_ptr = Packed(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      c.pending = pending_count;
      c.dead_count = 0;
      c.has_started = 0;
      c_ptr->store(c, std::memory_order_relaxed);
    }
  }
  NodeState node_state(Handle h) {
    if (h.is_large_) {
      return NodeStateForStruct(Large(h)->load(std::memory_order_relaxed));
    } else {
      return NodeStateForStruct(Packed(h)->load(std::memory_order_relaxed));
    }
  }
  void mark_started(Handle h) {
    DCHECK_EQ(pending(h), 0);
    if (h.is_large_) {
      std::atomic<LargeCounts>* c_ptr = Large(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      DCHECK_EQ(c.has_started, 0);
      c.has_started = 1;
      c_ptr->store(c, std::memory_order_relaxed);
    } else {
      std::atomic<PackedCounts>* c_ptr = Packed(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      DCHECK_EQ(c.has_started, 0);
      c.has_started = 1;
      c_ptr->store(c, std::memory_order_relaxed);
    }
  }
  void mark_completed(Handle h) {
    if (h.is_large_) {
      std::atomic<LargeCounts>* c_ptr = Large(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      DCHECK_EQ(c.has_started, 1);
      c.pending = 1;
      c_ptr->store(c, std::memory_order_relaxed);
    } else {
      std::atomic<PackedCounts>* c_ptr = Packed(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      DCHECK_EQ(c.has_started, 1);
      c.pending = 1;
      c_ptr->store(c, std::memory_order_relaxed);
    }
  }
  int pending(Handle h) {
    if (h.is_large_) {
      LargeCounts c = Large(h)->load(std::memory_order_relaxed);
      if (PENDING_NOTREADY == NodeStateForStruct(c)) {
        return c.pending;
      } else {
        return 0;
      }
    } else {
      PackedCounts c = Packed(h)->load(std::memory_order_relaxed);
      if (PENDING_NOTREADY == NodeStateForStruct(c)) {
        return c.pending;
      } else {
        return 0;
      }
    }
  }
  struct AdjustResult {
    int dead_count;
    int pending_count;
    AdjustResult(int dead_count, int pending_count)
        : dead_count(dead_count), pending_count(pending_count) {}
  };
  int decrement_pending(Handle h, int v) {
    DCHECK_GE(pending(h), v);
    if (h.is_large_) {
      std::atomic<LargeCounts>* c_ptr = Large(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      c.pending -= v;
      c_ptr->store(c, std::memory_order_relaxed);
      return c.pending;
    } else {
      std::atomic<PackedCounts>* c_ptr = Packed(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      c.pending -= v;
      c_ptr->store(c, std::memory_order_relaxed);
      return c.pending;
    }
  }
  void mark_live(Handle h) {
    if (h.is_large_) {
      std::atomic<LargeCounts>* c_ptr = Large(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      if (PENDING_NOTREADY == NodeStateForStruct(c)) {
        c.pending &= ~static_cast<int>(0x1);
        c_ptr->store(c, std::memory_order_relaxed);
      }
    } else {
      std::atomic<PackedCounts>* c_ptr = Packed(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      if (PENDING_NOTREADY == NodeStateForStruct(c)) {
        static_assert(7 == kMaxCountForPackedCounts,
                      "Live flag incorrect for max packed count");
        c.pending &= 0x6;
        c_ptr->store(c, std::memory_order_relaxed);
      }
    }
  }
  int dead_count(Handle h) {
    int r = h.is_large_ ? Large(h)->load(std::memory_order_relaxed).dead_count
                        : Packed(h)->load(std::memory_order_relaxed).dead_count;
    return r;
  }
  void increment_dead_count(Handle h) {
    if (h.is_large_) {
      std::atomic<LargeCounts>* c_ptr = Large(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      if (PENDING_NOTREADY == NodeStateForStruct(c)) {
        c.dead_count++;
        c_ptr->store(c, std::memory_order_relaxed);
      }
    } else {
      std::atomic<PackedCounts>* c_ptr = Packed(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      if (PENDING_NOTREADY == NodeStateForStruct(c)) {
        DCHECK_LT(c.dead_count, kMaxCountForPackedCounts);
        c.dead_count++;
        c_ptr->store(c, std::memory_order_relaxed);
      }
    }
  }
  AdjustResult adjust_for_mark_live(Handle h) {
    if (h.is_large_) {
      std::atomic<LargeCounts>* c_ptr = Large(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      auto ret_pending = 0;
      if (PENDING_NOTREADY == NodeStateForStruct(c)) {
        ret_pending = c.pending;
        c.pending &= ~static_cast<int>(0x1);
        c_ptr->store(c, std::memory_order_relaxed);
      }
      return AdjustResult(c.dead_count, ret_pending);
    } else {
      std::atomic<PackedCounts>* c_ptr = Packed(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      auto ret_pending = 0;
      if (PENDING_NOTREADY == NodeStateForStruct(c)) {
        static_assert(7 == kMaxCountForPackedCounts,
                      "Live flag incorrect for max packed count");
        ret_pending = c.pending;
        c.pending &= 0x6;
        c_ptr->store(c, std::memory_order_relaxed);
      }
      return AdjustResult(c.dead_count, ret_pending);
    }
  }
  AdjustResult adjust_for_mark_live_atomic(Handle h) {
    if (h.is_large_) {
      std::atomic<LargeCounts>* c_ptr = Large(h);
      auto old_val = c_ptr->load(std::memory_order_relaxed);
      while (true) {
        auto new_val = old_val;
        auto ret_pending = 0;
        if (PENDING_NOTREADY == NodeStateForStruct(new_val)) {
          ret_pending = old_val.pending;
          new_val.pending &= ~static_cast<int>(0x1);
        }
        AdjustResult ret(old_val.dead_count, ret_pending);
        if (TF_PREDICT_TRUE(c_ptr->compare_exchange_weak(old_val, new_val)))
          return ret;
      }
    } else {
      std::atomic<PackedCounts>* c_ptr = Packed(h);
      auto old_val = c_ptr->load(std::memory_order_relaxed);
      while (true) {
        auto new_val = old_val;
        auto ret_pending = 0;
        if (PENDING_NOTREADY == NodeStateForStruct(new_val)) {
          static_assert(7 == kMaxCountForPackedCounts,
                        "Live flag incorrect for max packed count");
          ret_pending = old_val.pending;
          new_val.pending &= 0x6;
        }
        AdjustResult ret(old_val.dead_count, ret_pending);
        if (TF_PREDICT_TRUE(c_ptr->compare_exchange_weak(old_val, new_val)))
          return ret;
      }
    }
  }
  AdjustResult adjust_for_increment_dead(Handle h) {
    if (h.is_large_) {
      return adjust_for_increment_dead_shared(Large(h));
    } else {
      return adjust_for_increment_dead_shared(Packed(h));
    }
  }
  AdjustResult adjust_for_increment_dead_atomic(Handle h) {
    if (h.is_large_) {
      return adjust_for_increment_dead_shared_atomic(Large(h));
    } else {
      return adjust_for_increment_dead_shared_atomic(Packed(h));
    }
  }
  AdjustResult adjust_for_decrement_pending(Handle h, int decrement_pending) {
    DCHECK_GE(pending(h), decrement_pending);
    if (h.is_large_) {
      return adjust_for_decrement_pending_shared(Large(h), decrement_pending);
    } else {
      return adjust_for_decrement_pending_shared(Packed(h), decrement_pending);
    }
  }
  AdjustResult adjust_for_decrement_pending_atomic(Handle h,
                                                   int decrement_pending) {
    DCHECK_GE(pending(h), decrement_pending);
    if (h.is_large_) {
      return adjust_for_decrement_pending_shared_atomic(Large(h),
                                                        decrement_pending);
    } else {
      return adjust_for_decrement_pending_shared_atomic(Packed(h),
                                                        decrement_pending);
    }
  }
  AdjustResult adjust_for_activation(Handle h, bool increment_dead) {
    DCHECK_GE(pending(h), 1);
    if (h.is_large_) {
      return adjust_for_activation_shared(Large(h), increment_dead);
    } else {
      return adjust_for_activation_shared(Packed(h), increment_dead);
    }
  }
  AdjustResult adjust_for_activation_atomic(Handle h, bool increment_dead) {
    DCHECK_GE(pending(h), 1);
    if (h.is_large_) {
      return adjust_for_activation_shared_atomic(Large(h), increment_dead);
    } else {
      return adjust_for_activation_shared_atomic(Packed(h), increment_dead);
    }
  }
  class Handle {
   public:
    Handle() : byte_offset_(0), is_large_(0) {}
   private:
    friend class PendingCounts;
    int byte_offset_ : 31;  
    bool is_large_ : 1;  
  };
 private:
  template <typename T>
  inline AdjustResult adjust_for_increment_dead_shared(std::atomic<T>* c) {
    T val = c->load(std::memory_order_relaxed);
    auto ret_pending = 0;
    if (PENDING_NOTREADY == NodeStateForStruct(val)) {
      val.dead_count++;
      ret_pending = val.pending;
      c->store(val, std::memory_order_relaxed);
    }
    return AdjustResult(val.dead_count, ret_pending);
  }
  template <typename T>
  inline AdjustResult adjust_for_increment_dead_shared_atomic(
      std::atomic<T>* c) {
    T old_val = c->load(std::memory_order_relaxed);
    while (true) {
      auto new_val = old_val;
      auto ret_pending = 0;
      if (PENDING_NOTREADY == NodeStateForStruct(new_val)) {
        ret_pending = new_val.pending;
        new_val.dead_count++;
      }
      AdjustResult ret(new_val.dead_count, ret_pending);
      if (TF_PREDICT_TRUE(c->compare_exchange_weak(old_val, new_val)))
        return ret;
    }
  }
  template <typename T>
  inline AdjustResult adjust_for_decrement_pending_shared(
      std::atomic<T>* c, int decrement_pending) {
    T val = c->load(std::memory_order_relaxed);
    DCHECK_GE(val.pending, decrement_pending);
    val.pending -= decrement_pending;
    c->store(val, std::memory_order_relaxed);
    return AdjustResult(val.dead_count, val.pending);
  }
  template <typename T>
  inline AdjustResult adjust_for_decrement_pending_shared_atomic(
      std::atomic<T>* c, int decrement_pending) {
    T old_val = c->load(std::memory_order_relaxed);
    while (true) {
      T new_val = old_val;
      DCHECK_GE(new_val.pending, decrement_pending);
      new_val.pending -= decrement_pending;
      AdjustResult ret(new_val.dead_count, new_val.pending);
      if (TF_PREDICT_TRUE(c->compare_exchange_weak(old_val, new_val)))
        return ret;
    }
  }
  template <typename T>
  inline AdjustResult adjust_for_activation_shared(std::atomic<T>* c,
                                                   bool increment_dead) {
    T val = c->load(std::memory_order_relaxed);
    if (increment_dead && PENDING_NOTREADY == NodeStateForStruct(val)) {
      val.dead_count++;
    }
    DCHECK_GE(val.pending, 1);
    val.pending--;
    c->store(val, std::memory_order_relaxed);
    return AdjustResult(val.dead_count, val.pending);
  }
  template <typename T>
  inline AdjustResult adjust_for_activation_shared_atomic(std::atomic<T>* c,
                                                          bool increment_dead) {
    T old_val = c->load(std::memory_order_relaxed);
    while (true) {
      T new_val = old_val;
      if (increment_dead && PENDING_NOTREADY == NodeStateForStruct(new_val)) {
        new_val.dead_count++;
      }
      DCHECK_GE(new_val.pending, 1);
      new_val.pending--;
      AdjustResult ret(new_val.dead_count, new_val.pending);
      if (TF_PREDICT_TRUE(c->compare_exchange_weak(old_val, new_val)))
        return ret;
    }
  }
  static constexpr int kMaxCountForPackedCounts = 7;
  struct PackedCounts {
    uint8 pending : 3;
    uint8 dead_count : 3;
    uint8 has_started : 1;
  };
  struct alignas(8) LargeCounts {
    uint32 pending;
    uint32 dead_count : 31;
    uint32 has_started : 1;
  };
  template <typename T>
  NodeState NodeStateForStruct(const T& c) const {
    if (c.has_started) {
      return (c.pending == 0) ? STARTED : COMPLETED;
    } else {
      return (c.pending == 0) ? PENDING_READY : PENDING_NOTREADY;
    }
  }
  inline std::atomic<LargeCounts>* Large(Handle h) {
    DCHECK(h.is_large_);
    DCHECK_LE(h.byte_offset_ + sizeof(std::atomic<LargeCounts>), num_bytes_);
    DCHECK_EQ(h.byte_offset_ % alignof(std::atomic<LargeCounts>), 0);
    return reinterpret_cast<std::atomic<LargeCounts>*>(bytes_ + h.byte_offset_);
  }
  inline std::atomic<PackedCounts>* Packed(Handle h) {
    DCHECK(!h.is_large_);
    DCHECK_LE(h.byte_offset_ + sizeof(PackedCounts), num_bytes_);
    return reinterpret_cast<std::atomic<PackedCounts>*>(bytes_ +
                                                        h.byte_offset_);
  }
  const int num_bytes_;  
  char* bytes_;          
  void operator=(const PendingCounts&) = delete;
};
inline PendingCounts::Handle PendingCounts::Layout::CreateHandle(
    size_t max_pending_count, size_t max_dead_count) {
  Handle result;
  if ((max_pending_count > kMaxCountForPackedCounts) ||
      (max_dead_count > kMaxCountForPackedCounts)) {
    constexpr int B = sizeof(std::atomic<LargeCounts>);
    static_assert(
        sizeof(std::atomic<LargeCounts>) >= alignof(std::atomic<LargeCounts>),
        "std::atomic<LargeCounts> must be packed");
    int64_t offset = ((static_cast<int64_t>(next_offset_) + B - 1) / B) * B;
    result.byte_offset_ = offset;
    result.is_large_ = true;
    next_offset_ = result.byte_offset_ + B;
  } else {
    result.byte_offset_ = next_offset_;
    result.is_large_ = false;
    static_assert(sizeof(std::atomic<PackedCounts>) == 1,
                  "std::atomic<PackedCounts> should be a single byte");
    next_offset_ += sizeof(std::atomic<PackedCounts>);
  }
  return result;
}
}  
#endif  