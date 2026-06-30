#ifndef TENSORFLOW_CORE_UTIL_PRESIZED_CUCKOO_MAP_H_
#define TENSORFLOW_CORE_UTIL_PRESIZED_CUCKOO_MAP_H_
#include <algorithm>
#include <vector>
#include "absl/base/prefetch.h"
#include "absl/numeric/int128.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/macros.h"
namespace tensorflow {
template <class value>
class PresizedCuckooMap {
 public:
  typedef uint64 key_type;
  explicit PresizedCuckooMap(uint64 num_entries) { Clear(num_entries); }
  void Clear(uint64 num_entries) {
    cpq_.reset(new CuckooPathQueue());
    double n(num_entries);
    n /= kLoadFactor;
    num_buckets_ = (static_cast<uint64>(n) / kSlotsPerBucket);
    num_buckets_ += 32;
    Bucket empty_bucket;
    for (int i = 0; i < kSlotsPerBucket; i++) {
      empty_bucket.keys[i] = kUnusedSlot;
    }
    buckets_.clear();
    buckets_.resize(num_buckets_, empty_bucket);
  }
  bool InsertUnique(const key_type k, const value& v) {
    uint64 tk = key_transform(k);
    uint64 b1 = fast_map_to_buckets(tk);
    uint64 b2 = fast_map_to_buckets(h2(tk));
    uint64 target_bucket = 0;
    int target_slot = kNoSpace;
    for (auto bucket : {b1, b2}) {
      Bucket* bptr = &buckets_[bucket];
      for (int slot = 0; slot < kSlotsPerBucket; slot++) {
        if (bptr->keys[slot] == k) {  
          return false;
        } else if (target_slot == kNoSpace && bptr->keys[slot] == kUnusedSlot) {
          target_bucket = bucket;
          target_slot = slot;
        }
      }
    }
    if (target_slot != kNoSpace) {
      InsertInternal(tk, v, target_bucket, target_slot);
      return true;
    }
    return CuckooInsert(tk, v, b1, b2);
  }
  bool Find(const key_type k, value* out) const {
    uint64 tk = key_transform(k);
    return FindInBucket(k, fast_map_to_buckets(tk), out) ||
           FindInBucket(k, fast_map_to_buckets(h2(tk)), out);
  }
  void PrefetchKey(const key_type k) const {
    const uint64 tk = key_transform(k);
    absl::PrefetchToLocalCache(&buckets_[fast_map_to_buckets(tk)].keys);
    absl::PrefetchToLocalCache(&buckets_[fast_map_to_buckets(h2(tk))].keys);
  }
  int64_t MemoryUsed() const {
    return sizeof(PresizedCuckooMap<value>) + sizeof(CuckooPathQueue);
  }
 private:
  static constexpr int kSlotsPerBucket = 4;
  static constexpr double kLoadFactor = 0.85;
  static constexpr uint8 kMaxBFSPathLen = 5;
  static constexpr int kMaxQueueSize = 682;
  static constexpr int kVisitedListSize = 170;
  static constexpr int kNoSpace = -1;  
  static constexpr uint64 kUnusedSlot = ~(0ULL);
  struct Bucket {
    key_type keys[kSlotsPerBucket];
    value values[kSlotsPerBucket];
  };
  struct CuckooPathEntry {
    uint64 bucket;
    int depth;
    int parent;       
    int parent_slot;  
  };
  class CuckooPathQueue {
   public:
    CuckooPathQueue() : head_(0), tail_(0) {}
    void push_back(CuckooPathEntry e) {
      queue_[tail_] = e;
      tail_ = (tail_ + 1) % kMaxQueueSize;
    }
    CuckooPathEntry pop_front() {
      CuckooPathEntry& e = queue_[head_];
      head_ = (head_ + 1) % kMaxQueueSize;
      return e;
    }
    bool empty() const { return head_ == tail_; }
    bool full() const { return ((tail_ + 1) % kMaxQueueSize) == head_; }
    void reset() { head_ = tail_ = 0; }
   private:
    CuckooPathEntry queue_[kMaxQueueSize];
    int head_;
    int tail_;
  };
  typedef std::array<CuckooPathEntry, kMaxBFSPathLen> CuckooPath;
  inline uint64 key_transform(const key_type k) const {
    return k + (k == kUnusedSlot);
  }
  inline uint64 h2(uint64 h) const {
    const uint64 m = 0xc6a4a7935bd1e995;
    return m * ((h >> 32) | (h << 32));
  }
  inline uint64 alt_bucket(key_type k, uint64 b) const {
    if (fast_map_to_buckets(k) != b) {
      return fast_map_to_buckets(k);
    }
    return fast_map_to_buckets(h2(k));
  }
  inline void InsertInternal(key_type k, const value& v, uint64 b, int slot) {
    Bucket* bptr = &buckets_[b];
    bptr->keys[slot] = k;
    bptr->values[slot] = v;
  }
  bool FindInBucket(key_type k, uint64 b, value* out) const {
    const Bucket& bref = buckets_[b];
    for (int i = 0; i < kSlotsPerBucket; i++) {
      if (bref.keys[i] == k) {
        *out = bref.values[i];
        return true;
      }
    }
    return false;
  }
  inline int SpaceAvailable(uint64 bucket) const {
    const Bucket& bref = buckets_[bucket];
    for (int i = 0; i < kSlotsPerBucket; i++) {
      if (bref.keys[i] == kUnusedSlot) {
        return i;
      }
    }
    return kNoSpace;
  }
  inline void CopyItem(uint64 src_bucket, int src_slot, uint64 dst_bucket,
                       int dst_slot) {
    Bucket& src_ref = buckets_[src_bucket];
    Bucket& dst_ref = buckets_[dst_bucket];
    dst_ref.keys[dst_slot] = src_ref.keys[src_slot];
    dst_ref.values[dst_slot] = src_ref.values[src_slot];
  }
  bool CuckooInsert(key_type k, const value& v, uint64 b1, uint64 b2) {
    int visited_end = 0;
    cpq_->reset();
    cpq_->push_back({b1, 1, 0, 0});  
    cpq_->push_back({b2, 1, 0, 0});
    while (!cpq_->empty()) {
      CuckooPathEntry e = cpq_->pop_front();
      int free_slot;
      free_slot = SpaceAvailable(e.bucket);
      if (free_slot != kNoSpace) {
        while (e.depth > 1) {
          CuckooPathEntry parent = visited_[e.parent];
          CopyItem(parent.bucket, e.parent_slot, e.bucket, free_slot);
          free_slot = e.parent_slot;
          e = parent;
        }
        InsertInternal(k, v, e.bucket, free_slot);
        return true;
      } else {
        if (e.depth < (kMaxBFSPathLen)) {
          auto parent_index = visited_end;
          visited_[visited_end] = e;
          visited_end++;
          int start_slot = (k + e.bucket) % kSlotsPerBucket;
          const Bucket& bref = buckets_[e.bucket];
          for (int i = 0; i < kSlotsPerBucket; i++) {
            int slot = (start_slot + i) % kSlotsPerBucket;
            uint64 next_bucket = alt_bucket(bref.keys[slot], e.bucket);
            uint64 e_parent_bucket = visited_[e.parent].bucket;
            if (next_bucket != e_parent_bucket) {
              cpq_->push_back({next_bucket, e.depth + 1, parent_index, slot});
            }
          }
        }
      }
    }
    LOG(WARNING) << "Cuckoo path finding failed: Table too small?";
    return false;
  }
  inline uint64 fast_map_to_buckets(uint64 x) const {
    return absl::Uint128High64(absl::uint128(x) * absl::uint128(num_buckets_));
  }
  uint64 num_buckets_;
  std::vector<Bucket> buckets_;
  std::unique_ptr<CuckooPathQueue> cpq_;
  CuckooPathEntry visited_[kVisitedListSize];
  PresizedCuckooMap(const PresizedCuckooMap&) = delete;
  void operator=(const PresizedCuckooMap&) = delete;
};
}  
#endif  