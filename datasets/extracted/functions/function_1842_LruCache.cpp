#ifndef AROLLA_UTIL_LRU_CACHE_H_
#define AROLLA_UTIL_LRU_CACHE_H_
#include <cstddef>
#include <functional>
#include <list>
#include <utility>
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
namespace arolla {
template <typename Key, typename Value, typename KeyHash = absl::Hash<Key>,
          typename KeyEq = std::equal_to<>>
class LruCache {
 public:
  explicit LruCache(size_t capacity) : capacity_(capacity) {
    DCHECK_GT(capacity, 0);
    index_.reserve(capacity + 1);
  }
  LruCache(const LruCache&) = delete;
  LruCache& operator=(const LruCache&) = delete;
  template <typename K>
  [[nodiscard]] absl::Nullable<const Value*> LookupOrNull(K&& key) {
    if (auto it = index_.find(std::forward<K>(key)); it != index_.end()) {
      entries_.splice(entries_.begin(), entries_, it->entry);
      return &it->entry->second;
    }
    return nullptr;
  }
  template <typename K, typename V>
  [[nodiscard]] absl::Nonnull<const Value*> Put(K&& key, V&& value) {
    entries_.emplace_front(std::forward<K>(key), std::forward<V>(value));
    const auto& [it, ok] = index_.emplace(IndexEntry{entries_.begin()});
    if (!ok) {
      entries_.pop_front();  
      entries_.splice(entries_.begin(), entries_, it->entry);
    } else if (entries_.size() > capacity_) {
      index_.erase(entries_.back().first);
      entries_.pop_back();
    }
    DCHECK_LE(entries_.size(), capacity_);
    DCHECK_EQ(entries_.size(), index_.size());
    return &entries_.front().second;
  }
  void Clear() {
    entries_.clear();
    index_.clear();
  }
 private:
  using Entry = std::pair<const Key, const Value>;
  struct IndexEntry {
    typename std::list<Entry>::iterator entry;
  };
  struct IndexRecordHash {
    using is_transparent = void;
    size_t operator()(const IndexEntry& index_record) const {
      return KeyHash()(index_record.entry->first);
    }
    template <typename K>
    size_t operator()(K&& key) const {
      return KeyHash()(std::forward<K>(key));
    }
  };
  struct IndexRecordEq {
    using is_transparent = void;
    bool operator()(const IndexEntry& lhs, const IndexEntry& rhs) const {
      return KeyEq()(lhs.entry->first, rhs.entry->first);
    }
    template <typename K>
    bool operator()(const IndexEntry& lhs, K&& rhs) const {
      return KeyEq()(lhs.entry->first, std::forward<K>(rhs));
    }
  };
  using Index = absl::flat_hash_set<IndexEntry, IndexRecordHash, IndexRecordEq>;
  size_t capacity_;
  std::list<Entry> entries_;
  Index index_;
};
}  
#endif  