#ifndef QUICHE_QUIC_CORE_QUIC_LRU_CACHE_H_
#define QUICHE_QUIC_CORE_QUIC_LRU_CACHE_H_
#include <memory>
#include "quiche/quic/platform/api/quic_export.h"
#include "quiche/quic/platform/api/quic_flag_utils.h"
#include "quiche/quic/platform/api/quic_flags.h"
#include "quiche/quic/platform/api/quic_logging.h"
#include "quiche/common/quiche_linked_hash_map.h"
namespace quic {
template <class K, class V, class Hash = std::hash<K>,
          class Eq = std::equal_to<K>>
class QUICHE_EXPORT QuicLRUCache {
 private:
  using HashMapType =
      typename quiche::QuicheLinkedHashMap<K, std::unique_ptr<V>, Hash, Eq>;
 public:
  using iterator = typename HashMapType::iterator;
  using const_iterator = typename HashMapType::const_iterator;
  using reverse_iterator = typename HashMapType::reverse_iterator;
  using const_reverse_iterator = typename HashMapType::const_reverse_iterator;
  explicit QuicLRUCache(size_t capacity) : capacity_(capacity) {}
  QuicLRUCache(const QuicLRUCache&) = delete;
  QuicLRUCache& operator=(const QuicLRUCache&) = delete;
  iterator begin() { return cache_.begin(); }
  const_iterator begin() const { return cache_.begin(); }
  iterator end() { return cache_.end(); }
  const_iterator end() const { return cache_.end(); }
  reverse_iterator rbegin() { return cache_.rbegin(); }
  const_reverse_iterator rbegin() const { return cache_.rbegin(); }
  reverse_iterator rend() { return cache_.rend(); }
  const_reverse_iterator rend() const { return cache_.rend(); }
  void Insert(const K& key, std::unique_ptr<V> value) {
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      cache_.erase(it);
    }
    cache_.emplace(key, std::move(value));
    if (cache_.size() > capacity_) {
      cache_.pop_front();
    }
    QUICHE_DCHECK_LE(cache_.size(), capacity_);
  }
  iterator Lookup(const K& key) {
    auto iter = cache_.find(key);
    if (iter == cache_.end()) {
      return iter;
    }
    std::unique_ptr<V> value = std::move(iter->second);
    cache_.erase(iter);
    auto result = cache_.emplace(key, std::move(value));
    QUICHE_DCHECK(result.second);
    return result.first;
  }
  iterator Erase(iterator iter) { return cache_.erase(iter); }
  void Clear() { cache_.clear(); }
  size_t MaxSize() const { return capacity_; }
  size_t Size() const { return cache_.size(); }
 private:
  quiche::QuicheLinkedHashMap<K, std::unique_ptr<V>, Hash, Eq> cache_;
  const size_t capacity_;
};
}  
#endif  