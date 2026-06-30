#ifndef XLA_PJRT_LRU_CACHE_H_
#define XLA_PJRT_LRU_CACHE_H_
#include <optional>
#include <unordered_map>
#include "absl/container/node_hash_map.h"
#include "tsl/platform/logging.h"
namespace xla {
template <typename Key, typename Value,
          typename Hash = typename absl::node_hash_map<Key, Value>::hasher,
          typename Eq = typename absl::node_hash_map<Key, Value>::key_equal>
class LRUCache {
 private:
  struct LRUListEntry {
    LRUListEntry* next;
    LRUListEntry* prev;
  };
 public:
  class LRUList {
   public:
    explicit LRUList(int capacity) : capacity_(capacity) {
      head_.next = &head_;
      head_.prev = &head_;
    }
    ~LRUList() {
      CHECK(head_.next == &head_);
      CHECK(head_.prev == &head_);
    }
    LRUList(const LRUList&) = delete;
    LRUList(LRUList&&) = delete;
    LRUList& operator=(const LRUList&) = delete;
    LRUList& operator=(LRUList&&) = delete;
    int Capacity() const { return capacity_; }
    int Size() const { return size_; }
    void Clear();
   private:
    friend class LRUCache;
    int capacity_;
    int size_ = 0;
    LRUListEntry head_;
  };
  explicit LRUCache(LRUList* lru_list) : lru_list_(lru_list) {}
  ~LRUCache();
  LRUCache(const LRUCache&) = delete;
  LRUCache(LRUCache&&) = delete;
  LRUCache& operator=(const LRUCache&) = delete;
  LRUCache& operator=(LRUCache&&) = delete;
  Value GetOrCreateIfAbsent(const Key& key,
                            const std::function<Value(const Key&)>& factory);
  void Remove(const Key& key);
  void Clear();
  int Size() const { return entries_.size(); }
  int Capacity() const { return lru_list_->Capacity(); }
  auto begin() const { return entries_.begin(); }
  auto end() const { return entries_.end(); }
 private:
  LRUList* lru_list_;
  struct Entry : public LRUListEntry {
    Entry() = default;
    const Key* key;
    LRUCache* container;
    std::optional<Value> value;
  };
  std::unordered_map<Key, Entry, Hash, Eq> entries_;
};
template <typename Key, typename Value, typename Hash, typename Eq>
void LRUCache<Key, Value, Hash, Eq>::LRUList::Clear() {
  while (head_.next != &head_) {
    static_cast<Entry*>(head_.next)->container->Clear();
  }
  size_ = 0;
}
template <typename Key, typename Value, typename Hash, typename Eq>
void LRUCache<Key, Value, Hash, Eq>::Clear() {
  for (auto& e : entries_) {
    LRUListEntry* l = &e.second;
    l->next->prev = l->prev;
    l->prev->next = l->next;
    --lru_list_->size_;
  }
  entries_.clear();
}
template <typename Key, typename Value, typename Hash, typename Eq>
LRUCache<Key, Value, Hash, Eq>::~LRUCache() {
  Clear();
}
template <typename Key, typename Value, typename Hash, typename Eq>
void LRUCache<Key, Value, Hash, Eq>::Remove(const Key& key) {
  LRUListEntry* l = &entries_[key];
  l->next->prev = l->prev;
  l->prev->next = l->next;
  --lru_list_->size_;
  entries_.erase(key);
}
template <typename Key, typename Value, typename Hash, typename Eq>
Value LRUCache<Key, Value, Hash, Eq>::GetOrCreateIfAbsent(
    const Key& key, const std::function<Value(const Key&)>& factory) {
  auto [it, inserted] = entries_.try_emplace(key);
  Entry& entry = it->second;
  if (inserted) {
    entry.key = &it->first;
    entry.value = factory(*entry.key);
    ++lru_list_->size_;
  } else {
    entry.prev->next = entry.next;
    entry.next->prev = entry.prev;
  }
  LRUListEntry& lru_head = lru_list_->head_;
  entry.container = this;
  entry.prev = lru_head.prev;
  entry.next = &lru_head;
  lru_head.prev->next = &entry;
  lru_head.prev = &entry;
  Value v = *entry.value;
  if (lru_list_->size_ > lru_list_->capacity_) {
    Entry* to_remove = static_cast<Entry*>(lru_head.next);
    to_remove->next->prev = &lru_head;
    lru_head.next = to_remove->next;
    to_remove->container->entries_.extract(*to_remove->key);
    --lru_list_->size_;
  }
  return v;
}
}  
#endif  