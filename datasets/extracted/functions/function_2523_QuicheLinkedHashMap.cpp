#ifndef QUICHE_COMMON_QUICHE_LINKED_HASH_MAP_H_
#define QUICHE_COMMON_QUICHE_LINKED_HASH_MAP_H_
#include <functional>
#include <list>
#include <tuple>
#include <type_traits>
#include <utility>
#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "quiche/common/platform/api/quiche_export.h"
#include "quiche/common/platform/api/quiche_logging.h"
namespace quiche {
template <class Key,                      
          class Value,                    
          class Hash = absl::Hash<Key>,   
          class Eq = std::equal_to<Key>>  
class QuicheLinkedHashMap {               
 private:
  typedef std::list<std::pair<Key, Value>> ListType;
  typedef absl::flat_hash_map<Key, typename ListType::iterator, Hash, Eq>
      MapType;
 public:
  typedef typename ListType::iterator iterator;
  typedef typename ListType::reverse_iterator reverse_iterator;
  typedef typename ListType::const_iterator const_iterator;
  typedef typename ListType::const_reverse_iterator const_reverse_iterator;
  typedef typename MapType::key_type key_type;
  typedef typename ListType::value_type value_type;
  typedef typename ListType::size_type size_type;
  QuicheLinkedHashMap() = default;
  explicit QuicheLinkedHashMap(size_type bucket_count) : map_(bucket_count) {}
  QuicheLinkedHashMap(const QuicheLinkedHashMap& other) = delete;
  QuicheLinkedHashMap& operator=(const QuicheLinkedHashMap& other) = delete;
  QuicheLinkedHashMap(QuicheLinkedHashMap&& other) = default;
  QuicheLinkedHashMap& operator=(QuicheLinkedHashMap&& other) = default;
  iterator begin() { return list_.begin(); }
  const_iterator begin() const { return list_.begin(); }
  iterator end() { return list_.end(); }
  const_iterator end() const { return list_.end(); }
  reverse_iterator rbegin() { return list_.rbegin(); }
  const_reverse_iterator rbegin() const { return list_.rbegin(); }
  reverse_iterator rend() { return list_.rend(); }
  const_reverse_iterator rend() const { return list_.rend(); }
  const value_type& front() const { return list_.front(); }
  value_type& front() { return list_.front(); }
  const value_type& back() const { return list_.back(); }
  value_type& back() { return list_.back(); }
  void clear() {
    map_.clear();
    list_.clear();
  }
  bool empty() const { return list_.empty(); }
  void pop_front() { erase(begin()); }
  size_type erase(const Key& key) {
    typename MapType::iterator found = map_.find(key);
    if (found == map_.end()) {
      return 0;
    }
    list_.erase(found->second);
    map_.erase(found);
    return 1;
  }
  iterator erase(iterator position) {
    typename MapType::iterator found = map_.find(position->first);
    QUICHE_CHECK(found->second == position)
        << "Inconsistent iterator for map and list, or the iterator is "
           "invalid.";
    map_.erase(found);
    return list_.erase(position);
  }
  iterator erase(iterator first, iterator last) {
    while (first != last && first != end()) {
      first = erase(first);
    }
    return first;
  }
  iterator find(const Key& key) {
    typename MapType::iterator found = map_.find(key);
    if (found == map_.end()) {
      return end();
    }
    return found->second;
  }
  const_iterator find(const Key& key) const {
    typename MapType::const_iterator found = map_.find(key);
    if (found == map_.end()) {
      return end();
    }
    return found->second;
  }
  bool contains(const Key& key) const { return find(key) != end(); }
  Value& operator[](const key_type& key) {
    return (*((this->insert(std::make_pair(key, Value()))).first)).second;
  }
  std::pair<iterator, bool> insert(const std::pair<Key, Value>& pair) {
    return InsertInternal(pair);
  }
  std::pair<iterator, bool> insert(std::pair<Key, Value>&& pair) {
    return InsertInternal(std::move(pair));
  }
  size_type size() const { return map_.size(); }
  template <typename... Args>
  std::pair<iterator, bool> emplace(Args&&... args) {
    ListType node_donor;
    auto node_pos =
        node_donor.emplace(node_donor.end(), std::forward<Args>(args)...);
    const auto& k = node_pos->first;
    auto ins = map_.insert({k, node_pos});
    if (!ins.second) {
      return {ins.first->second, false};
    }
    list_.splice(list_.end(), node_donor, node_pos);
    return {ins.first->second, true};
  }
  void swap(QuicheLinkedHashMap& other) {
    map_.swap(other.map_);
    list_.swap(other.list_);
  }
 private:
  template <typename U>
  std::pair<iterator, bool> InsertInternal(U&& pair) {
    auto insert_result = map_.try_emplace(pair.first);
    auto map_iter = insert_result.first;
    if (!insert_result.second) {
      return {map_iter->second, false};
    }
    auto list_iter = list_.insert(list_.end(), std::forward<U>(pair));
    map_iter->second = list_iter;
    return {list_iter, true};
  }
  MapType map_;
  ListType list_;
};
}  
#endif  