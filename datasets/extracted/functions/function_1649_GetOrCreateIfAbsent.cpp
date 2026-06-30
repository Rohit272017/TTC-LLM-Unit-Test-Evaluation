#ifndef XLA_REFCOUNTING_HASH_MAP_H_
#define XLA_REFCOUNTING_HASH_MAP_H_
#include <functional>
#include <memory>
#include "absl/base/thread_annotations.h"
#include "absl/container/node_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
namespace xla {
template <typename K, typename V>
class RefcountingHashMap {
 public:
  RefcountingHashMap() = default;
  RefcountingHashMap(const RefcountingHashMap&) = delete;
  RefcountingHashMap(RefcountingHashMap&&) = delete;
  RefcountingHashMap& operator=(const RefcountingHashMap&) = delete;
  RefcountingHashMap& operator=(RefcountingHashMap&&) = delete;
  std::shared_ptr<V> GetOrCreateIfAbsent(
      const K& key,
      absl::FunctionRef<std::unique_ptr<V>(const K&)> value_factory) {
    absl::MutexLock lock(&mu_);
    auto it = map_.find(key);
    if (it != map_.end()) {
      if (std::shared_ptr<V> value = it->second.lock()) {
        return value;
      }
    }
    it = map_.emplace(key, std::weak_ptr<V>()).first;
    std::shared_ptr<V> value(value_factory(key).release(),
                             Deleter{it->first, *this});
    it->second = value;  
    return value;
  }
 private:
  struct Deleter {
    const K& key;  
    RefcountingHashMap& parent;
    void operator()(V* v) {
      delete v;
      absl::MutexLock lock(&parent.mu_);
      auto it = parent.map_.find(key);
      if (it != parent.map_.end() && it->second.expired()) {
        parent.map_.erase(it);
      }
    }
  };
  absl::Mutex mu_;
  absl::node_hash_map<K, std::weak_ptr<V>> map_ ABSL_GUARDED_BY(mu_);
};
}  
#endif  