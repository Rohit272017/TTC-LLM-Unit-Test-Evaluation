#ifndef TENSORFLOW_TSL_PLATFORM_CLOUD_EXPIRING_LRU_CACHE_H_
#define TENSORFLOW_TSL_PLATFORM_CLOUD_EXPIRING_LRU_CACHE_H_
#include <list>
#include <map>
#include <memory>
#include <string>
#include "tsl/platform/env.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/thread_annotations.h"
#include "tsl/platform/types.h"
namespace tsl {
template <typename T>
class ExpiringLRUCache {
 public:
  ExpiringLRUCache(uint64 max_age, size_t max_entries,
                   Env* env = Env::Default())
      : max_age_(max_age), max_entries_(max_entries), env_(env) {}
  void Insert(const string& key, const T& value) {
    if (max_age_ == 0) {
      return;
    }
    mutex_lock lock(mu_);
    InsertLocked(key, value);
  }
  bool Delete(const string& key) {
    mutex_lock lock(mu_);
    return DeleteLocked(key);
  }
  bool Lookup(const string& key, T* value) {
    if (max_age_ == 0) {
      return false;
    }
    mutex_lock lock(mu_);
    return LookupLocked(key, value);
  }
  typedef std::function<absl::Status(const string&, T*)> ComputeFunc;
  absl::Status LookupOrCompute(const string& key, T* value,
                               const ComputeFunc& compute_func) {
    if (max_age_ == 0) {
      return compute_func(key, value);
    }
    mutex_lock lock(mu_);
    if (LookupLocked(key, value)) {
      return absl::OkStatus();
    }
    absl::Status s = compute_func(key, value);
    if (s.ok()) {
      InsertLocked(key, *value);
    }
    return s;
  }
  void Clear() {
    mutex_lock lock(mu_);
    cache_.clear();
    lru_list_.clear();
  }
  uint64 max_age() const { return max_age_; }
  size_t max_entries() const { return max_entries_; }
 private:
  struct Entry {
    uint64 timestamp;
    T value;
    std::list<string>::iterator lru_iterator;
  };
  bool LookupLocked(const string& key, T* value)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      return false;
    }
    lru_list_.erase(it->second.lru_iterator);
    if (env_->NowSeconds() - it->second.timestamp > max_age_) {
      cache_.erase(it);
      return false;
    }
    *value = it->second.value;
    lru_list_.push_front(it->first);
    it->second.lru_iterator = lru_list_.begin();
    return true;
  }
  void InsertLocked(const string& key, const T& value)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    lru_list_.push_front(key);
    Entry entry{env_->NowSeconds(), value, lru_list_.begin()};
    auto insert = cache_.insert(std::make_pair(key, entry));
    if (!insert.second) {
      lru_list_.erase(insert.first->second.lru_iterator);
      insert.first->second = entry;
    } else if (max_entries_ > 0 && cache_.size() > max_entries_) {
      cache_.erase(lru_list_.back());
      lru_list_.pop_back();
    }
  }
  bool DeleteLocked(const string& key) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      return false;
    }
    lru_list_.erase(it->second.lru_iterator);
    cache_.erase(it);
    return true;
  }
  const uint64 max_age_;
  const size_t max_entries_;
  Env* const env_;  
  mutex mu_;
  std::map<string, Entry> cache_ TF_GUARDED_BY(mu_);
  std::list<string> lru_list_ TF_GUARDED_BY(mu_);
};
}  
#endif  