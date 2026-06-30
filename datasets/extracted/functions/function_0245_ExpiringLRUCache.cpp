#ifndef TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_GCS_EXPIRING_LRU_CACHE_H_
#define TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_GCS_EXPIRING_LRU_CACHE_H_
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <string>
#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/c/env.h"
#include "tensorflow/c/tf_status.h"
namespace tf_gcs_filesystem {
template <typename T>
class ExpiringLRUCache {
 public:
  ExpiringLRUCache(uint64_t max_age, size_t max_entries,
                   std::function<uint64_t()> timer_seconds = TF_NowSeconds)
      : max_age_(max_age),
        max_entries_(max_entries),
        timer_seconds_(timer_seconds) {}
  void Insert(const std::string& key, const T& value) {
    if (max_age_ == 0) {
      return;
    }
    absl::MutexLock lock(&mu_);
    InsertLocked(key, value);
  }
  bool Delete(const std::string& key) {
    absl::MutexLock lock(&mu_);
    return DeleteLocked(key);
  }
  bool Lookup(const std::string& key, T* value) {
    if (max_age_ == 0) {
      return false;
    }
    absl::MutexLock lock(&mu_);
    return LookupLocked(key, value);
  }
  typedef std::function<void(const std::string&, T*, TF_Status*)> ComputeFunc;
  void LookupOrCompute(const std::string& key, T* value,
                       const ComputeFunc& compute_func, TF_Status* status) {
    if (max_age_ == 0) {
      return compute_func(key, value, status);
    }
    absl::MutexLock lock(&mu_);
    if (LookupLocked(key, value)) {
      return TF_SetStatus(status, TF_OK, "");
    }
    compute_func(key, value, status);
    if (TF_GetCode(status) == TF_OK) {
      InsertLocked(key, *value);
    }
  }
  void Clear() {
    absl::MutexLock lock(&mu_);
    cache_.clear();
    lru_list_.clear();
  }
  uint64_t max_age() const { return max_age_; }
  size_t max_entries() const { return max_entries_; }
 private:
  struct Entry {
    uint64_t timestamp;
    T value;
    std::list<std::string>::iterator lru_iterator;
  };
  bool LookupLocked(const std::string& key, T* value)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      return false;
    }
    lru_list_.erase(it->second.lru_iterator);
    if (timer_seconds_() - it->second.timestamp > max_age_) {
      cache_.erase(it);
      return false;
    }
    *value = it->second.value;
    lru_list_.push_front(it->first);
    it->second.lru_iterator = lru_list_.begin();
    return true;
  }
  void InsertLocked(const std::string& key, const T& value)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    lru_list_.push_front(key);
    Entry entry{timer_seconds_(), value, lru_list_.begin()};
    auto insert = cache_.insert(std::make_pair(key, entry));
    if (!insert.second) {
      lru_list_.erase(insert.first->second.lru_iterator);
      insert.first->second = entry;
    } else if (max_entries_ > 0 && cache_.size() > max_entries_) {
      cache_.erase(lru_list_.back());
      lru_list_.pop_back();
    }
  }
  bool DeleteLocked(const std::string& key) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      return false;
    }
    lru_list_.erase(it->second.lru_iterator);
    cache_.erase(it);
    return true;
  }
  const uint64_t max_age_;
  const size_t max_entries_;
  std::function<uint64_t()> timer_seconds_;
  absl::Mutex mu_;
  std::map<std::string, Entry> cache_ ABSL_GUARDED_BY(mu_);
  std::list<std::string> lru_list_ ABSL_GUARDED_BY(mu_);
};
}  
#endif  