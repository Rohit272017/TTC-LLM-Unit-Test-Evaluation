#ifndef XLA_TSL_PROFILER_UTILS_PER_THREAD_H_
#define XLA_TSL_PROFILER_UTILS_PER_THREAD_H_
#include <memory>
#include <vector>
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
namespace tsl {
namespace profiler {
template <typename T>
class PerThread {
 public:
  static T& Get() {
    static thread_local ThreadLocalPtr thread;
    return thread.Get();
  }
  static std::vector<std::shared_ptr<T>> StartRecording() {
    return Registry::Get().StartRecording();
  }
  static std::vector<std::shared_ptr<T>> StopRecording() {
    return Registry::Get().StopRecording();
  }
 private:
  PerThread() = delete;
  ~PerThread() = delete;
  class Registry {
   public:
    static Registry& Get() {
      static Registry* singleton = new Registry();
      return *singleton;
    }
    std::vector<std::shared_ptr<T>> StartRecording() {
      std::vector<std::shared_ptr<T>> threads;
      absl::MutexLock lock(&mutex_);
      threads.reserve(threads_.size());
      for (auto iter = threads_.begin(); iter != threads_.end(); ++iter) {
        threads.push_back(iter->first);
      }
      recording_ = true;
      return threads;
    }
    std::vector<std::shared_ptr<T>> StopRecording() {
      std::vector<std::shared_ptr<T>> threads;
      absl::MutexLock lock(&mutex_);
      threads.reserve(threads_.size());
      for (auto iter = threads_.begin(); iter != threads_.end();) {
        if (!iter->second) {  
          threads.push_back(std::move(iter->first));
          threads_.erase(iter++);
        } else {
          threads.push_back(iter->first);
          ++iter;
        }
      }
      recording_ = false;
      return threads;
    }
    void Register(std::shared_ptr<T> thread) {
      absl::MutexLock lock(&mutex_);
      threads_.insert_or_assign(std::move(thread), true);
    }
    void Unregister(const std::shared_ptr<T>& thread) {
      absl::MutexLock lock(&mutex_);
      if (!recording_) {
        threads_.erase(thread);
      } else {
        if (auto it = threads_.find(thread); it != threads_.end()) {
          it->second = false;
        }
      }
    }
   private:
    Registry() = default;
    Registry(const Registry&) = delete;
    void operator=(const Registry&) = delete;
    absl::Mutex mutex_;
    absl::flat_hash_map<std::shared_ptr<T>, bool> threads_
        ABSL_GUARDED_BY(mutex_);
    bool recording_ ABSL_GUARDED_BY(mutex_) = false;
  };
  class ThreadLocalPtr {
   public:
    ThreadLocalPtr() : ptr_(std::make_shared<T>()) {
      Registry::Get().Register(ptr_);
    }
    ~ThreadLocalPtr() { Registry::Get().Unregister(ptr_); }
    T& Get() { return *ptr_; }
   private:
    std::shared_ptr<T> ptr_;
  };
};
}  
}  
#endif  