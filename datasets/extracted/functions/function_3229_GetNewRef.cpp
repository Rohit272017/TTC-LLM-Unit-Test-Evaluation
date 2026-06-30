#ifndef TENSORFLOW_TSL_PLATFORM_REFCOUNT_H_
#define TENSORFLOW_TSL_PLATFORM_REFCOUNT_H_
#include <atomic>
#include <map>
#include <memory>
#include "tsl/platform/logging.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/thread_annotations.h"
namespace tsl {
namespace core {
class RefCounted {
 public:
  RefCounted();
  void Ref() const;
  bool Unref() const;
  int_fast32_t RefCount() const;
  bool RefCountIsOne() const;
 protected:
  virtual ~RefCounted();
  bool TryRef() const;
  virtual void NotifyDeleted() const;
 private:
  mutable std::atomic_int_fast32_t ref_;
  RefCounted(const RefCounted&) = delete;
  void operator=(const RefCounted&) = delete;
};
struct RefCountDeleter {
  void operator()(const RefCounted* o) const { o->Unref(); }
};
template <typename T>
class RefCountPtr;
template <typename T>
ABSL_MUST_USE_RESULT RefCountPtr<T> GetNewRef(T* ptr) {
  static_assert(std::is_base_of<RefCounted, T>::value);
  if (ptr == nullptr) return RefCountPtr<T>();
  ptr->Ref();
  RefCountPtr<T> ret(ptr);
  return ret;
}
template <typename T>
class RefCountPtr : public std::unique_ptr<T, RefCountDeleter> {
 public:
  using std::unique_ptr<T, RefCountDeleter>::unique_ptr;
  ABSL_MUST_USE_RESULT RefCountPtr GetNewRef() const {
    if (this->get() == nullptr) return RefCountPtr<T>();
    this->get()->Ref();
    return RefCountPtr<T>(this->get());
  }
};
class ScopedUnref {
 public:
  explicit ScopedUnref(const RefCounted* o) : obj_(o) {}
  ~ScopedUnref() {
    if (obj_) obj_->Unref();
  }
 private:
  const RefCounted* obj_;
  ScopedUnref(const ScopedUnref&) = delete;
  void operator=(const ScopedUnref&) = delete;
};
template <typename T>
class WeakPtr;
using WeakNotifyFn = std::function<void()>;
class WeakRefCounted : public RefCounted {
 public:
  int WeakRefCount() const {
    return data_->RefCount() - 1;
  }
 protected:
  void NotifyDeleted() const override { data_->Notify(); }
 private:
  struct WeakRefData : public RefCounted {
    explicit WeakRefData(WeakRefCounted* ptr) : ptr(ptr), next_notifier_id(1) {}
    mutable mutex mu;
    WeakRefCounted* ptr TF_GUARDED_BY(mu);
    std::map<int, WeakNotifyFn> notifiers;
    int next_notifier_id;
    void Notify() {
      mutex_lock ml(mu);
      while (!notifiers.empty()) {
        auto iter = notifiers.begin();
        WeakNotifyFn notify_fn = std::move(iter->second);
        notifiers.erase(iter);
        mu.unlock();
        notify_fn();
        mu.lock();
      }
      ptr = nullptr;
    }
    WeakRefCounted* GetNewRef() {
      mutex_lock ml(mu);
      if (ptr != nullptr && ptr->TryRef()) {
        return ptr;
      }
      return nullptr;
    }
    int AddNotifier(WeakNotifyFn notify_fn) {
      mutex_lock ml(mu);
      if (ptr == nullptr) {
        return 0;
      }
      int notifier_id = next_notifier_id++;
      notifiers.emplace(notifier_id, std::move(notify_fn));
      return notifier_id;
    }
    int DupNotifier(int notifier_id) {
      mutex_lock ml(mu);
      auto iter = notifiers.find(notifier_id);
      if (iter != notifiers.end()) {
        int notifier_id = next_notifier_id++;
        notifiers.emplace(notifier_id, iter->second);
        return notifier_id;
      }
      return 0;
    }
    void RemoveNotifier(int notifier_id) {
      mutex_lock ml(mu);
      notifiers.erase(notifier_id);
    }
  };
  mutable RefCountPtr<WeakRefData> data_{new WeakRefData(this)};
  template <typename T>
  friend class WeakPtr;
  friend struct WeakRefData;
};
template <typename T>
class WeakPtr {
 public:
  explicit WeakPtr(WeakRefCounted* ptr = nullptr,
                   WeakNotifyFn notify_fn = nullptr)
      : data_(nullptr), notifier_id_(0) {
    if (ptr != nullptr) {
      ptr->data_->Ref();
      data_.reset(ptr->data_.get());
      if (notify_fn) {
        notifier_id_ = data_->AddNotifier(notify_fn);
      }
    }
  }
  ~WeakPtr() {
    if (data_ != nullptr && notifier_id_ != 0) {
      data_->RemoveNotifier(notifier_id_);
    }
  }
  WeakPtr(const WeakPtr& other) { operator=(other); }
  WeakPtr& operator=(const WeakPtr& other) {
    if (data_ != nullptr && notifier_id_ != 0) {
      data_->RemoveNotifier(notifier_id_);
    }
    other.data_->Ref();
    data_.reset(other.data_.get());
    notifier_id_ = data_->DupNotifier(other.notifier_id_);
    return *this;
  }
  WeakPtr(WeakPtr&& other) noexcept {
    data_ = std::move(other.data_);
    notifier_id_ = other.notifier_id_;
    other.notifier_id_ = 0;
  }
  WeakPtr& operator=(WeakPtr&& other) noexcept {
    if (this != &other) {
      if (data_ != nullptr && notifier_id_ != 0) {
        data_->RemoveNotifier(notifier_id_);
      }
      data_ = std::move(other.data_);
      notifier_id_ = other.notifier_id_;
      other.notifier_id_ = 0;
    }
    return *this;
  }
  RefCountPtr<T> GetNewRef() const {
    RefCountPtr<T> ref;
    if (data_ != nullptr) {
      WeakRefCounted* ptr = data_->GetNewRef();
      ref.reset(static_cast<T*>(ptr));
    }
    return std::move(ref);
  }
 private:
  RefCountPtr<WeakRefCounted::WeakRefData> data_;
  int notifier_id_;
};
inline RefCounted::RefCounted() : ref_(1) {}
inline RefCounted::~RefCounted() {
  DCHECK_EQ(ref_.load(), 0);
}
inline void RefCounted::Ref() const {
  int_fast32_t old_ref = ref_.fetch_add(1, std::memory_order_relaxed);
  DCHECK_GT(old_ref, 0);
}
inline bool RefCounted::TryRef() const {
  int_fast32_t old_ref = ref_.load();
  while (old_ref != 0) {
    if (ref_.compare_exchange_weak(old_ref, old_ref + 1)) {
      return true;
    }
  }
  return false;
}
inline bool RefCounted::Unref() const {
  DCHECK_GT(ref_.load(), 0);
  if (ref_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    NotifyDeleted();
    delete this;
    return true;
  }
  return false;
}
inline int_fast32_t RefCounted::RefCount() const {
  return ref_.load(std::memory_order_acquire);
}
inline void RefCounted::NotifyDeleted() const {}
inline bool RefCounted::RefCountIsOne() const {
  return (ref_.load(std::memory_order_acquire) == 1);
}
}  
}  
#endif  