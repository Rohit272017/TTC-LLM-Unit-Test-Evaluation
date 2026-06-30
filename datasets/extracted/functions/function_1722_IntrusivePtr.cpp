#ifndef TENSORFLOW_TSL_PLATFORM_INTRUSIVE_PTR_H_
#define TENSORFLOW_TSL_PLATFORM_INTRUSIVE_PTR_H_
#include <algorithm>
namespace tsl {
namespace core {
template <class T>
class IntrusivePtr {
 public:
  IntrusivePtr(T* h, bool add_ref) { reset(h, add_ref); }
  IntrusivePtr(const IntrusivePtr& o) { reset(o.handle_, true); }
  IntrusivePtr(IntrusivePtr&& o) noexcept { *this = std::move(o); }
  IntrusivePtr() {}
  void reset(T* h, bool add_ref) {
    if (h != handle_) {
      if (add_ref && h) h->Ref();
      if (handle_) handle_->Unref();
      handle_ = h;
    }
  }
  IntrusivePtr& operator=(const IntrusivePtr& o) {
    reset(o.handle_, true);
    return *this;
  }
  IntrusivePtr& operator=(IntrusivePtr&& o) noexcept {
    if (handle_ != o.handle_) {
      reset(o.detach(), false);
    }
    return *this;
  }
  bool operator==(const IntrusivePtr& o) const { return handle_ == o.handle_; }
  T* operator->() const { return handle_; }
  T& operator*() const { return *handle_; }
  explicit operator bool() const noexcept { return get(); }
  T* get() const { return handle_; }
  T* detach() {
    T* handle = handle_;
    handle_ = nullptr;
    return handle;
  }
  ~IntrusivePtr() {
    if (handle_) handle_->Unref();
  }
 private:
  T* handle_ = nullptr;
};
}  
}  
#endif  