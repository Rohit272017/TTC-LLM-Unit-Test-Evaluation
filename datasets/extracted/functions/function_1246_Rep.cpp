#ifndef THIRD_PARTY_CEL_CPP_INTERNAL_COPY_ON_WRITE_H_
#define THIRD_PARTY_CEL_CPP_INTERNAL_COPY_ON_WRITE_H_
#include <algorithm>
#include <atomic>
#include <memory>
#include <type_traits>
#include <utility>
#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/log/absl_check.h"
namespace cel::internal {
template <typename T>
class ABSL_ATTRIBUTE_TRIVIAL_ABI CopyOnWrite final {
 private:
  struct Rep final {
    Rep() = default;
    template <typename... Args,
              typename = std::enable_if_t<std::is_constructible_v<T, Args...>>>
    explicit Rep(Args&&... args) : value(std::forward<Args>(value)...) {}
    Rep(const Rep&) = delete;
    Rep(Rep&&) = delete;
    Rep& operator=(const Rep&) = delete;
    Rep& operator=(Rep&&) = delete;
    std::atomic<int32_t> refs = 1;
    T value;
    void Ref() {
      const auto count = refs.fetch_add(1, std::memory_order_relaxed);
      ABSL_DCHECK_GT(count, 0);
    }
    void Unref() {
      const auto count = refs.fetch_sub(1, std::memory_order_acq_rel);
      ABSL_DCHECK_GT(count, 0);
      if (count == 1) {
        delete this;
      }
    }
    bool Unique() const {
      const auto count = refs.load(std::memory_order_acquire);
      ABSL_DCHECK_GT(count, 0);
      return count == 1;
    }
  };
 public:
  static_assert(std::is_copy_constructible_v<T>,
                "T must be copy constructible");
  static_assert(std::is_destructible_v<T>, "T must be destructible");
  template <typename = std::enable_if_t<std::is_default_constructible_v<T>>>
  CopyOnWrite() : rep_(new Rep()) {}
  CopyOnWrite(const CopyOnWrite<T>& other) : rep_(other.rep_) { rep_->Ref(); }
  CopyOnWrite(CopyOnWrite<T>&& other) noexcept : rep_(other.rep_) {
    other.rep_ = nullptr;
  }
  ~CopyOnWrite() {
    if (rep_ != nullptr) {
      rep_->Unref();
    }
  }
  CopyOnWrite<T>& operator=(const CopyOnWrite<T>& other) {
    ABSL_DCHECK_NE(this, std::addressof(other));
    other.rep_->Ref();
    rep_->Unref();
    rep_ = other.rep_;
    return *this;
  }
  CopyOnWrite<T>& operator=(CopyOnWrite<T>&& other) noexcept {
    ABSL_DCHECK_NE(this, std::addressof(other));
    rep_->Unref();
    rep_ = other.rep_;
    other.rep_ = nullptr;
    return *this;
  }
  T& mutable_get() ABSL_ATTRIBUTE_LIFETIME_BOUND {
    ABSL_DCHECK(rep_ != nullptr) << "Object in moved-from state.";
    if (ABSL_PREDICT_FALSE(!rep_->Unique())) {
      auto* rep = new Rep(static_cast<const T&>(rep_->value));
      rep_->Unref();
      rep_ = rep;
    }
    return rep_->value;
  }
  const T& get() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    ABSL_DCHECK(rep_ != nullptr) << "Object in moved-from state.";
    return rep_->value;
  }
  void swap(CopyOnWrite<T>& other) noexcept {
    using std::swap;
    swap(rep_, other.rep_);
  }
 private:
  Rep* rep_;
};
template <typename T>
void swap(CopyOnWrite<T>& lhs, CopyOnWrite<T>& rhs) noexcept {
  lhs.swap(rhs);
}
}  
#endif  