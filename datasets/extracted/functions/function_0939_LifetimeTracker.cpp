#ifndef QUICHE_COMMON_LIFETIME_TRACKING_H_
#define QUICHE_COMMON_LIFETIME_TRACKING_H_
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include "absl/strings/str_format.h"
#include "quiche/common/platform/api/quiche_export.h"
#include "quiche/common/platform/api/quiche_logging.h"
#include "quiche/common/platform/api/quiche_stack_trace.h"
namespace quiche {
namespace test {
class LifetimeTrackingTest;
}  
struct QUICHE_EXPORT LifetimeInfo {
  bool IsDead() const { return destructor_stack.has_value(); }
  std::optional<std::vector<void*>> destructor_stack;
};
class QUICHE_EXPORT LifetimeTracker {
 public:
  LifetimeTracker(const LifetimeTracker& other) { CopyFrom(other); }
  LifetimeTracker& operator=(const LifetimeTracker& other) {
    CopyFrom(other);
    return *this;
  }
  LifetimeTracker(LifetimeTracker&& other) { CopyFrom(other); }
  LifetimeTracker& operator=(LifetimeTracker&& other) {
    CopyFrom(other);
    return *this;
  }
  bool IsTrackedObjectDead() const { return info_->IsDead(); }
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const LifetimeTracker& tracker) {
    if (tracker.info_->IsDead()) {
      absl::Format(&sink, "Tracked object has died with %v",
                   SymbolizeStackTrace(*tracker.info_->destructor_stack));
    } else {
      absl::Format(&sink, "Tracked object is alive.");
    }
  }
 private:
  friend class LifetimeTrackable;
  explicit LifetimeTracker(std::shared_ptr<const LifetimeInfo> info)
      : info_(std::move(info)) {
    QUICHE_CHECK(info_ != nullptr)
        << "Passed a null info pointer into the lifetime tracker";
  }
  void CopyFrom(const LifetimeTracker& other) { info_ = other.info_; }
  std::shared_ptr<const LifetimeInfo> info_;
};
class QUICHE_EXPORT LifetimeTrackable {
 public:
  LifetimeTrackable() = default;
  virtual ~LifetimeTrackable() {
    if (info_ != nullptr) {
      info_->destructor_stack = CurrentStackTrace();
    }
  }
  LifetimeTrackable(const LifetimeTrackable&) : LifetimeTrackable() {}
  LifetimeTrackable& operator=(const LifetimeTrackable&) { return *this; }
  LifetimeTrackable(LifetimeTrackable&&) : LifetimeTrackable() {}
  LifetimeTrackable& operator=(LifetimeTrackable&&) { return *this; }
  LifetimeTracker NewTracker() {
    if (info_ == nullptr) {
      info_ = std::make_shared<LifetimeInfo>();
    }
    return LifetimeTracker(info_);
  }
 private:
  friend class test::LifetimeTrackingTest;
  std::shared_ptr<LifetimeInfo> info_;
};
}  
#endif  