#include "arolla/lazy/lazy_qtype.h"
#include <memory>
#include <string>
#include <utility>
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "arolla/lazy/lazy.h"
#include "arolla/qtype/qtype.h"
#include "arolla/qtype/simple_qtype.h"
#include "arolla/qtype/typed_value.h"
#include "arolla/util/fast_dynamic_downcast_final.h"
#include "arolla/util/meta.h"
namespace arolla {
namespace {
class LazyQType final : public SimpleQType {
 public:
  explicit LazyQType(QTypePtr value_qtype)
      : SimpleQType(meta::type<LazyPtr>(),
                    "LAZY[" + std::string(value_qtype->name()) + "]",
                    value_qtype,
                    "::arolla::LazyQType") {}
};
class LazyQTypeRegistry {
 public:
  QTypePtr GetLazyQType(QTypePtr value_qtype) {
    absl::WriterMutexLock l(&lock_);
    auto& result = registry_[value_qtype];
    if (!result) {
      result = std::make_unique<LazyQType>(value_qtype);
    }
    return result.get();
  }
 private:
  absl::Mutex lock_;
  absl::flat_hash_map<QTypePtr, std::unique_ptr<LazyQType>> registry_
      ABSL_GUARDED_BY(lock_);
};
}  
bool IsLazyQType(const QType* qtype) {
  return fast_dynamic_downcast_final<const LazyQType*>(qtype) != nullptr;
}
QTypePtr GetLazyQType(QTypePtr value_qtype) {
  static absl::NoDestructor<LazyQTypeRegistry> registry;
  return registry->GetLazyQType(value_qtype);
}
TypedValue MakeLazyQValue(LazyPtr lazy) {
  DCHECK_NE(lazy, nullptr);
  auto result = TypedValue::FromValueWithQType(
      std::move(lazy), GetLazyQType(lazy->value_qtype()));
  DCHECK_OK(result.status());
  return *std::move(result);
}
}  