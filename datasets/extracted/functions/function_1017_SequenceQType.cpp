#include "arolla/sequence/sequence_qtype.h"
#include <memory>
#include <string>
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "arolla/qtype/qtype.h"
#include "arolla/qtype/simple_qtype.h"
#include "arolla/sequence/sequence.h"
#include "arolla/util/fast_dynamic_downcast_final.h"
#include "arolla/util/meta.h"
namespace arolla {
namespace {
class SequenceQType final : public SimpleQType {
 public:
  explicit SequenceQType(QTypePtr value_qtype)
      : SimpleQType(meta::type<Sequence>(),
                    "SEQUENCE[" + std::string(value_qtype->name()) + "]",
                    value_qtype,
                    "::arolla::SequenceQType") {}
};
class SequenceQTypeRegistry {
 public:
  QTypePtr GetSequenceQType(QTypePtr value_qtype) {
    absl::WriterMutexLock l(&lock_);
    auto& result = registry_[value_qtype];
    if (!result) {
      result = std::make_unique<SequenceQType>(value_qtype);
    }
    return result.get();
  }
 private:
  absl::Mutex lock_;
  absl::flat_hash_map<QTypePtr, std::unique_ptr<SequenceQType>> registry_
      ABSL_GUARDED_BY(lock_);
};
}  
bool IsSequenceQType(const QType* qtype) {
  return fast_dynamic_downcast_final<const SequenceQType*>(qtype) != nullptr;
}
QTypePtr GetSequenceQType(QTypePtr value_qtype) {
  static absl::NoDestructor<SequenceQTypeRegistry> registry;
  return registry->GetSequenceQType(value_qtype);
}
}  