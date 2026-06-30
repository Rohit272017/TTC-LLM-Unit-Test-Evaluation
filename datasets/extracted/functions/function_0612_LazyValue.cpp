#include "arolla/lazy/lazy.h"
#include <memory>
#include <utility>
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "arolla/qtype/qtype.h"
#include "arolla/qtype/typed_value.h"
#include "arolla/util/fingerprint.h"
#include "arolla/util/repr.h"
namespace arolla {
namespace {
class LazyValue final : public Lazy {
 public:
  explicit LazyValue(TypedValue&& value)
      : Lazy(value.GetType(), FingerprintHasher("::arolla::LazyValue")
                                  .Combine(value.GetFingerprint())
                                  .Finish()),
        value_(std::move(value)) {}
  absl::StatusOr<TypedValue> Get() const final { return value_; }
 private:
  TypedValue value_;
};
class LazyCallable final : public Lazy {
 public:
  using Callable = absl::AnyInvocable<absl::StatusOr<TypedValue>() const>;
  explicit LazyCallable(QTypePtr value_qtype, Callable&& callable)
      : Lazy(value_qtype, RandomFingerprint()),
        callable_(std::move(callable)) {}
  absl::StatusOr<TypedValue> Get() const final {
    auto result = callable_();
    if (result.ok() && result->GetType() != value_qtype()) {
      return absl::FailedPreconditionError(
          absl::StrFormat("expected a lazy callable to return %s, got %s",
                          value_qtype()->name(), result->GetType()->name()));
    }
    return result;
  }
 private:
  Callable callable_;
};
}  
LazyPtr MakeLazyFromQValue(TypedValue value) {
  return std::make_shared<LazyValue>(std::move(value));
}
LazyPtr MakeLazyFromCallable(QTypePtr value_qtype,
                             LazyCallable::Callable callable) {
  return std::make_shared<LazyCallable>(value_qtype, std::move(callable));
}
void FingerprintHasherTraits<LazyPtr>::operator()(FingerprintHasher* hasher,
                                                  const LazyPtr& value) const {
  if (value != nullptr) {
    hasher->Combine(value->fingerprint());
  }
}
ReprToken ReprTraits<LazyPtr>::operator()(const LazyPtr& value) const {
  if (value == nullptr) {
    return ReprToken{"lazy[?]{nullptr}"};
  }
  return ReprToken{absl::StrCat("lazy[", value->value_qtype()->name(), "]")};
}
}  