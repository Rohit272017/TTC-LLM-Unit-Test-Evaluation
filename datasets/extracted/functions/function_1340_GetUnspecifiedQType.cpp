#include "arolla/qtype/unspecified_qtype.h"
#include "absl/base/no_destructor.h"
#include "absl/strings/string_view.h"
#include "arolla/memory/frame.h"
#include "arolla/qtype/qtype.h"
#include "arolla/qtype/typed_value.h"
#include "arolla/util/fingerprint.h"
#include "arolla/util/repr.h"
namespace arolla {
namespace {
struct Unspecified {};
class UnspecifiedQType final : public QType {
 public:
  UnspecifiedQType()
      : QType(ConstructorArgs{.name = "UNSPECIFIED",
                              .type_info = typeid(Unspecified),
                              .type_layout = MakeTypeLayout<Unspecified>()}) {}
  ReprToken UnsafeReprToken(const void* source) const override {
    return ReprToken{"unspecified"};
  }
  void UnsafeCopy(const void* ,
                  void* ) const override {}
  void UnsafeCombineToFingerprintHasher(
      const void* , FingerprintHasher* hasher) const override {
    hasher->Combine(absl::string_view("::arolla::UnspecifiedQValue"));
  }
};
}  
QTypePtr GetUnspecifiedQType() {
  static const absl::NoDestructor<UnspecifiedQType> result;
  return result.get();
}
const TypedValue& GetUnspecifiedQValue() {
  static const absl::NoDestructor<TypedValue> result(
      TypedValue::UnsafeFromTypeDefaultConstructed(GetUnspecifiedQType()));
  return *result;
}
}  