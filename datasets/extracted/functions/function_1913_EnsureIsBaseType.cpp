#include "arolla/qtype/shape_qtype.h"
#include <string>
#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "arolla/qtype/base_types.h"
#include "arolla/qtype/optional_qtype.h"
#include "arolla/qtype/qtype.h"
#include "arolla/qtype/qtype_traits.h"
#include "arolla/util/fingerprint.h"
#include "arolla/util/meta.h"
#include "arolla/util/repr.h"
#include "arolla/util/unit.h"
#include "arolla/util/status_macros_backport.h"
namespace arolla {
namespace {
absl::Status EnsureIsBaseType(QTypePtr qtype) {
  return IsScalarQType(qtype) || IsOptionalQType(qtype)
             ? absl::OkStatus()
             : absl::InvalidArgumentError(absl::StrFormat(
                   "Shape::WithValueQType supports only scalar and "
                   "optional values, got %s",
                   qtype->name()));
}
class ScalarShapeQType final : public ShapeQType {
 public:
  ScalarShapeQType() : ShapeQType(meta::type<ScalarShape>(), "SCALAR_SHAPE") {}
  absl::StatusOr<QTypePtr> WithValueQType(QTypePtr value_qtype) const final {
    RETURN_IF_ERROR(EnsureIsBaseType(value_qtype));
    return value_qtype;
  }
  QTypePtr presence_qtype() const final { return GetQType<Unit>(); }
};
class OptionalScalarShapeQType final : public ShapeQType {
 public:
  OptionalScalarShapeQType()
      : ShapeQType(meta::type<OptionalScalarShape>(), "OPTIONAL_SCALAR_SHAPE") {
  }
  absl::StatusOr<QTypePtr> WithValueQType(QTypePtr value_qtype) const final {
    RETURN_IF_ERROR(EnsureIsBaseType(value_qtype));
    return ToOptionalQType(value_qtype);
  }
  QTypePtr presence_qtype() const final { return GetOptionalQType<Unit>(); }
};
}  
QTypePtr QTypeTraits<ScalarShape>::type() {
  static const absl::NoDestructor<ScalarShapeQType> shape_qtype;
  return shape_qtype.get();
}
QTypePtr QTypeTraits<OptionalScalarShape>::type() {
  static const absl::NoDestructor<OptionalScalarShapeQType> shape_qtype;
  return shape_qtype.get();
}
ReprToken ReprTraits<ScalarShape>::operator()(
    const ScalarShape& ) const {
  return ReprToken{"scalar_shape"};
}
void FingerprintHasherTraits<ScalarShape>::operator()(
    FingerprintHasher* hasher, const ScalarShape& ) const {
  hasher->Combine(absl::string_view("scalar_shape"));
}
ReprToken ReprTraits<OptionalScalarShape>::operator()(
    const OptionalScalarShape& ) const {
  return ReprToken{"optional_scalar_shape"};
}
void FingerprintHasherTraits<OptionalScalarShape>::operator()(
    FingerprintHasher* hasher, const OptionalScalarShape& ) const {
  hasher->Combine(absl::string_view("optional_scalar_shape"));
}
}  