#include "arolla/qtype/standard_type_properties/properties.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "arolla/qtype/array_like/array_like_qtype.h"
#include "arolla/qtype/base_types.h"
#include "arolla/qtype/optional_qtype.h"
#include "arolla/qtype/qtype.h"
#include "arolla/qtype/qtype_traits.h"
#include "arolla/qtype/shape_qtype.h"
namespace arolla {
const QType*  GetScalarQTypeOrNull(
    const QType*  qtype) {
  if (qtype != nullptr) {
    if (auto* value_qtype = qtype->value_qtype()) {
      return value_qtype;
    }
    if (IsScalarQType(qtype)) {
      return qtype;
    }
  }
  return nullptr;
}
absl::StatusOr<QTypePtr> GetScalarQType(QTypePtr qtype) {
  DCHECK(qtype);
  if (auto* result = GetScalarQTypeOrNull(qtype)) {
    return result;
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "there is no corresponding scalar type for %s", qtype->name()));
}
const ShapeQType*  GetShapeQTypeOrNull(
    const QType*  qtype) {
  if (qtype != nullptr) {
    if (qtype->value_qtype() == nullptr) {
      if (IsScalarQType(qtype)) {
        return static_cast<const ShapeQType*>(GetQType<ScalarShape>());
      }
    } else {
      if (IsOptionalQType(qtype)) {
        return static_cast<const ShapeQType*>(GetQType<OptionalScalarShape>());
      }
      if (auto* array_qtype = dynamic_cast<const ArrayLikeQType*>(qtype)) {
        return array_qtype->shape_qtype();
      }
    }
  }
  return nullptr;
}
absl::StatusOr<const ShapeQType*> GetShapeQType(QTypePtr qtype) {
  DCHECK(qtype);
  if (auto* result = GetShapeQTypeOrNull(qtype)) {
    return result;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("no shape type for %s", qtype->name()));
}
QTypePtr DecayContainerQType(QTypePtr qtype) {
  DCHECK(qtype);
  auto* value_qtype = qtype->value_qtype();
  if (value_qtype != nullptr) {
    return value_qtype;
  }
  return qtype;
}
absl::StatusOr<QTypePtr> WithScalarQType(QTypePtr qtype,
                                         QTypePtr new_scalar_qtype) {
  DCHECK(qtype);
  DCHECK(new_scalar_qtype);
  if (!IsScalarQType(new_scalar_qtype)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "unable to replace scalar type in %s with a non-scalar type %s",
        qtype->name(), new_scalar_qtype->name()));
  }
  if (auto shape_qtype = GetShapeQType(qtype); shape_qtype.ok()) {
    return (**shape_qtype).WithValueQType(new_scalar_qtype);
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("unable to replace scalar type in %s", qtype->name()));
}
absl::StatusOr<QTypePtr> GetPresenceQType(QTypePtr qtype) {
  DCHECK(qtype);
  if (auto shape_qtype = GetShapeQType(qtype); shape_qtype.ok()) {
    return (**shape_qtype).presence_qtype();
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("no type to represent presence in %s", qtype->name()));
}
bool IsOptionalLikeQType(const QType* qtype) {
  return qtype != nullptr && qtype->value_qtype() != nullptr &&
         (IsOptionalQType(qtype) || IsArrayLikeQType(qtype));
}
absl::StatusOr<QTypePtr> ToOptionalLikeQType(QTypePtr qtype) {
  DCHECK(qtype);
  if (qtype->value_qtype() == nullptr) {
    if (IsScalarQType(qtype)) {
      return ToOptionalQType(qtype);
    }
  } else if (IsOptionalLikeQType(qtype)) {
    return qtype;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("no optional-like qtype for %s", qtype->name()));
}
}  