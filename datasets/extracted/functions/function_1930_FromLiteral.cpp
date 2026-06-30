#include "xla/service/constant_value.h"
#include <string>
namespace xla {
absl::StatusOr<ConstantValue> ConstantValue::FromLiteral(
    const Literal& literal) {
  CHECK_EQ(literal.shape().dimensions_size(), 0) << "Expected scalar literal";
  return primitive_util::PrimitiveTypeSwitch<absl::StatusOr<ConstantValue>>(
      [&](auto primitive_type_constant) -> absl::StatusOr<ConstantValue> {
        if constexpr (primitive_util::IsIntegralType(primitive_type_constant)) {
          return ConstantValue(
              static_cast<uint64_t>(
                  literal.GetFirstElement<
                      primitive_util::NativeTypeOf<primitive_type_constant>>()),
              primitive_util::BitWidth(primitive_type_constant),
              primitive_util::IsSignedIntegralType(primitive_type_constant));
        }
        return InvalidArgument("Unsupported type");
      },
      literal.shape().element_type());
}
ConstantValue ConstantValue::div(const ConstantValue& other) const {
  if (!is_signed_) {
    return ConstantValue(value_ / other.value_, bitwidth_, is_signed_);
  }
  return ConstantValue(
      absl::bit_cast<uint64_t>(absl::bit_cast<int64_t>(value_) /
                               absl::bit_cast<int64_t>(other.value_)),
      bitwidth_, is_signed_);
}
ConstantValue ConstantValue::mod(const ConstantValue& other) const {
  if (!is_signed_) {
    return ConstantValue(value_ % other.value_, bitwidth_, is_signed_);
  }
  return ConstantValue(
      absl::bit_cast<uint64_t>(absl::bit_cast<int64_t>(value_) %
                               absl::bit_cast<int64_t>(other.value_)),
      bitwidth_, is_signed_);
}
ConstantValue ConstantValue::mul(const ConstantValue& other) const {
  if (!is_signed_) {
    return ConstantValue(value_ * other.value_, bitwidth_, is_signed_);
  }
  return ConstantValue(
      absl::bit_cast<uint64_t>(absl::bit_cast<int64_t>(value_) *
                               absl::bit_cast<int64_t>(other.value_)),
      bitwidth_, is_signed_);
}
bool ConstantValue::lt(const ConstantValue& other) const {
  if (!is_signed_) {
    return value_ < other.value_;
  }
  return absl::bit_cast<int64_t>(value_) <
         absl::bit_cast<int64_t>(other.value_);
}
bool ConstantValue::gt(const ConstantValue& other) const {
  if (!is_signed_) {
    return value_ > other.value_;
  }
  return absl::bit_cast<int64_t>(value_) >
         absl::bit_cast<int64_t>(other.value_);
}
std::string ConstantValue::ToString() const {
  return is_signed_ ? absl::StrCat(GetSignedValue())
                    : absl::StrCat(GetUnsignedValue());
}
}  