#include "xla/primitive_util.h"
#include <cstdint>
#include <limits>
#include <string>
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"
namespace xla {
namespace primitive_util {
int SignificandWidth(PrimitiveType type) {
  return FloatingPointTypeSwitch<int>(
      [&](auto constant_type) -> int {
        return std::numeric_limits<NativeTypeOf<constant_type>>::digits;
      },
      type);
}
int ExponentWidth(PrimitiveType type) {
  int total_bit_width = BitWidth(type);
  int trailing_significand_field_width = SignificandWidth(type) - 1;
  int kSignBitWidth = 1;
  return total_bit_width - (trailing_significand_field_width + kSignBitWidth);
}
int UnderflowExponent(PrimitiveType type) {
  return FloatingPointTypeSwitch<int>(
      [&](auto constant_type) -> int {
        return std::numeric_limits<NativeTypeOf<constant_type>>::min_exponent;
      },
      type);
}
int OverflowExponent(PrimitiveType type) {
  return FloatingPointTypeSwitch<int>(
      [&](auto constant_type) -> int {
        return std::numeric_limits<NativeTypeOf<constant_type>>::max_exponent;
      },
      type);
}
int ExponentBias(PrimitiveType type) {
  return (1 - UnderflowExponent(type)) + 1;
}
bool HasInfinity(PrimitiveType type) {
  if (ABSL_PREDICT_TRUE(IsFloatingPointType(type))) {
    return FloatingPointTypeSwitch<bool>(
        [&](auto constant_type) -> bool {
          return std::numeric_limits<NativeTypeOf<constant_type>>::has_infinity;
        },
        type);
  }
  return false;
}
bool HasNegativeZero(PrimitiveType type) {
  if (ABSL_PREDICT_TRUE(IsFloatingPointType(type))) {
    return FloatingPointTypeSwitch<bool>(
        [&](auto constant_type) -> bool {
          return has_negative_zero_v<NativeTypeOf<constant_type>>;
        },
        type);
  }
  return false;
}
xla::PrimitiveType SignedIntegralTypeForBitWidth(int64_t src_bitwidth) {
  switch (src_bitwidth) {
    case 2:
      return xla::S2;
    case 4:
      return xla::S4;
    case 8:
      return xla::S8;
    case 16:
      return xla::S16;
    case 32:
      return xla::S32;
    case 64:
      return xla::S64;
    default:
      return xla::PRIMITIVE_TYPE_INVALID;
  }
}
class PrimitiveTypeNameGenerator {
 public:
  PrimitiveTypeNameGenerator() {
    for (int i = 0; i < PrimitiveType_ARRAYSIZE; i++) {
      if (i == static_cast<int>(OPAQUE_TYPE)) {
        lowercase_name_[i] = "opaque";
      } else if (PrimitiveType_IsValid(i)) {
        lowercase_name_[i] = absl::AsciiStrToLower(
            PrimitiveType_Name(static_cast<PrimitiveType>(i)));
      }
    }
  }
  const std::string& LowercaseName(PrimitiveType t) {
    CHECK_LT(t, PrimitiveType_ARRAYSIZE);
    return lowercase_name_[static_cast<int>(t)];
  }
 private:
  std::string lowercase_name_[PrimitiveType_ARRAYSIZE];
};
const std::string& LowercasePrimitiveTypeName(PrimitiveType s) {
  static auto* gen = new PrimitiveTypeNameGenerator();
  return gen->LowercaseName(s);
}
namespace {
const absl::flat_hash_map<std::string, PrimitiveType>&
GetPrimitiveTypeStringMap() {
  static absl::flat_hash_map<std::string, PrimitiveType>* name_to_type = [] {
    static auto* map = new absl::flat_hash_map<std::string, PrimitiveType>;
    for (int i = 0; i < PrimitiveType_ARRAYSIZE; i++) {
      if (PrimitiveType_IsValid(i) && i != PRIMITIVE_TYPE_INVALID) {
        auto value = static_cast<PrimitiveType>(i);
        (*map)[LowercasePrimitiveTypeName(value)] = value;
      }
    }
    (*map)["opaque"] = OPAQUE_TYPE;
    return map;
  }();
  return *name_to_type;
}
}  
absl::StatusOr<PrimitiveType> StringToPrimitiveType(absl::string_view name) {
  const auto& map = GetPrimitiveTypeStringMap();
  auto found = map.find(name);
  if (found == map.end()) {
    return InvalidArgument("Invalid element type string: \"%s\".", name);
  }
  return found->second;
}
bool IsPrimitiveTypeName(absl::string_view name) {
  const auto& map = GetPrimitiveTypeStringMap();
  auto found = map.find(name);
  return found != map.end();
}
}  
}  