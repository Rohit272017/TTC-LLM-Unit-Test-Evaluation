#include "xla/comparison_util.h"
#include <optional>
#include <string>
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/primitive_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"  
namespace xla {
namespace {
bool IsValidComparison(xla::PrimitiveType type, Comparison::Order order) {
  if (primitive_util::IsFloatingPointType(type) ||
      primitive_util::IsComplexType(type)) {
    return true;
  }
  if (primitive_util::IsIntegralType(type) || type == PRED) {
    return order == Comparison::Order::kTotal;
  }
  LOG(FATAL) << "Unsupported type: " << PrimitiveType_Name(type);
}
PrimitiveType DefaultPrimitiveType(Comparison::Type type) {
  switch (type) {
    case Comparison::Type::kFloat:
    case Comparison::Type::kFloatTotalOrder:
      return PrimitiveType::F32;
    case Comparison::Type::kSigned:
      return PrimitiveType::S32;
    case Comparison::Type::kUnsigned:
      return PrimitiveType::U32;
  }
}
Comparison::Order DefaultOrdering(Comparison::Type type) {
  switch (type) {
    case Comparison::Type::kFloat:
      return Comparison::Order::kPartial;
    case Comparison::Type::kFloatTotalOrder:
    case Comparison::Type::kSigned:
    case Comparison::Type::kUnsigned:
      return Comparison::Order::kTotal;
  }
}
Comparison::Order DefaultOrdering(PrimitiveType type) {
  if (primitive_util::IsFloatingPointType(type) ||
      primitive_util::IsComplexType(type)) {
    return Comparison::Order::kPartial;
  }
  if (primitive_util::IsIntegralType(type) || type == PRED) {
    return Comparison::Order::kTotal;
  }
  LOG(FATAL) << "Unsupported type: " << PrimitiveType_Name(type);
}
Comparison::Direction Converse(Comparison::Direction direction) {
  switch (direction) {
    case Comparison::Direction::kEq:
      return Comparison::Direction::kEq;
    case Comparison::Direction::kNe:
      return Comparison::Direction::kNe;
    case Comparison::Direction::kGe:
      return Comparison::Direction::kLe;
    case Comparison::Direction::kGt:
      return Comparison::Direction::kLt;
    case Comparison::Direction::kLe:
      return Comparison::Direction::kGe;
    case Comparison::Direction::kLt:
      return Comparison::Direction::kGt;
  }
}
Comparison::Direction Inverse(Comparison::Direction direction) {
  switch (direction) {
    case Comparison::Direction::kEq:
      return Comparison::Direction::kNe;
    case Comparison::Direction::kNe:
      return Comparison::Direction::kEq;
    case Comparison::Direction::kGe:
      return Comparison::Direction::kLt;
    case Comparison::Direction::kGt:
      return Comparison::Direction::kLe;
    case Comparison::Direction::kLe:
      return Comparison::Direction::kGt;
    case Comparison::Direction::kLt:
      return Comparison::Direction::kGe;
  }
}
}  
std::string ComparisonDirectionToString(Comparison::Direction direction) {
  switch (direction) {
    case Comparison::Direction::kEq:
      return "EQ";
    case Comparison::Direction::kNe:
      return "NE";
    case Comparison::Direction::kGe:
      return "GE";
    case Comparison::Direction::kGt:
      return "GT";
    case Comparison::Direction::kLe:
      return "LE";
    case Comparison::Direction::kLt:
      return "LT";
    default:
      LOG(FATAL) << "Attempted to print uninitialized comparison direction";
  }
}
std::string ComparisonTypeToString(Comparison::Type type) {
  switch (type) {
    case Comparison::Type::kFloat:
      return "FLOAT";
    case Comparison::Type::kFloatTotalOrder:
      return "TOTALORDER";
    case Comparison::Type::kSigned:
      return "SIGNED";
    case Comparison::Type::kUnsigned:
      return "UNSIGNED";
  }
}
absl::string_view ComparisonPrimitiveTypeToString(PrimitiveType type) {
  return PrimitiveType_Name(type);
}
absl::string_view ComparisonOrderToString(Comparison::Order order) {
  switch (order) {
    case Comparison::Order::kPartial:
      return "PARTIALORDER";
    case Comparison::Order::kTotal:
      return "TOTALORDER";
  }
}
absl::StatusOr<Comparison::Direction> StringToComparisonDirection(
    absl::string_view direction) {
  static auto* map =
      new absl::flat_hash_map<std::string, Comparison::Direction>({
          {"EQ", Comparison::Direction::kEq},
          {"NE", Comparison::Direction::kNe},
          {"GE", Comparison::Direction::kGe},
          {"GT", Comparison::Direction::kGt},
          {"LE", Comparison::Direction::kLe},
          {"LT", Comparison::Direction::kLt},
      });
  auto it = map->find(direction);
  if (it == map->end()) {
    return InvalidArgument("Unknown comparison direction: %s", direction);
  }
  return it->second;
}
absl::StatusOr<Comparison::Order> StringToComparisonOrder(
    absl::string_view order) {
  static auto* map = new absl::flat_hash_map<std::string, Comparison::Order>({
      {"TOTALORDER", Comparison::Order::kTotal},
      {"PARTIALORDER", Comparison::Order::kPartial},
  });
  auto it = map->find(order);
  if (it == map->end()) {
    return InvalidArgument("Unknown comparison type: %s", order);
  }
  return it->second;
}
absl::StatusOr<Comparison::Type> StringToComparisonType(
    absl::string_view comparison) {
  static auto* map = new absl::flat_hash_map<std::string, Comparison::Type>({
      {"FLOAT", Comparison::Type::kFloat},
      {"TOTALORDER", Comparison::Type::kFloatTotalOrder},
      {"SIGNED", Comparison::Type::kSigned},
      {"UNSIGNED", Comparison::Type::kUnsigned},
  });
  auto it = map->find(comparison);
  if (it == map->end()) {
    return InvalidArgument("Unknown comparison type: %s", comparison);
  }
  return it->second;
}
Comparison::Type Comparison::DefaultComparisonType(PrimitiveType type) {
  if (primitive_util::IsFloatingPointType(type) ||
      primitive_util::IsComplexType(type)) {
    return Type::kFloat;
  }
  if (primitive_util::IsSignedIntegralType(type)) {
    return Type::kSigned;
  }
  if (primitive_util::IsUnsignedIntegralType(type) || type == PRED) {
    return Type::kUnsigned;
  }
  LOG(FATAL) << "Unexpected: " << PrimitiveType_Name(type);
}
Comparison::Comparison(Direction dir, PrimitiveType type, Order order)
    : dir_(dir),
      primitive_type_(type),
      order_(order),
      type_(DefaultComparisonType(type)) {
  CHECK(IsValidComparison(primitive_type_, order_));
}
Comparison::Comparison(Direction dir, PrimitiveType type)
    : dir_(dir),
      primitive_type_(type),
      order_(DefaultOrdering(type)),
      type_(DefaultComparisonType(type)) {
  CHECK(IsValidComparison(primitive_type_, order_));
}
Comparison::Comparison(Direction dir, Type type)
    : dir_(dir),
      primitive_type_(DefaultPrimitiveType(type)),
      order_(DefaultOrdering(type)),
      type_(type) {
  CHECK(IsValidComparison(primitive_type_, order_));
}
Comparison Comparison::Converse() const {
  return Comparison(xla::Converse(dir_), primitive_type_, order_);
}
std::optional<Comparison> Comparison::Inverse() const {
  if (IsPartialOrder()) {
    return std::nullopt;
  }
  if (primitive_util::IsArrayType(primitive_type_)) {
    return Comparison(xla::Inverse(dir_), primitive_type_, order_);
  }
  return std::nullopt;
}
bool Comparison::IsReflexive() const {
  switch (dir_) {
    case Direction::kEq:
    case Direction::kGe:
    case Direction::kLe:
      return IsTotalOrder();
    case Direction::kNe:
    case Direction::kGt:
    case Direction::kLt:
      return false;
  }
}
bool Comparison::IsAntireflexive() const {
  switch (dir_) {
    case Direction::kNe:
      return IsTotalOrder();
    case Direction::kGt:
    case Direction::kLt:
      return true;
    case Direction::kEq:
    case Direction::kGe:
    case Direction::kLe:
      return false;
  }
}
std::string Comparison::ToString(std::string prefix1, std::string prefix2,
                                 std::string prefix3) const {
  return absl::StrCat(prefix1, ComparisonDirectionToString(dir_), prefix2,
                      ComparisonPrimitiveTypeToString(primitive_type_), prefix3,
                      ComparisonOrderToString(order_));
}
}  