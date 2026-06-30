#include "common/type.h"
#include "absl/base/nullability.h"
#include "absl/types/span.h"
#include "google/protobuf/arena.h"
namespace cel {
namespace common_internal {
struct TypeTypeData final {
  static TypeTypeData* Create(absl::Nonnull<google::protobuf::Arena*> arena,
                              const Type& type) {
    return google::protobuf::Arena::Create<TypeTypeData>(arena, type);
  }
  explicit TypeTypeData(const Type& type) : type(type) {}
  TypeTypeData() = delete;
  TypeTypeData(const TypeTypeData&) = delete;
  TypeTypeData(TypeTypeData&&) = delete;
  TypeTypeData& operator=(const TypeTypeData&) = delete;
  TypeTypeData& operator=(TypeTypeData&&) = delete;
  const Type type;
};
}  
TypeType::TypeType(absl::Nonnull<google::protobuf::Arena*> arena, const Type& parameter)
    : TypeType(common_internal::TypeTypeData::Create(arena, parameter)) {}
TypeParameters TypeType::GetParameters() const {
  if (data_) {
    return TypeParameters(absl::MakeConstSpan(&data_->type, 1));
  }
  return {};
}
Type TypeType::GetType() const {
  if (data_) {
    return data_->type;
  }
  return Type();
}
}  