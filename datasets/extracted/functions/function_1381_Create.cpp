#include <string>
#include "absl/base/attributes.h"
#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/strings/str_cat.h"
#include "common/type.h"
#include "google/protobuf/arena.h"
#include "google/protobuf/descriptor.h"
namespace cel {
namespace common_internal {
namespace {
ABSL_CONST_INIT const ListTypeData kDynListTypeData;
}  
absl::Nonnull<ListTypeData*> ListTypeData::Create(
    absl::Nonnull<google::protobuf::Arena*> arena, const Type& element) {
  return ::new (arena->AllocateAligned(
      sizeof(ListTypeData), alignof(ListTypeData))) ListTypeData(element);
}
ListTypeData::ListTypeData(const Type& element) : element(element) {}
}  
ListType::ListType() : ListType(&common_internal::kDynListTypeData) {}
ListType::ListType(absl::Nonnull<google::protobuf::Arena*> arena, const Type& element)
    : ListType(element.IsDyn()
                   ? &common_internal::kDynListTypeData
                   : common_internal::ListTypeData::Create(arena, element)) {}
std::string ListType::DebugString() const {
  return absl::StrCat("list<", element().DebugString(), ">");
}
TypeParameters ListType::GetParameters() const {
  return TypeParameters(GetElement());
}
Type ListType::GetElement() const {
  ABSL_DCHECK_NE(data_, 0);
  if ((data_ & kBasicBit) == kBasicBit) {
    return reinterpret_cast<const common_internal::ListTypeData*>(data_ &
                                                                  kPointerMask)
        ->element;
  }
  if ((data_ & kProtoBit) == kProtoBit) {
    return common_internal::SingularMessageFieldType(
        reinterpret_cast<const google::protobuf::FieldDescriptor*>(data_ & kPointerMask));
  }
  return Type();
}
Type ListType::element() const { return GetElement(); }
}  