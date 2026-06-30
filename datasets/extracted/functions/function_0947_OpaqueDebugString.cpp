#include <cstddef>
#include <cstring>
#include <string>
#include <type_traits>
#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "absl/utility/utility.h"
#include "common/type.h"
#include "google/protobuf/arena.h"
namespace cel {
namespace {
std::string OpaqueDebugString(absl::string_view name,
                              absl::Span<const Type> parameters) {
  if (parameters.empty()) {
    return std::string(name);
  }
  return absl::StrCat(
      name, "<", absl::StrJoin(parameters, ", ", absl::StreamFormatter()), ">");
}
}  
namespace common_internal {
absl::Nonnull<OpaqueTypeData*> OpaqueTypeData::Create(
    absl::Nonnull<google::protobuf::Arena*> arena, absl::string_view name,
    absl::Span<const Type> parameters) {
  return ::new (arena->AllocateAligned(
      offsetof(OpaqueTypeData, parameters) + (parameters.size() * sizeof(Type)),
      alignof(OpaqueTypeData))) OpaqueTypeData(name, parameters);
}
OpaqueTypeData::OpaqueTypeData(absl::string_view name,
                               absl::Span<const Type> parameters)
    : name(name), parameters_size(parameters.size()) {
  std::memcpy(this->parameters, parameters.data(),
              parameters_size * sizeof(Type));
}
}  
OpaqueType::OpaqueType(absl::Nonnull<google::protobuf::Arena*> arena,
                       absl::string_view name,
                       absl::Span<const Type> parameters)
    : OpaqueType(
          common_internal::OpaqueTypeData::Create(arena, name, parameters)) {}
std::string OpaqueType::DebugString() const {
  ABSL_DCHECK(*this);
  return OpaqueDebugString(name(), GetParameters());
}
absl::string_view OpaqueType::name() const {
  ABSL_DCHECK(*this);
  return data_->name;
}
TypeParameters OpaqueType::GetParameters() const {
  ABSL_DCHECK(*this);
  return TypeParameters(
      absl::MakeConstSpan(data_->parameters, data_->parameters_size));
}
bool OpaqueType::IsOptional() const {
  return name() == OptionalType::kName && GetParameters().size() == 1;
}
absl::optional<OptionalType> OpaqueType::AsOptional() const {
  if (IsOptional()) {
    return OptionalType(absl::in_place, *this);
  }
  return absl::nullopt;
}
OptionalType OpaqueType::GetOptional() const {
  ABSL_DCHECK(IsOptional()) << DebugString();
  return OptionalType(absl::in_place, *this);
}
}  