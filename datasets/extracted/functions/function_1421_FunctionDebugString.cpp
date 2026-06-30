#include <cstddef>
#include <cstring>
#include <string>
#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "common/type.h"
#include "google/protobuf/arena.h"
namespace cel {
namespace {
struct TypeFormatter {
  void operator()(std::string* out, const Type& type) const {
    out->append(type.DebugString());
  }
};
std::string FunctionDebugString(const Type& result,
                                absl::Span<const Type> args) {
  return absl::StrCat("(", absl::StrJoin(args, ", ", TypeFormatter{}), ") -> ",
                      result.DebugString());
}
}  
namespace common_internal {
absl::Nonnull<FunctionTypeData*> FunctionTypeData::Create(
    absl::Nonnull<google::protobuf::Arena*> arena, const Type& result,
    absl::Span<const Type> args) {
  return ::new (arena->AllocateAligned(
      offsetof(FunctionTypeData, args) + ((1 + args.size()) * sizeof(Type)),
      alignof(FunctionTypeData))) FunctionTypeData(result, args);
}
FunctionTypeData::FunctionTypeData(const Type& result,
                                   absl::Span<const Type> args)
    : args_size(1 + args.size()) {
  this->args[0] = result;
  std::memcpy(this->args + 1, args.data(), args.size() * sizeof(Type));
}
}  
FunctionType::FunctionType(absl::Nonnull<google::protobuf::Arena*> arena,
                           const Type& result, absl::Span<const Type> args)
    : FunctionType(
          common_internal::FunctionTypeData::Create(arena, result, args)) {}
std::string FunctionType::DebugString() const {
  return FunctionDebugString(result(), args());
}
TypeParameters FunctionType::GetParameters() const {
  ABSL_DCHECK(*this);
  return TypeParameters(absl::MakeConstSpan(data_->args, data_->args_size));
}
const Type& FunctionType::result() const {
  ABSL_DCHECK(*this);
  return data_->args[0];
}
absl::Span<const Type> FunctionType::args() const {
  ABSL_DCHECK(*this);
  return absl::MakeConstSpan(data_->args + 1, data_->args_size - 1);
}
}  