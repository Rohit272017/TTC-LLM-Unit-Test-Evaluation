#include "runtime/internal/mutable_list_impl.h"
#include <memory>
#include <string>
#include <utility>
#include "common/native_type.h"
#include "common/type.h"
#include "common/value.h"
namespace cel::runtime_internal {
using ::cel::NativeTypeId;
MutableListValue::MutableListValue(
    cel::Unique<cel::ListValueBuilder> list_builder)
    : cel::OpaqueValueInterface(), list_builder_(std::move(list_builder)) {}
absl::Status MutableListValue::Append(cel::Value element) {
  return list_builder_->Add(std::move(element));
}
absl::StatusOr<cel::ListValue> MutableListValue::Build() && {
  return std::move(*list_builder_).Build();
}
std::string MutableListValue::DebugString() const {
  return kMutableListTypeName;
}
NativeTypeId MutableListValue::GetNativeTypeId() const {
  return cel::NativeTypeId::For<MutableListValue>();
}
}  