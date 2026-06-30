#include <string>
#include "absl/base/nullability.h"
#include "absl/base/optimization.h"
#include "absl/strings/str_cat.h"
#include "common/type.h"
#include "google/protobuf/descriptor.h"
namespace cel {
using google::protobuf::EnumDescriptor;
bool IsWellKnownEnumType(absl::Nonnull<const EnumDescriptor*> descriptor) {
  return descriptor->full_name() == "google.protobuf.NullValue";
}
std::string EnumType::DebugString() const {
  if (ABSL_PREDICT_TRUE(static_cast<bool>(*this))) {
    static_assert(sizeof(descriptor_) == 8 || sizeof(descriptor_) == 4,
                  "sizeof(void*) is neither 8 nor 4");
    return absl::StrCat(name(), "@0x",
                        absl::Hex(descriptor_, sizeof(descriptor_) == 8
                                                   ? absl::PadSpec::kZeroPad16
                                                   : absl::PadSpec::kZeroPad8));
  }
  return std::string();
}
}  