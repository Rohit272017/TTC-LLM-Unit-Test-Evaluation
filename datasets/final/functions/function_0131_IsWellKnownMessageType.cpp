#include <string>
#include "absl/base/attributes.h"
#include "absl/base/nullability.h"
#include "absl/base/optimization.h"
#include "absl/log/absl_check.h"
#include "absl/strings/str_cat.h"
#include "common/type.h"
#include "google/protobuf/descriptor.h"
namespace cel {
using google::protobuf::Descriptor;
bool IsWellKnownMessageType(absl::Nonnull<const Descriptor*> descriptor) {
  switch (descriptor->well_known_type()) {
    case Descriptor::WELLKNOWNTYPE_BOOLVALUE:
      ABSL_FALLTHROUGH_INTENDED;
    case Descriptor::WELLKNOWNTYPE_INT32VALUE:
      ABSL_FALLTHROUGH_INTENDED;
    case Descriptor::WELLKNOWNTYPE_INT64VALUE:
      ABSL_FALLTHROUGH_INTENDED;
    case Descriptor::WELLKNOWNTYPE_UINT32VALUE:
      ABSL_FALLTHROUGH_INTENDED;
    case Descriptor::WELLKNOWNTYPE_UINT64VALUE:
      ABSL_FALLTHROUGH_INTENDED;
    case Descriptor::WELLKNOWNTYPE_FLOATVALUE:
      ABSL_FALLTHROUGH_INTENDED;
    case Descriptor::WELLKNOWNTYPE_DOUBLEVALUE:
      ABSL_FALLTHROUGH_INTENDED;
    case Descriptor::WELLKNOWNTYPE_BYTESVALUE:
      ABSL_FALLTHROUGH_INTENDED;
    case Descriptor::WELLKNOWNTYPE_STRINGVALUE:
      ABSL_FALLTHROUGH_INTENDED;
    case Descriptor::WELLKNOWNTYPE_ANY:
      ABSL_FALLTHROUGH_INTENDED;
    case Descriptor::WELLKNOWNTYPE_DURATION:
      ABSL_FALLTHROUGH_INTENDED;
    case Descriptor::WELLKNOWNTYPE_TIMESTAMP:
      ABSL_FALLTHROUGH_INTENDED;
    case Descriptor::WELLKNOWNTYPE_VALUE:
      ABSL_FALLTHROUGH_INTENDED;
    case Descriptor::WELLKNOWNTYPE_LISTVALUE:
      ABSL_FALLTHROUGH_INTENDED;
    case Descriptor::WELLKNOWNTYPE_STRUCT:
      return true;
    default:
      return false;
  }
}
std::string MessageType::DebugString() const {
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
std::string MessageTypeField::DebugString() const {
  if (ABSL_PREDICT_TRUE(static_cast<bool>(*this))) {
    static_assert(sizeof(descriptor_) == 8 || sizeof(descriptor_) == 4,
                  "sizeof(void*) is neither 8 nor 4");
    return absl::StrCat("[", (*this)->number(), "]", (*this)->name(), "@0x",
                        absl::Hex(descriptor_, sizeof(descriptor_) == 8
                                                   ? absl::PadSpec::kZeroPad16
                                                   : absl::PadSpec::kZeroPad8));
  }
  return std::string();
}
Type MessageTypeField::GetType() const {
  ABSL_DCHECK(*this);
  return Type::Field(descriptor_);
}
}  