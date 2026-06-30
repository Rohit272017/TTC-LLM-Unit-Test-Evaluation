#include "xla/python/ifrt/array_spec.h"
#include <string>
#include <utility>
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/python/ifrt/array_spec.pb.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace ifrt {
absl::StatusOr<ArraySpec> ArraySpec::FromProto(
    DeviceList::LookupDeviceFunc lookup_device, const ArraySpecProto& proto) {
  TF_ASSIGN_OR_RETURN(auto dtype, DType::FromProto(proto.dtype()));
  TF_ASSIGN_OR_RETURN(auto shape, Shape::FromProto(proto.shape()));
  TF_ASSIGN_OR_RETURN(auto sharding,
                      Sharding::FromProto(lookup_device, proto.sharding()));
  return ArraySpec{dtype, std::move(shape),
                   std::move(sharding)};
}
absl::StatusOr<ArraySpecProto> ArraySpec::ToProto() const {
  ArraySpecProto proto;
  *proto.mutable_dtype() = dtype.ToProto();
  *proto.mutable_shape() = shape.ToProto();
  TF_ASSIGN_OR_RETURN(*proto.mutable_sharding(), sharding->ToProto());
  return proto;
}
std::string ArraySpec::DebugString() const {
  return absl::StrCat("ArraySpec(dtype=", dtype.DebugString(),
                      ",shape=", shape.DebugString(),
                      ",sharding=", sharding->DebugString(), ")");
}
}  
}  