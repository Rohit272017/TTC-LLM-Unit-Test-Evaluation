#include "tensorflow/core/kernels/composite_tensor_variant.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/composite_tensor_variant.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"
namespace tensorflow {
constexpr const char CompositeTensorVariant::kTypeName[];
CompositeTensorVariant::CompositeTensorVariant(
    const CompositeTensorVariantMetadata& metadata,
    absl::Span<Tensor> flat_components)
    : flat_components_(flat_components.begin(), flat_components.end()),
      metadata_(new CompositeTensorVariantMetadata()) {
  *metadata_ = metadata;
}
CompositeTensorVariant::CompositeTensorVariant()
    : metadata_(new CompositeTensorVariantMetadata()) {}
CompositeTensorVariant::CompositeTensorVariant(
    const CompositeTensorVariant& other)
    : flat_components_(other.flat_components_),
      metadata_(new CompositeTensorVariantMetadata()) {
  *metadata_ = *other.metadata_;
}
void CompositeTensorVariant::Encode(VariantTensorData* data) const {
  data->set_type_name(TypeName());
  metadata_->SerializeToString(&data->metadata_string());
  for (const Tensor& tensor : flat_components_) {
    data->add_tensor(tensor);
  }
}
bool CompositeTensorVariant::Decode(const VariantTensorData& data) {
  if (!metadata_->ParseFromString(data.metadata_string())) {
    return false;
  }
  flat_components_ = data.tensors();
  return true;
}
string CompositeTensorVariant::DebugString() const {
  string result("<CompositeTensorVariant type=");
  result.append(TypeSpecProto::TypeSpecClass_Name(
      metadata_->type_spec_proto().type_spec_class()));
  result.append(", components=[");
  for (const auto& tensor : flat_components_) {
    if (&tensor != &flat_components_[0]) {
      result.append(", ");
    }
    result.append(tensor.DebugString());
  }
  result.append("]>");
  return result;
}
REGISTER_UNARY_VARIANT_DECODE_FUNCTION(CompositeTensorVariant,
                                       CompositeTensorVariant::kTypeName);
}  