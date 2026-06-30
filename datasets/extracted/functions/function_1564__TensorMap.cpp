#include "tensorflow/core/kernels/tensor_map.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/lib/core/coding.h"
namespace tensorflow {
TensorMap::~TensorMap() {
  if (tensors_) tensors_->Unref();
}
void TensorMap::Encode(VariantTensorData* data) const {
  data->set_type_name(TypeName());
  absl::flat_hash_map<TensorKey, Tensor>::const_iterator map_it =
      tensors().begin();
  while (map_it != tensors().end()) {
    Tensor k = map_it->first;
    Tensor v = map_it->second;
    CHECK_NE(k.dtype(), DT_INVALID);
    CHECK_NE(v.dtype(), DT_INVALID);
    *data->add_tensors() = k;
    *data->add_tensors() = v;
    map_it++;
  }
}
static Status TensorMapDeviceCopy(
    const TensorMap& from, TensorMap* to,
    const UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn& copy) {
  for (const std::pair<TensorKey, Tensor>& p : from.tensors()) {
    TensorKey to_key(p.first.dtype());
    Tensor to_val(p.second.dtype());
    TF_RETURN_IF_ERROR(copy(p.first, &to_key));
    TF_RETURN_IF_ERROR(copy(p.second, &to_val));
    to->tensors().emplace(to_key, to_val);
  }
  return absl::OkStatus();
}
#define REGISTER_LIST_COPY(DIRECTION)                                        \
  INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION(TensorMap, DIRECTION, \
                                                       TensorMapDeviceCopy)
REGISTER_LIST_COPY(VariantDeviceCopyDirection::HOST_TO_DEVICE);
REGISTER_LIST_COPY(VariantDeviceCopyDirection::DEVICE_TO_HOST);
REGISTER_LIST_COPY(VariantDeviceCopyDirection::DEVICE_TO_DEVICE);
REGISTER_UNARY_VARIANT_DECODE_FUNCTION(TensorMap, TensorMap::kTypeName);
bool TensorMap::Decode(const VariantTensorData& data) {
  std::vector<Tensor>::const_iterator tensors_it = data.tensors().begin();
  while (tensors_it != data.tensors().end()) {
    if (std::next(tensors_it) == data.tensors().end()) {
      return false;
    }
    tensors().emplace(tensors_it[0], tensors_it[1]);
    tensors_it += 2;
  }
  return true;
}
const char TensorMap::kTypeName[] = "tensorflow::TensorMap";
}  