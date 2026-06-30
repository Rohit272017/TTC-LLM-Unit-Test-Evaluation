#include "tensorflow/lite/core/async/interop/attribute_map_internal.h"
#include "tensorflow/lite/core/async/interop/reconcile_fns.h"
namespace tflite {
namespace interop {
bool AttributeMap::ReconcileAttributes(const AttributeMap* other,
                                       AttributeMap* merged,
                                       AttributeMap* conflict) const {
  if (other == nullptr || merged == nullptr) return false;
  if (type_ != other->type_) return false;
  merged->type_ = type_;
  if (conflict) conflict->type_ = type_;
  return tflite::interop::ReconcileGeneralAttributeKeys(
      type_, &attrs_, &other->attrs_, &merged->attrs_,
      conflict ? &conflict->attrs_ : nullptr);
}
bool AttributeMap::CheckAttributeCoverage(const AttributeMap* other,
                                          AttributeMap* conflict) const {
  if (other == nullptr) return false;
  if (type_ != other->type_) return false;
  if (conflict) conflict->type_ = type_;
  return tflite::interop::CheckGeneralAttributeKeysCoverage(
      type_, &attrs_, &other->attrs_, conflict ? &conflict->attrs_ : nullptr);
}
}  
}  