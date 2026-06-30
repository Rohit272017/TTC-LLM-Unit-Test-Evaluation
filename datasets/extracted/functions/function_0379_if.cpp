#include "eval/public/containers/internal_field_backed_list_impl.h"
#include "eval/public/cel_value.h"
#include "eval/public/structs/field_access_impl.h"
namespace google::api::expr::runtime::internal {
int FieldBackedListImpl::size() const {
  return reflection_->FieldSize(*message_, descriptor_);
}
CelValue FieldBackedListImpl::operator[](int index) const {
  auto result = CreateValueFromRepeatedField(message_, descriptor_, index,
                                             factory_, arena_);
  if (!result.ok()) {
    CreateErrorValue(arena_, result.status().ToString());
  }
  return *result;
}
}  