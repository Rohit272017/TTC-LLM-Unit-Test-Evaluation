#include "base/internal/unknown_set.h"
#include "absl/base/no_destructor.h"
namespace cel::base_internal {
const AttributeSet& EmptyAttributeSet() {
  static const absl::NoDestructor<AttributeSet> empty_attribute_set;
  return *empty_attribute_set;
}
const FunctionResultSet& EmptyFunctionResultSet() {
  static const absl::NoDestructor<FunctionResultSet> empty_function_result_set;
  return *empty_function_result_set;
}
}  