#include "common/reference.h"
#include "absl/base/no_destructor.h"
namespace cel {
const VariableReference& VariableReference::default_instance() {
  static const absl::NoDestructor<VariableReference> instance;
  return *instance;
}
const FunctionReference& FunctionReference::default_instance() {
  static const absl::NoDestructor<FunctionReference> instance;
  return *instance;
}
}  