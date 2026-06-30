#ifndef TENSORFLOW_DTENSOR_CC_DTENSOR_OPERATION_H_
#define TENSORFLOW_DTENSOR_CC_DTENSOR_OPERATION_H_
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
namespace tensorflow {
namespace dtensor {
struct DTensorOperation {
  const char* name;
  const FunctionDef* function_def;
  const Mesh default_mesh;
  const StackTracesMap& stack_traces;
  inline bool is_func() const { return function_def != nullptr; }
  inline bool is_pure() const {
    if (is_func()) {
      return false;
    }
    const OpDef* op_def = nullptr;
    Status status = OpRegistry::Global()->LookUpOpDef(name, &op_def);
    DCHECK(status.ok());  
    if (!status.ok()) {
      return false;
    }
    return !op_def->is_stateful();
  }
};
}  
}  
#endif  