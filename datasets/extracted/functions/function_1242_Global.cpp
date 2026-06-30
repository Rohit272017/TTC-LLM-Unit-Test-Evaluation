#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include <set>
#include <string>
#include "tensorflow/core/platform/logging.h"
namespace tensorflow {
InputColocationExemptionRegistry* InputColocationExemptionRegistry::Global() {
  static InputColocationExemptionRegistry* registry =
      new InputColocationExemptionRegistry;
  return registry;
}
void InputColocationExemptionRegistry::Register(const string& op) {
  auto it = ops_.find(op);
  if (it != ops_.end()) {
    LOG(WARNING) << "Input colocation exemption for op: " << op
                 << " already registered";
  } else {
    ops_.insert(op);
  }
}
}  