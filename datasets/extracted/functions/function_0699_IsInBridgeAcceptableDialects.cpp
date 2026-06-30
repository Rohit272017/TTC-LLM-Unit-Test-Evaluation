#include "tensorflow/compiler/mlir/tf2xla/internal/utils/dialect_detection_utils.h"
#include <set>
#include <string>
#include "mlir/IR/Dialect.h"  
#include "mlir/IR/Operation.h"  
#include "mlir/IR/Visitors.h"  
namespace tensorflow {
namespace tf2xla {
namespace internal {
bool IsInBridgeAcceptableDialects(mlir::Operation* op) {
  const std::set<std::string> kBuiltinNamespaces = {"func", "return",
                                                    "builtin"};
  const std::set<std::string> kBridgeAcceptableNamespaces = {"tf", "tf_device"};
  bool isInDefaulNamespaces =
      kBuiltinNamespaces.find(op->getDialect()->getNamespace().str()) !=
      kBuiltinNamespaces.end();
  bool isInBridgeAcceptableNamespaces =
      kBridgeAcceptableNamespaces.find(
          op->getDialect()->getNamespace().str()) !=
      kBridgeAcceptableNamespaces.end();
  return isInDefaulNamespaces || isInBridgeAcceptableNamespaces;
}
}  
}  
}  