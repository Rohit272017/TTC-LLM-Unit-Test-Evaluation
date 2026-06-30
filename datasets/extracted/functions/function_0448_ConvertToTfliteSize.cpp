#include "tensorflow/compiler/mlir/lite/utils/size_utils.h"
#include <cstdint>
#include "mlir/IR/BuiltinTypeInterfaces.h"  
namespace mlir {
namespace TFL {
int32_t ConvertToTfliteSize(int64_t size) {
  return mlir::ShapedType::isDynamic(size) ? -1 : static_cast<int32_t>(size);
}
}  
}  