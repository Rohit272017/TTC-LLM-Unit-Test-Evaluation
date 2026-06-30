#include "tensorflow/compiler/mlir/quantization/stablehlo/utils/bfloat16_type.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"  
#include "mlir/IR/BuiltinTypes.h"  
#include "mlir/IR/TypeUtilities.h"  
#include "mlir/IR/Types.h"  
#include "mlir/Support/LLVM.h"  
namespace mlir::quant::stablehlo {
bool IsLargeFloatType(Type type) {
  type = getElementTypeOrSelf(type);
  return isa<FloatType>(type) && type.getIntOrFloatBitWidth() > 16;
}
Type ToBfloat16Type(Type type) {
  if (auto shaped = mlir::dyn_cast<ShapedType>(type)) {
    const Type elem = shaped.getElementType();
    if (IsLargeFloatType(elem)) {
      return shaped.clone(BFloat16Type::get(type.getContext()));
    }
  } else if (IsLargeFloatType(type)) {
    return BFloat16Type::get(type.getContext());
  }
  return type;
}
}  