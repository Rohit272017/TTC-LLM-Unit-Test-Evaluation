#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_UTILS_STABLEHLO_TYPE_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_UTILS_STABLEHLO_TYPE_UTILS_H_
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/Transforms/DialectConversion.h"  
#include "stablehlo/dialect/StablehloOps.h"  
namespace mlir::quant::stablehlo {
inline bool IsStablehloOp(Operation* op) {
  return op->getDialect()->getNamespace() ==
         mlir::stablehlo::StablehloDialect::getDialectNamespace();
}
}  
#endif  