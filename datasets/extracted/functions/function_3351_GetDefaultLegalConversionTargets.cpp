#include "tensorflow/compiler/mlir/tf2xla/transforms/xla_legalize_targets.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/Dialect/Shape/IR/Shape.h"  
#include "mlir/Dialect/Tensor/IR/Tensor.h"  
#include "mlir/Transforms/DialectConversion.h"  
#include "stablehlo/dialect/ChloOps.h"  
#include "stablehlo/dialect/StablehloOps.h"  
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
namespace mlir {
namespace mhlo {
ConversionTarget GetDefaultLegalConversionTargets(MLIRContext& mlir_context,
                                                  bool legalize_chlo) {
  ConversionTarget target(mlir_context);
  if (legalize_chlo) {
    target.addIllegalDialect<chlo::ChloDialect>();
    target.addIllegalDialect<stablehlo::StablehloDialect>();
  } else {
    target.addLegalDialect<chlo::ChloDialect>();
  }
  target.addLegalDialect<MhloDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<tensor::TensorDialect>();
  target.addLegalDialect<shape::ShapeDialect>();
  target.addLegalOp<func::CallOp>();
  target.addLegalOp<TF::_XlaHostComputeMlirOp, TF::XlaSendToHostOp,
                    TF::XlaRecvFromHostOp>();
  return target;
}
}  
}  