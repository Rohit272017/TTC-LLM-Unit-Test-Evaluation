#include "tensorflow/compiler/mlir/register_common_dialects.h"
#include "mlir/Dialect/Quant/IR/Quant.h"  
#include "mlir/Dialect/Shape/IR/Shape.h"  
#include "mlir/Dialect/Tensor/IR/Tensor.h"  
#include "mlir/Dialect/Tosa/IR/TosaOps.h"  
#include "mlir/InitAllDialects.h"  
#include "mlir/InitAllExtensions.h"  
#include "stablehlo/dialect/Register.h"  
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "xla/mlir/framework/ir/xla_framework.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"
#include "tensorflow/core/ir/types/dialect.h"
namespace mlir {
void RegisterCommonToolingDialects(mlir::DialectRegistry& registry) {
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::TFL::TensorFlowLiteDialect>();
  registry.insert<mlir::kernel_gen::tf_framework::TFFrameworkDialect>();
  registry.insert<mlir::quant::QuantDialect>();
  registry.insert<mlir::quantfork::QuantizationForkDialect>();
  registry.insert<mlir::shape::ShapeDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::tosa::TosaDialect>();
  registry.insert<mlir::xla_framework::XLAFrameworkDialect,
                  mlir::TF::TensorFlowDialect, mlir::tf_type::TFTypeDialect>();
}
};  