#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/pre_calibration.h"
#include "absl/base/nullability.h"
#include "absl/log/die_if_null.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"  
#include "mlir/Pass/PassManager.h"  
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/pass_pipeline.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/run_passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tsl/platform/errors.h"
namespace mlir::quant::stablehlo {
using ::stablehlo::quantization::QuantizationConfig;
using ::tensorflow::quantization::RunPasses;
PreCalibrationComponent::PreCalibrationComponent(
    absl::Nonnull<MLIRContext*> ctx)
    : ctx_(ABSL_DIE_IF_NULL(ctx)) {}  
absl::StatusOr<ModuleOp> PreCalibrationComponent::Run(
    ModuleOp module_op, const QuantizationConfig& config) {
  TF_RETURN_IF_ERROR(RunPasses(
      kName, 
      [&config](PassManager& pm) {
        AddPreCalibrationPasses(pm, config.calibration_options(),
                                config.specs(), config.debugger_config());
      },
      *ctx_, module_op));
  return module_op;
}
}  