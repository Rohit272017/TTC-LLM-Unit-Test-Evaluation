#include "tensorflow/compiler/mlir/tfrt/function/function.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "mlir/IR/OperationSupport.h"  
#include "mlir/Pass/PassManager.h"  
#include "mlir/Support/LogicalResult.h"  
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/tfrt_pipeline_options.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tfrt/bef/bef_buffer.h"  
#include "tfrt/bef_converter/mlir_to_bef.h"  
namespace tensorflow {
Status CompileTFMLIRToBEF(const TfrtFunctionCompileOptions& options,
                          mlir::ModuleOp module, tfrt::BefBuffer* bef_buffer) {
  mlir::OpPrintingFlags print_flags;
  print_flags.elideLargeElementsAttrs();
  if (VLOG_IS_ON(1)) {
    VLOG(1) << "Input TF Executor dialect:";
    DumpMlirOpToFile("tf_to_tfrt_tf_executor_dialect", module);
  }
  mlir::StatusScopedDiagnosticHandler diag_handler(module.getContext());
  mlir::PassManager pm(module.getContext());
  tensorflow::applyTensorflowAndCLOptions(pm);
  tensorflow::TfrtPipelineOptions pass_options;
  if (!options.default_device.empty()) {
    pass_options.default_device = options.default_device;
  }
  if (!options.force_data_format.empty()) {
    pass_options.force_data_format = options.force_data_format;
  }
  if (absl::StrContains(pass_options.default_device, "CPU")) {
    pass_options.skip_fold_transpose_in_ops = true;
  }
  pass_options.enable_optimizer = options.enable_optimizer;
  pass_options.target_tpurt = false;
  pass_options.tpu_use_core_selector = options.tpu_use_core_selector;
  pass_options.tpu_use_bundled_transfer = options.tpu_use_bundled_transfer;
  pass_options.tpu_lower_to_fallback = options.tpu_lower_to_fallback;
  pass_options.tpu_fuse_ops = options.tpu_fuse_ops;
  pass_options.tpu_transfer_result_to_host =
      options.tpu_transfer_result_to_host;
  Status status = tensorflow::CreateTfExecutorToTfrtPipeline(pm, pass_options);
  if (!status.ok()) {
    return diag_handler.Combine(status);
  }
  if (mlir::failed(pm.run(module)))
    return diag_handler.Combine(tensorflow::errors::Internal(
        "failed to lower TF Dialect to CoreRT dialect."));
  if (VLOG_IS_ON(1)) {
    VLOG(1) << "TFRT dialect: ";
    DumpMlirOpToFile("tf_to_tfrt_tfrt_dialect", module);
  }
  *bef_buffer =
      tfrt::ConvertMLIRToBEF(module,  true);
  if (bef_buffer->empty())
    return diag_handler.Combine(
        tensorflow::errors::Internal("failed to convert MLIR to BEF."));
  return absl::OkStatus();
}
}  