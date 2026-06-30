#include "tensorflow/compiler/mlir/lite/sparsity/sparsify_model.h"
#include <cstdint>
#include <string>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/buffer.h"  
#include "flatbuffers/flatbuffer_builder.h"  
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/IR/BuiltinOps.h"  
#include "mlir/IR/Location.h"  
#include "mlir/IR/MLIRContext.h"  
#include "mlir/IR/OwningOpRef.h"  
#include "mlir/Pass/PassManager.h"  
#include "mlir/Support/LogicalResult.h"  
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/compiler/mlir/lite/tools/optimize/reduced_precision_metadata.h"
#include "tensorflow/compiler/mlir/lite/transforms/dense_to_sparse_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/pass_registry_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/types.pb.h"
namespace mlir {
namespace lite {
absl::Status SparsifyModel(const tflite::ModelT& input_model,
                           flatbuffers::FlatBufferBuilder* builder) {
  MLIRContext context;
  StatusScopedDiagnosticHandler statusHandler(&context,
                                              true);
  flatbuffers::FlatBufferBuilder input_builder;
  flatbuffers::Offset<tflite::Model> input_model_location =
      tflite::Model::Pack(input_builder, &input_model);
  tflite::FinishModelBuffer(input_builder, input_model_location);
  std::string serialized_model(
      reinterpret_cast<const char*>(input_builder.GetBufferPointer()),
      input_builder.GetSize());
  OwningOpRef<mlir::ModuleOp> module = tflite::FlatBufferToMlir(
      serialized_model, &context, UnknownLoc::get(&context));
  if (!module) {
    LOG(ERROR) << "Couldn't import flatbuffer to MLIR.";
    return absl::InternalError("Couldn't import flatbuffer to MLIR.");
  }
  PassManager pm((*module)->getName(), OpPassManager::Nesting::Implicit);
  pm.addPass(TFL::Create<TFL::DenseToSparsePass>());
  if (failed(pm.run(module.get()))) {
    LOG(ERROR) << "Failed to sparsify: "
               << statusHandler.ConsumeStatus().message();
    return absl::InternalError(absl::StrCat(
        "Failed to sparsify: ", statusHandler.ConsumeStatus().message()));
  }
  std::string result;
  tflite::FlatbufferExportOptions options;
  options.converter_flags.set_force_select_tf_ops(false);
  options.converter_flags.set_enable_select_tf_ops(true);
  options.converter_flags.set_allow_custom_ops(true);
  for (const auto& metadata : input_model.metadata) {
    if (metadata->name != tflite::optimize::kTfLiteReducedPrecisionKey) {
      continue;
    }
    const auto& data = input_model.buffers[metadata->buffer]->data;
    options.metadata[metadata->name] = std::string(data.begin(), data.end());
    break;
  }
  if (!tflite::MlirToFlatBufferTranslateFunction(module.get(), options,
                                                 &result)) {
    LOG(ERROR) << "Failed to export MLIR to flatbuffer.";
    return absl::InternalError("Failed to export MLIR to flatbuffer.");
  }
  builder->PushFlatBuffer(reinterpret_cast<const uint8_t*>(result.data()),
                          result.size());
  return absl::OkStatus();
}
}  
}  