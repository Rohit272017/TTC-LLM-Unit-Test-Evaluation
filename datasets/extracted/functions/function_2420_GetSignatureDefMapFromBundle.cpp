#include "tensorflow/compiler/mlir/lite/quantization/stablehlo/quantization.h"
#include <string>
#include <unordered_set>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/IR/BuiltinOps.h"  
#include "mlir/Pass/PassManager.h"  
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/tf_stablehlo_pass.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/config.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/static_range_ptq.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/weight_only_ptq.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_freeze_variables.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
namespace tensorflow {
namespace {
using ::mlir::quant::stablehlo::StaticRangePtqComponent;
using ::mlir::quant::stablehlo::WeightOnlyPtqComponent;
using ::stablehlo::quantization::Method;
using ::stablehlo::quantization::PopulateDefaults;
using ::stablehlo::quantization::QuantizationConfig;
using ::tensorflow::SignatureDef;
using ::tensorflow::quantization::PyFunctionLibrary;
absl::flat_hash_map<std::string, SignatureDef> GetSignatureDefMapFromBundle(
    const SavedModelBundle& saved_model_bundle) {
  const protobuf::Map<std::string, SignatureDef>& signatures =
      saved_model_bundle.GetSignatures();
  absl::flat_hash_map<std::string, SignatureDef> signature_def_map(
      signatures.begin(), signatures.end());
  signature_def_map.erase(kSavedModelInitOpSignatureKey);
  return signature_def_map;
}
absl::flat_hash_map<std::string, std::string> GetFunctionAliases(
    const SavedModelBundle& saved_model_bundle) {
  const protobuf::Map<std::string, std::string>& function_aliases =
      saved_model_bundle.meta_graph_def.meta_info_def().function_aliases();
  return absl::flat_hash_map<std::string, std::string>(function_aliases.begin(),
                                                       function_aliases.end());
}
}  
absl::StatusOr<mlir::ModuleOp> RunQuantization(
    const SavedModelBundle* saved_model_bundle,
    const absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& saved_model_tags,
    const QuantizationConfig& quantization_config,
    const PyFunctionLibrary* quantization_py_function_lib,
    mlir::ModuleOp module_op) {
  if (saved_model_bundle == nullptr) {
    return absl::InvalidArgumentError(
        "Failed to run quantization. `saved_model_bundle` should not be "
        "nullptr.");
  }
  if (quantization_py_function_lib == nullptr) {
    return absl::InvalidArgumentError(
        "Failed to run quantization. `quantization_py_function_lib` should not "
        "be nullptr.");
  }
  LOG(INFO) << "User-provided quantization config: "
            << quantization_config.DebugString();
  const QuantizationConfig updated_config =
      ExpandPresets(PopulateDefaults(quantization_config));
  LOG(INFO) << "Updated quantization config: " << updated_config.DebugString();
  const absl::flat_hash_map<std::string, SignatureDef> signature_def_map =
      GetSignatureDefMapFromBundle(*saved_model_bundle);
  std::vector<std::string> exported_names;
  for (const auto& [key, value_unused] : signature_def_map) {
    exported_names.push_back(key);
  }
  if (failed(mlir::tf_saved_model::FreezeVariables(
          module_op, saved_model_bundle->GetSession()))) {
    return absl::InternalError("Failed to freeze variables.");
  }
  mlir::PassManager pm(module_op.getContext());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  mlir::odml::AddLegalizeTFToStablehloPasses(pm, true,
                                             false,
                                             false);
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::stablehlo::createRemoveShardingCustomCallPass());
  if (failed(pm.run(module_op))) {
    return absl::InternalError("Failed to run legalize TF to StableHLO.");
  }
  absl::StatusOr<mlir::ModuleOp> quantized_module_op;
  if (HasQuantizationMethod(updated_config.specs(),
                            Method::MethodCase::kStaticRangePtq)) {
    StaticRangePtqComponent static_range_ptq_component(
        module_op.getContext(), quantization_py_function_lib, saved_model_dir,
        exported_names, saved_model_tags, signature_def_map,
        GetFunctionAliases(*saved_model_bundle));
    quantized_module_op =
        static_range_ptq_component.Run(module_op, updated_config);
  } else if (HasQuantizationMethod(updated_config.specs(),
                                   Method::MethodCase::kWeightOnlyPtq)) {
    WeightOnlyPtqComponent weight_only_ptq_component(module_op.getContext());
    quantized_module_op =
        weight_only_ptq_component.Run(module_op, updated_config);
  } else {
    return absl::InvalidArgumentError(
        "Quantization config must have either static_range_ptq_preset or "
        "weight_only_ptq_preset.");
  }
  if (!quantized_module_op.ok()) {
    return absl::InternalError("Failed to run quantization. Status msg: " +
                               quantized_module_op.status().ToString());
  }
  return quantized_module_op;
}
}  