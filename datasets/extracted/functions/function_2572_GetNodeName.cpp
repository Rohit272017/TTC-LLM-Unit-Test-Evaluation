#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/saved_model_export.h"
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/IR/BuiltinOps.h"  
#include "mlir/Pass/PassManager.h"  
#include "mlir/Support/LLVM.h"  
#include "mlir/Support/LogicalResult.h"  
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/io.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/pass_pipeline.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/types.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/convert_asset_args.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/run_passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/constants.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/unfreeze_constants.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v2/tf_executor_to_graph.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saver.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace mlir::quant::stablehlo {
namespace {
using ::mlir::tf_saved_model::kTfSavedModelIndexPathAttr;
using ::mlir::tf_saved_model::kTfSavedModelInitializerInitType;
using ::mlir::tf_saved_model::kTfSavedModelInitializerRestoreType;
using ::stablehlo::quantization::QuantizationConfig;
using ::stablehlo::quantization::io::GetLocalTmpFileName;
using ::tensorflow::AssetFileDef;
using ::tensorflow::FunctionDefLibrary;
using ::tensorflow::FunctionLibraryDefinition;
using ::tensorflow::Graph;
using ::tensorflow::GraphDef;
using ::tensorflow::Node;
using ::tensorflow::NodeDef;
using ::tensorflow::OpRegistry;
using ::tensorflow::SaverDef;
using ::tensorflow::quantization::ExportedModel;
using ::tensorflow::quantization::RunPasses;
using ::tensorflow::quantization::UnfreezeConstantsAndSaveVariables;
std::string GetNodeName(const std::vector<std::string>& control_ret_node_names,
                        const absl::string_view contains) {
  for (const std::string& node_name : control_ret_node_names) {
    if (absl::StrContains(node_name, contains)) {
      VLOG(1) << "Node found: " << node_name << ", contains: " << contains;
      return node_name;
    }
  }
  VLOG(1) << "Could not find node whose name conatins: " << contains;
  return "";
}
std::string FindFilePrefixTensorName(const GraphDef& graph_def) {
  for (const NodeDef& node_def : graph_def.node()) {
    if (node_def.op() == FunctionLibraryDefinition::kArgOp) {
      const auto index_path_attr_itr =
          node_def.attr().find(kTfSavedModelIndexPathAttr.str());
      if (index_path_attr_itr != node_def.attr().end()) {
        const auto& index_paths = index_path_attr_itr->second.list().s();
        if (absl::c_find(index_paths, kTfFilePrefix.str()) !=
            index_paths.end()) {
          return absl::StrCat(node_def.name(), ":0");
        }
      }
    }
  }
  return "";
}
}  
absl::StatusOr<ExportedModel> CreateExportedModel(
    const std::vector<std::string>& signature_keys,
    const std::unordered_set<std::string>& tags,
    const QuantizationConfig& quantization_config,
    absl::string_view debug_name_prefix,
    const absl::flat_hash_map<FunctionName, FunctionAlias>& function_aliases,
    MLIRContext& ctx ABSL_ATTRIBUTE_LIFETIME_BOUND, ModuleOp module_op) {
  TF_ASSIGN_OR_RETURN(const std::string checkpoint_dir, GetLocalTmpFileName());
  const ExportOptions export_opts = {
      true,
      false, checkpoint_dir,
      absl::StrCat(debug_name_prefix, kExportStepSuffix)};
  TF_ASSIGN_OR_RETURN(const SmallVector<AssetFileDef> asset_file_defs,
                      RunExportPasses(export_opts, ctx, module_op));
  return ConvertMlirModuleToExportedModel(
      module_op, checkpoint_dir, function_aliases,
      {asset_file_defs.begin(), asset_file_defs.end()});
}
ExportedModel CreateExportedModelFromGraphDef(
    GraphDef&& graph_def, const absl::string_view init_node_name,
    const absl::string_view checkpoint_dir,
    const std::optional<SaverDef> saver_def,
    const absl::flat_hash_map<FunctionName, FunctionAlias>& function_aliases,
    const std::vector<AssetFileDef>& asset_file_defs) {
  ExportedModel exported_model{};
  *exported_model.mutable_graph_def() = graph_def;
  exported_model.set_init_node_name(std::string(init_node_name));
  exported_model.set_checkpoint_dir(std::string(checkpoint_dir));
  exported_model.mutable_function_aliases()->insert(function_aliases.begin(),
                                                    function_aliases.end());
  for (const AssetFileDef& asset_file_def : asset_file_defs) {
    *exported_model.mutable_asset_file_defs()->Add() = asset_file_def;
  }
  if (saver_def != std::nullopt) {
    *exported_model.mutable_saver_def() = *std::move(saver_def);
  }
  return exported_model;
}
void AddExportPasses(mlir::PassManager& pm,
                     const bool duplicate_shape_determining_constants) {
  AddCallModuleSerializationPasses(pm);
  if (duplicate_shape_determining_constants) {
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::quant::CreateDuplicateShapeDeterminingConstantsPass());
  }
  pm.addPass(mlir::quant::CreateInsertMainFunctionPass());
  pm.addPass(mlir::quant::CreateLiftHashTableOpsAsArgsPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::CreateFunctionalToExecutorDialectConversionPass());
  pm.addPass(mlir::CreateBreakUpIslandsPass());
  pm.addPass(mlir::quant::CreateMergeInitializerFunctionOpsToMainPass());
  pm.addPass(mlir::quant::CreateMergeSaveFunctionOpsToMainPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::CreateMergeDuplicateResourceOpsPass());
  pm.addPass(mlir::TF::CreateStripNoinlineAttributePass());
}
absl::StatusOr<std::optional<SaverDef>> CreateSaverDef(
    const std::vector<std::string>& control_ret_node_names,
    const GraphDef& graph_def) {
  const std::string filename_tensor_name = FindFilePrefixTensorName(graph_def);
  const std::string restore_op_name =
      GetNodeName(control_ret_node_names, kTfSavedModelInitializerRestoreType);
  const std::string save_node_name =
      GetNodeName(control_ret_node_names, kTfQuantSaveOpName);
  const std::vector<absl::string_view> fields = {
      filename_tensor_name, restore_op_name, save_node_name};
  const auto is_empty_predicate = [](const absl::string_view s) {
    return s.empty();
  };
  if (absl::c_all_of(fields, is_empty_predicate)) {
    return std::nullopt;
  } else if (absl::c_none_of(fields, is_empty_predicate)) {
    SaverDef saver_def{};
    saver_def.set_version(SaverDef::V2);
    saver_def.set_filename_tensor_name(filename_tensor_name);
    saver_def.set_restore_op_name(restore_op_name);
    saver_def.set_save_tensor_name(absl::StrCat(save_node_name, ":0"));
    return saver_def;
  } else {
    return absl::InternalError(
        absl::StrCat("Failed to create SaverDef. Fields should be either all "
                     "empty strings or all non-empty strings. Got fields: ",
                     absl::StrJoin(fields, ",")));
  }
}
absl::StatusOr<ExportedModel> ConvertMlirModuleToExportedModel(
    const mlir::ModuleOp module_op, const absl::string_view checkpoint_dir,
    const absl::flat_hash_map<FunctionName, FunctionAlias>& function_aliases,
    const std::vector<AssetFileDef>& asset_file_defs) {
  const tensorflow::GraphExportConfig config{};
  FunctionLibraryDefinition flib_def{OpRegistry::Global(),
                                     FunctionDefLibrary()};
  std::unique_ptr<Graph> graph;
  absl::flat_hash_set<Node*> control_ret_nodes{};
  TF_RETURN_IF_ERROR(tensorflow::tf2xla::v2::ConvertTfExecutorToGraph(
      module_op, config, &graph, &flib_def, &control_ret_nodes));
  GraphDef graph_def{};
  graph->ToGraphDef(&graph_def);
  std::vector<std::string> control_ret_node_names{};
  for (Node* node : control_ret_nodes) {
    control_ret_node_names.push_back(node->name());
  }
  const std::string init_node_name =
      GetNodeName(control_ret_node_names, kTfSavedModelInitializerInitType);
  TF_ASSIGN_OR_RETURN(const std::optional<SaverDef> saver_def,
                      CreateSaverDef(control_ret_node_names, graph_def));
  return CreateExportedModelFromGraphDef(std::move(graph_def), init_node_name,
                                         checkpoint_dir, std::move(saver_def),
                                         function_aliases, asset_file_defs);
}
absl::StatusOr<SmallVector<AssetFileDef>> RunExportPasses(
    const ExportOptions& export_opts, MLIRContext& ctx, ModuleOp module_op) {
  if (export_opts.unfreeze_constants) {
    TF_RETURN_IF_ERROR(UnfreezeConstantsAndSaveVariables(
        export_opts.checkpoint_dir, ctx, module_op));
    LOG(INFO) << "Unfrozen constants and saved variables to checkpoint file: "
              << export_opts.checkpoint_dir;
  }
  TF_RETURN_IF_ERROR(RunPasses(
      export_opts.debug_name,
      [dup_constants = export_opts.duplicate_shape_determining_constants](
          PassManager& pm) { AddExportPasses(pm, dup_constants); },
      ctx, module_op));
  FailureOr<SmallVector<AssetFileDef>> asset_file_defs =
      quant::ConvertAssetArgs(module_op);
  if (failed(asset_file_defs)) {
    return absl::InternalError("Failed to convert asset args.");
  }
  return *asset_file_defs;
}
}  