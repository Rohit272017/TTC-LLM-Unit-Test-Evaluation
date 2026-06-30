#include "xla/service/spmd/shardy/shardy_xla_pass.h"
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "mhlo/transforms/passes.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/file_utils.h"
#include "shardy/dialect/sdy/transforms/propagation/passes.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"
#include "xla/hlo/utils/hlo_sharding_util.h"
#include "xla/layout.h"
#include "xla/map_util.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/service/computation_layout.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/mhlo_round_trip/mhlo_export.h"
#include "xla/service/spmd/shardy/mhlo_round_trip/mhlo_import.h"
#include "xla/service/spmd/shardy/sdy_round_trip/pipelines.h"
#include "xla/service/spmd/shardy/utils.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/path.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace sdy {
namespace {
absl::Status createFromProtoAndReplaceComputations(
    HloModule* module, const HloModuleProto& proto) {
  absl::flat_hash_map<int64_t, HloComputation*> idToComputation;
  std::vector<std::unique_ptr<HloComputation>> computations;
  HloComputation* entryComputation = nullptr;
  for (const HloComputationProto& computationProto : proto.computations()) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloComputation> computation,
        HloComputation::CreateFromProto(computationProto, idToComputation));
    CHECK_NE(computation.get(), nullptr);
    const int64_t computationId = computationProto.id();
    CHECK_NE(computationId, -1);
    CHECK(!ContainsKey(idToComputation, computationId));
    idToComputation[computationId] = computation.get();
    if (computationId == proto.entry_computation_id()) {
      CHECK_EQ(entryComputation, nullptr);
      entryComputation = computation.get();
    }
    computations.push_back(std::move(computation));
  }
  CHECK_NE(entryComputation, nullptr);
  absl::c_sort(computations, [](const std::unique_ptr<HloComputation>& a,
                                const std::unique_ptr<HloComputation>& b) {
    return a->unique_id() < b->unique_id();
  });
  for (std::unique_ptr<HloComputation>& computation : computations) {
    HloComputation* newComputation =
        module->AddComputationAndUnifyNamesAndIds(std::move(computation),
                                                  false);
    if (newComputation == entryComputation) {
      module->ReplaceEntryComputation(newComputation);
    }
  }
  CHECK_OK(HloDCE().Run(module));
  return absl::OkStatus();
}
using OriginalParamIndexToFlattenedNum =
    std::vector<absl::flat_hash_map<ShapeIndex, int64_t>>;
int64_t getFlattenedParamNumber(
    const OriginalParamIndexToFlattenedNum& originalParamIndexToFlattenedNum,
    int64_t paramNumber, const ShapeIndex& paramIndex) {
  return originalParamIndexToFlattenedNum[paramNumber].at(paramIndex);
}
OriginalParamIndexToFlattenedNum getOriginalParamIndexToFlattenedNum(
    HloModule* hloModule) {
  OriginalParamIndexToFlattenedNum originalParamIndexToFlattened;
  HloComputation* entryComputation = hloModule->entry_computation();
  originalParamIndexToFlattened.reserve(entryComputation->num_parameters());
  int64_t paramNumber = 0;
  for (HloInstruction* paramInstruction :
       entryComputation->parameter_instructions()) {
    auto& paramMap = originalParamIndexToFlattened.emplace_back();
    ShapeUtil::ForEachLeafShape(paramInstruction->shape(),
                                [&](const Shape&, const ShapeIndex& index) {
                                  paramMap[index] = paramNumber++;
                                });
  }
  return originalParamIndexToFlattened;
}
Shape getFlattenedShape(const Shape& shape) {
  std::vector<Shape> flattenedShapes;
  ShapeUtil::ForEachLeafShape(
      shape, [&](const Shape& subShape, const ShapeIndex& index) {
        flattenedShapes.push_back(subShape);
      });
  if (flattenedShapes.empty()) {
    return Shape();
  }
  return ShapeUtil::MakeMaybeTupleShape(flattenedShapes);
}
ComputationLayout getFlattenedComputationLayout(
    const ComputationLayout& computationLayout, bool useTupleArgs) {
  if (!computationLayout.AnyLayoutSet()) {
    return computationLayout;
  }
  ComputationLayout flattenedComputationLayout = ComputationLayout(
      ShapeLayout(getFlattenedShape(computationLayout.result_shape())));
  Shape tupleShape;
  tupleShape.set_element_type(PrimitiveType::TUPLE);
  for (int64_t i = 0; i != computationLayout.parameter_count(); ++i) {
    ShapeUtil::ForEachLeafShape(
        computationLayout.parameter_shape(i),
        [&](const Shape& subShape, const ShapeIndex& index) {
          if (useTupleArgs) {
            *tupleShape.add_tuple_shapes() = subShape;
          } else {
            flattenedComputationLayout.add_parameter_layout(
                ShapeLayout(subShape));
          }
        });
  }
  if (useTupleArgs) {
    flattenedComputationLayout.add_parameter_layout(ShapeLayout(tupleShape));
  }
  return flattenedComputationLayout;
}
std::pair<int64_t, ShapeIndex> getFlattenedParamNumberAndIndex(
    const OriginalParamIndexToFlattenedNum& originalParamIndexToFlattenedNum,
    int64_t parameterNumber, const ShapeIndex& parameterIndex,
    bool useTupleArgs) {
  int64_t flattenedIndex = getFlattenedParamNumber(
      originalParamIndexToFlattenedNum, parameterNumber, parameterIndex);
  if (useTupleArgs) {
    return {0, ShapeIndex{flattenedIndex}};
  }
  return {flattenedIndex, ShapeIndex()};
}
HloInputOutputAliasConfig getFlattenedInputOutputAliasConfig(
    const HloInputOutputAliasConfig& inputOutputAliasConfig,
    const OriginalParamIndexToFlattenedNum& originalParamIndexToFlattenedNum,
    bool useTupleArgs) {
  HloInputOutputAliasConfig flattenedInputOutputAliasConfig(
      getFlattenedShape(inputOutputAliasConfig.shape()));
  int64_t resultIndex = 0;
  ShapeUtil::ForEachLeafShape(
      inputOutputAliasConfig.shape(),
      [&](const Shape&, const ShapeIndex& index) {
        if (const std::optional<HloInputOutputAliasConfig::Alias>& alias =
                inputOutputAliasConfig.GetAliasedParameter(index)) {
          auto [paramNumber, paramIndex] = getFlattenedParamNumberAndIndex(
              originalParamIndexToFlattenedNum, alias->parameter_number,
              alias->parameter_index, useTupleArgs);
          CHECK_OK(flattenedInputOutputAliasConfig.SetUpAlias(
              flattenedInputOutputAliasConfig.shape().IsTuple()
                  ? ShapeIndex{resultIndex}
                  : ShapeIndex(),
              paramNumber, paramIndex, alias->kind));
        }
        ++resultIndex;
      });
  return flattenedInputOutputAliasConfig;
}
HloBufferDonorConfig getFlattenedBufferDonorsConfig(
    const HloBufferDonorConfig& bufferDonorsConfig,
    const OriginalParamIndexToFlattenedNum& originalParamIndexToFlattenedNum,
    bool useTupleArgs) {
  HloBufferDonorConfig flattenedBufferDonorsConfig;
  for (const HloBufferDonorConfig::BufferDonor& bufferDonor :
       bufferDonorsConfig.buffer_donor()) {
    auto [paramNumber, paramIndex] = getFlattenedParamNumberAndIndex(
        originalParamIndexToFlattenedNum, bufferDonor.param_number,
        bufferDonor.param_index, useTupleArgs);
    CHECK_OK(
        flattenedBufferDonorsConfig.AddBufferDonor(paramNumber, paramIndex));
  }
  return flattenedBufferDonorsConfig;
}
void removeFrontendAttributes(HloModule* hloModule,
                              mlir::ArrayRef<mlir::StringRef> attributeNames) {
  FrontendAttributes feAttrs = hloModule->frontend_attributes();
  auto* map = feAttrs.mutable_map();
  for (const auto& attributeName : attributeNames) {
    map->erase(attributeName);
  }
  hloModule->set_frontend_attributes(feAttrs);
}
}  
absl::StatusOr<bool> ShardyXLA::Run(
    HloModule* hloModule,
    const absl::flat_hash_set<absl::string_view>& executionThreads) {
  LOG(INFO) << "Using Shardy for XLA SPMD propagation.";
  auto mlirContext = std::make_unique<mlir::MLIRContext>();
  loadAllRequiredDialects(mlirContext.get());
  mlir::OwningOpRef<mlir::ModuleOp> mlirModule =
      xla::llvm_ir::CreateMlirModuleOp(
          mlir::UnknownLoc::get(mlirContext.get()));
  TF_RETURN_IF_ERROR(
      ConvertHloToMlirHlo(*mlirModule, hloModule,
                          false,
                          true));
  std::string shardyDir = hloModule->config().debug_options().xla_dump_to();
  if (shardyDir == "sponge") {
    shardyDir = getenv("TEST_UNDECLARED_OUTPUTS_DIR");
    if (shardyDir.empty()) {
      LOG(WARNING) << "\"sponge\" specified as dump directory but "
                      "TEST_UNDECLARED_OUTPUTS_DIR is not set!";
    }
  }
  if (!shardyDir.empty()) {
    shardyDir =
        tsl::io::JoinPath(shardyDir, "shardy",
                          std::string_view(mlirModule->getName().value_or("")));
    LOG(INFO) << "Using Shardy output directory: " << shardyDir;
  }
  bool enableVerifier = false;
#ifndef NDEBUG
  enableVerifier = true;
#endif
  mlir::PassManager pm(mlirContext.get());
  pm.enableVerifier(enableVerifier);
  pm.addPass(mlir::sdy::createSaveModuleOpPass(shardyDir,
                                               "sdy_module_before_xla_import"));
  bool useTupleArgs = false;
  mlir::DictionaryAttr moduleFrontendAttrs = getFrontendAttrs(*mlirModule);
  if (moduleFrontendAttrs && moduleFrontendAttrs.get(kUseTupleArgs)) {
    useTupleArgs = true;
    removeFrontendAttribute(*mlirModule, kUseTupleArgs);
  }
  if (moduleFrontendAttrs &&
      moduleFrontendAttrs.get(kPythonIntegrationComplete)) {
    removeFrontendAttribute(*mlirModule, kPythonIntegrationComplete);
    addSdyRoundTripImportPipeline(pm);
  } else {
    auto spanToArrayRef = [](absl::Span<const bool> span) {
      return mlir::ArrayRef<bool>(span.data(), span.size());
    };
    addMhloImportPipeline(
        pm,
        spanToArrayRef(hloModule->config()
                           .allow_spmd_sharding_propagation_to_parameters()),
        spanToArrayRef(
            hloModule->config().allow_spmd_sharding_propagation_to_output()));
  }
  ComputationLayout flattenedEntryComputationLayout =
      getFlattenedComputationLayout(hloModule->entry_computation_layout(),
                                    useTupleArgs);
  OriginalParamIndexToFlattenedNum originalParamIndexToFlattenedNum =
      getOriginalParamIndexToFlattenedNum(hloModule);
  HloInputOutputAliasConfig flattenedInputOutputAliasConfig =
      getFlattenedInputOutputAliasConfig(hloModule->input_output_alias_config(),
                                         originalParamIndexToFlattenedNum,
                                         useTupleArgs);
  HloBufferDonorConfig flattenedBufferDonorsConfig =
      getFlattenedBufferDonorsConfig(hloModule->buffer_donor_config(),
                                     originalParamIndexToFlattenedNum,
                                     useTupleArgs);
  if (runSdyShardingPropagation) {
    pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
    mlir::sdy::addPropagationPipeline(
        pm, shardyDir,
        hloModule->use_auto_spmd_partitioning());
    pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
  }
  addMhloExportPipeline(pm);
  pm.addPass(mlir::sdy::createSaveModuleOpPass(shardyDir,
                                               "sdy_module_after_xla_export"));
  tsl::StatusScopedDiagnosticHandler diagnosticHandler(mlirContext.get());
  TF_RETURN_IF_ERROR(diagnosticHandler.consumeStatus(pm.run(*mlirModule)));
  HloProto hloProto;
  TF_RETURN_IF_ERROR(ConvertMlirHloToHlo(*mlirModule, &hloProto, useTupleArgs,
                                         false));
  TF_RETURN_IF_ERROR(
      createFromProtoAndReplaceComputations(hloModule, hloProto.hlo_module()));
  CHECK_OK(TupleSimplifier().Run(hloModule));
  *hloModule->mutable_entry_computation_layout() =
      flattenedEntryComputationLayout;
  hloModule->set_input_output_alias_config(
      std::move(flattenedInputOutputAliasConfig));
  hloModule->set_buffer_donor_config(std::move(flattenedBufferDonorsConfig));
  TF_RETURN_IF_ERROR(
      hlo_sharding_util::CanonicalizeLayoutAfterShardingPropagation(
          hloModule, true,
          true));
  removeFrontendAttributes(
      hloModule,
      {kUseTupleArgs, kPythonIntegrationComplete, kMeshesRoundTripAttr});
  return true;
}
}  
}  