#include "tensorflow/compiler/mlir/tf2xla/internal/clustering_bridge_passes.h"
#include <string>
#include "absl/log/log.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/Pass/PassManager.h"  
#include "mlir/Transforms/Passes.h"  
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/sparsecore/sparsecore_passes.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/passes/clustering_passes.h"
namespace tensorflow {
namespace tf2xla {
namespace internal {
using mlir::OpPassManager;
using mlir::func::FuncOp;
void AddReplicatedBridgeClusteringPipelinePasses(OpPassManager& pm,
                                                 llvm::StringRef module_name) {
  const llvm::SmallVector<std::string, 4> ops_to_preserve = {
      "tf.TPUReplicateMetadata", "tf.TPUCompilationResult",
      "tf.TPUReplicatedOutput"};
  bool strict_clusters =
      tensorflow::GetMlirCommonFlags()->tf_mlir_enable_strict_clusters;
  pm.addNestedPass<FuncOp>(
      mlir::tf_executor::CreateTFExecutorGraphPruningPass(ops_to_preserve));
  pm.addNestedPass<FuncOp>(
      mlir::CreateExecutorDialectToFunctionalConversionPass());
  pm.addPass(mlir::TF::CreateGuaranteeAllFuncsOneUsePass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  pm.addNestedPass<FuncOp>(mlir::TFTPU::CreateTPUPartitionedOpConversionPass());
  pm.addNestedPass<FuncOp>(
      mlir::TFTPU::CreateTPUReorderReplicateAndPartitionedInputsPass());
  pm.addNestedPass<FuncOp>(mlir::TF::CreateDecomposeReduceDatasetPass());
  pm.addPass(mlir::TFDevice::CreateEmbeddingPipeliningPass());
  pm.addPass(mlir::TFDevice::CreateEmbeddingSequencingPass());
  pm.addPass(tensorflow::tf2xla::internal::CreateTPUClusterFormationPass(
      strict_clusters));
  pm.addPass(mlir::TF::CreateGuaranteeAllFuncsOneUsePass());
  pm.addPass(mlir::TFTPU::CreateTPUClusterCleanupAttributesPass());
  pm.addNestedPass<FuncOp>(mlir::TFDevice::CreateDeviceAttributeToLaunchPass());
  pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::TFDevice::CreateDecomposeResourceOpsInClusterPass());
  {
    OpPassManager& func_pm = pm.nest<FuncOp>();
    func_pm.addPass(mlir::TFTPU::CreateTPUHostComputationExpansionPass());
    func_pm.addPass(mlir::TFTPU::CreateTPUUpdateEmbeddingEnqueueOpInputsPass());
  }
  pm.addNestedPass<FuncOp>(mlir::TFDevice::CreateLaunchToDeviceAttributePass());
  pm.addPass(mlir::TF::CreateTFFunctionalControlFlowToRegions());
  pm.addPass(mlir::createInlinerPass());
  pm.addNestedPass<FuncOp>(
      mlir::TF::CreateDropWhileShapeInvariantInDeviceClusterPass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::TFTPU::CreateTPUClusterCleanupAttributesPass());
  pm.addPass(mlir::TFDevice::CreateResourceOpLiftingPass());
  pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(mlir::createCSEPass());
  if (tensorflow::GetMlirCommonFlags()
          ->tf_mlir_enable_merge_control_flow_pass) {
    pm.addPass(mlir::TFDevice::CreateMergeControlFlowPass());
  }
  pm.addPass(
      tensorflow::tf2xla::internal::CreateMarkOpsForOutsideCompilationPass());
  pm.addPass(tensorflow::tf2xla::internal::
                 CreateExtractHeadTailOutsideCompilationPass());
  pm.addPass(
      tensorflow::tf2xla::internal::CreateExtractOutsideCompilationPass());
  pm.addNestedPass<FuncOp>(
      mlir::TFDevice::CreateVerifyNoOutsideCompilationMarkersPass());
  pm.addNestedPass<FuncOp>(mlir::TFDevice::CreateClusterConstantSinkingPass());
  pm.addPass(mlir::TF::CreateResourceDeviceInferencePass());
  pm.addNestedPass<FuncOp>(
      tensorflow::tf2xla::internal::CreateHoistBroadcastReadPass());
  pm.addNestedPass<FuncOp>(
      tensorflow::tf2xla::internal::CreateXlaBroadcastPass());
  pm.addPass(mlir::TFDevice::CreateClusterOutliningPass());
  pm.addPass(mlir::TFTPU::CreateTPUResourceReadForWritePass());
  pm.addPass(mlir::TFDevice::CreateMarkInputOutputAliasesPass());
  pm.addPass(
      tensorflow::tf2xla::internal::CreateTPUShardingIdentificationPass());
  pm.addNestedPass<FuncOp>(
      mlir::TFTPU::CreateTPUResourceReadsWritesPartitioningPass());
  pm.addPass(mlir::TFDevice::CreateAnnotateParameterReplicationPass());
  pm.addNestedPass<FuncOp>(mlir::TF::CreateRewriteTPUEmbeddingOpsPass());
  pm.addPass(mlir::TFTPU::CreateTPUAnnotateDynamicShapeInputsPass());
  pm.addNestedPass<FuncOp>(
      mlir::TF::CreateHoistReplicateInvariantResourceWritesPass());
  pm.addNestedPass<FuncOp>(
      tensorflow::tf2xla::internal::CreateVerifyClusteringPass());
}
void NoCanonicalization(OpPassManager& pm) {}
void AddNonReplicatedBridgeClusteringPipelinePasses(OpPassManager& pm) {
  VLOG(2) << "Create TF XLA Bridge pipeline";
  pm.addPass(mlir::TFDevice::CreateXlaValidateInputsPass());
  pm.addNestedPass<FuncOp>(
      mlir::TF::CreateCanonicalizeCompileAndReplicateAttributesPass());
  const llvm::SmallVector<std::string, 4> ops_to_preserve = {};
  pm.addNestedPass<FuncOp>(
      mlir::tf_executor::CreateTFExecutorGraphPruningPass(ops_to_preserve));
  pm.addNestedPass<FuncOp>(
      mlir::CreateExecutorDialectToFunctionalConversionPass());
  pm.addPass(mlir::TF::CreateGuaranteeAllFuncsOneUsePass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  pm.addPass(tensorflow::tf2xla::internal::CreateXlaClusterFormationPass());
  pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::TFDevice::CreateDecomposeResourceOpsInClusterPass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createInlinerPass({}, NoCanonicalization));
  pm.addPass(mlir::TFDevice::CreateResourceOpLiftingPass());
  pm.addPass(mlir::TFDevice::CreateClusterOutliningPass());
  pm.addNestedPass<FuncOp>(
      tensorflow::tf2xla::internal::CreateVerifyClusteringPass());
}
};  
};  
};  