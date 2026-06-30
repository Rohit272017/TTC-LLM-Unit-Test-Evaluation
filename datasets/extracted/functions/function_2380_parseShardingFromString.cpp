#include "xla/service/spmd/shardy/mhlo_round_trip/mhlo_import.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/hlo/translate/mhlo_to_hlo/attribute_exporter.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/mhlo_round_trip/shard_map_import.h"
#include "xla/service/spmd/shardy/round_trip_common/pipeline_passes.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
namespace xla {
namespace sdy {
namespace {
using ::llvm::SmallDenseMap;
using ::llvm::SmallDenseSet;
using ::mlir::ArrayRef;
using ::mlir::LogicalResult;
using ::mlir::ModuleOp;
using ::mlir::OpBuilder;
using ::mlir::OperationPass;
using ::mlir::Pass;
using ::mlir::PassWrapper;
using ::mlir::ShapedType;
using ::mlir::SmallVector;
using ::mlir::StringAttr;
using ::mlir::StringRef;
using ::mlir::func::FuncOp;
using ::mlir::sdy::AxisRefAttr;
using ::mlir::sdy::DimensionShardingAttr;
using ::mlir::sdy::kShardingAttr;
using ::mlir::sdy::MeshAttr;
using ::mlir::sdy::MeshAxisAttr;
using ::mlir::sdy::MeshOp;
using ::mlir::sdy::SdyDialect;
using ::mlir::sdy::TensorShardingAttr;
using ::mlir::sdy::TensorShardingPerValueAttr;
struct SubDimInfo {
  int64_t tileDimIndex;     
  int64_t tileSubDimIndex;  
  int64_t reshapeDimIndex;  
  int64_t size;             
};
struct AnalyzeTileAssignmentResult {
  SmallVector<SubDimInfo> subDims;
  SmallVector<int64_t> localMesh;
};
}  
xla::HloSharding parseShardingFromString(const StringAttr& sharding) {
  const std::optional<xla::OpSharding> shardingProto =
      xla::ConvertSharding(sharding.getValue());
  CHECK(shardingProto) << sharding.getValue().str();
  absl::StatusOr<HloSharding> hloSharding =
      xla::HloSharding::FromProto(*shardingProto);
  CHECK_OK(hloSharding) << shardingProto->DebugString();
  return *hloSharding;
}
namespace {
SmallVector<int64_t> shortestCommonFactorization(ArrayRef<int64_t> array1,
                                                 ArrayRef<int64_t> array2) {
  SmallVector<int64_t> result;
  result.reserve(std::max(array1.size(), array2.size()));
  auto nextIndexWithNonOneElement = [](ArrayRef<int64_t> array,
                                       int64_t index) -> int64_t {
    while (index < array.size() && array[index] == 1) {
      index++;
    }
    return index;
  };
  int64_t index1 = nextIndexWithNonOneElement(array1, 0);
  int64_t index2 = nextIndexWithNonOneElement(array2, 0);
  int64_t nextStride1 = 1;
  int64_t nextStride2 = 1;
  int64_t accumulatedFactor = 1;
  while (index1 < array1.size() || index2 < array2.size()) {
    if (index1 < array1.size() && nextStride1 == accumulatedFactor) {
      nextStride1 *= array1[index1++];
    }
    if (index2 < array2.size() && nextStride2 == accumulatedFactor) {
      nextStride2 *= array2[index2++];
    }
    const auto [smallFactor, largeFactor] = std::minmax(
        {nextStride1 / accumulatedFactor, nextStride2 / accumulatedFactor});
    if (largeFactor % smallFactor != 0 || smallFactor == 1) {
      return {};
    }
    result.push_back(smallFactor);
    accumulatedFactor *= smallFactor;
    CHECK_EQ(accumulatedFactor, Product(result));
    index1 = nextIndexWithNonOneElement(array1, index1);
    index2 = nextIndexWithNonOneElement(array2, index2);
  }
  return result;
}
SmallVector<SubDimInfo> getOrderedSubDimsFromIotaTileAssignment(
    const xla::IotaTileAssignment& iota) {
  SmallVector<int64_t> deviceShape(iota.transpose_perm().size());
  for (auto [index, perm_i] : llvm::enumerate(iota.transpose_perm())) {
    deviceShape[index] = iota.reshape_dims()[perm_i];
  }
  const SmallVector<int64_t> axisSizes = shortestCommonFactorization(
      ArrayRef<int64_t>(iota.dims().begin(), iota.dims().end()), deviceShape);
  if (axisSizes.empty()) {
    return {};
  }
  SmallVector<SubDimInfo> subDims;
  subDims.reserve(axisSizes.size());
  int64_t tileDimIndex = iota.ndims() - 1;
  int64_t transPermIndex = iota.transpose_perm().size() - 1;
  int64_t accTileSize = 1;
  int64_t accDeviceSize = 1;
  int64_t subDim = 0;
  for (const int64_t axisSize : llvm::reverse(axisSizes)) {
    while (iota.dim(tileDimIndex) == 1) {
      tileDimIndex--;
    }
    subDims.push_back(SubDimInfo{
         tileDimIndex,
         subDim++,
         iota.transpose_perm()[transPermIndex],
         axisSize,
    });
    accTileSize *= axisSize;
    accDeviceSize *= axisSize;
    if (iota.dim(tileDimIndex) == accTileSize) {
      tileDimIndex--;
      accTileSize = 1;
      subDim = 0;
    }
    if (deviceShape[transPermIndex] == accDeviceSize) {
      accDeviceSize = 1;
      transPermIndex--;
    }
  }
  absl::c_sort(subDims, [](const SubDimInfo& a, const SubDimInfo& b) {
    return std::forward_as_tuple(a.reshapeDimIndex, a.tileDimIndex) <
           std::forward_as_tuple(b.reshapeDimIndex, b.tileDimIndex);
  });
  return subDims;
}
AnalyzeTileAssignmentResult analyzeTileAssignment(
    const xla::TileAssignment& tileAssignment) {
  const std::optional<IotaTileAssignment>& iota = tileAssignment.iota();
  CHECK(iota.has_value()) << "tile assignment: " << tileAssignment.ToString();
  const SmallVector<SubDimInfo> subDims =
      getOrderedSubDimsFromIotaTileAssignment(*iota);
  CHECK(!subDims.empty()) << "tile assignment: " << tileAssignment.ToString();
  SmallVector<int64_t> mesh;
  mesh.reserve(subDims.size());
  for (SubDimInfo subDimInfo : subDims) {
    mesh.push_back(subDimInfo.size);
  }
  return AnalyzeTileAssignmentResult{
       std::move(subDims),
       std::move(mesh),
  };
}
absl::flat_hash_set<xla::HloSharding> collectXlaHloShardings(
    ModuleOp moduleOp) {
  absl::flat_hash_set<xla::HloSharding> oldShardings;
  for (FuncOp funcOp : moduleOp.getOps<FuncOp>()) {
    for (int64_t argNum = 0; argNum < funcOp.getNumArguments(); ++argNum) {
      if (auto oldSharding =
              funcOp.getArgAttrOfType<StringAttr>(argNum, kXlaShardingAttr)) {
        oldShardings.insert(parseShardingFromString(oldSharding));
      }
    }
    for (int64_t resNum = 0; resNum < funcOp.getNumResults(); ++resNum) {
      if (auto oldSharding = funcOp.getResultAttrOfType<StringAttr>(
              resNum, kXlaShardingAttr)) {
        oldShardings.insert(parseShardingFromString(oldSharding));
      }
    }
    funcOp.front().walk([&](mlir::Operation* op) {
      if (auto oldSharding = op->getAttrOfType<StringAttr>(kXlaShardingAttr)) {
        const xla::HloSharding hloSharding =
            parseShardingFromString(oldSharding);
        if (hloSharding.IsTuple()) {
          for (const xla::HloSharding& element : hloSharding.tuple_elements()) {
            oldShardings.insert(element);
          }
        } else {
          oldShardings.insert(hloSharding);
        }
      }
    });
  }
  return oldShardings;
}
struct MeshAxesAndIds {
  SmallVector<MeshAxisAttr> namedAxes;
  SmallVector<int64_t> maximalDeviceIds;
};
MeshAxesAndIds findMeshAxesAndIds(ModuleOp moduleOp) {
  MeshAxesAndIds result;
  auto& [namedAxes, maximalDeviceIds] = result;
  const absl::flat_hash_set<xla::HloSharding> oldShardings =
      collectXlaHloShardings(moduleOp);
  SmallVector<int64_t> axes;
  llvm::SmallDenseSet<int64_t> maximalDeviceIdSet;
  for (const xla::HloSharding& hloSharding : oldShardings) {
    if (hloSharding.HasUniqueDevice()) {
      maximalDeviceIdSet.insert(hloSharding.GetUniqueDevice());
      continue;
    }
    CHECK(!hloSharding.IsTuple());
    if (hloSharding.IsReplicated() || hloSharding.IsManual() ||
        hloSharding.IsUnknown()) {
      continue;
    }
    CHECK(hloSharding.IsTiled());
    const AnalyzeTileAssignmentResult result =
        analyzeTileAssignment(hloSharding.tile_assignment());
    axes = (axes.empty()) ? result.localMesh
                          : shortestCommonFactorization(result.localMesh, axes);
    CHECK(!axes.empty());
  }
  namedAxes.reserve(axes.size());
  for (auto [axisIndex, axisSize] : llvm::enumerate(axes)) {
    auto name = StringAttr::get(moduleOp->getContext(),
                                absl::StrCat("axis_", axisIndex));
    namedAxes.push_back(
        MeshAxisAttr::get(moduleOp->getContext(), name, axisSize));
  }
  maximalDeviceIds = llvm::to_vector(maximalDeviceIdSet);
  llvm::sort(maximalDeviceIds);
  return result;
}
}  
TensorShardingAttr convertToSdySharding(
    const xla::HloSharding& hloSharding, MeshAttr globalMesh,
    const SmallDenseMap<int64_t, StringRef>& deviceIdToMaximalMeshName,
    int64_t rank, bool openDims) {
  mlir::MLIRContext* ctx = globalMesh.getContext();
  if (hloSharding.HasUniqueDevice()) {
    return TensorShardingAttr::getFullyClosed(
        ctx, rank,
        deviceIdToMaximalMeshName.lookup(hloSharding.GetUniqueDevice()));
  }
  CHECK(!hloSharding.IsTuple());
  if (hloSharding.IsReplicated() || hloSharding.IsManual() ||
      hloSharding.IsUnknown()) {
    return hloSharding.IsUnknown() || openDims
               ? TensorShardingAttr::getFullyOpen(ctx, rank, kGlobalMeshName)
               : TensorShardingAttr::getFullyClosed(ctx, rank, kGlobalMeshName);
  }
  CHECK(hloSharding.IsTiled());
  const AnalyzeTileAssignmentResult result =
      analyzeTileAssignment(hloSharding.tile_assignment());
  SmallVector<SmallVector<AxisRefAttr>> localAxisIndexToGlobalAxes;
  localAxisIndexToGlobalAxes.reserve(result.localMesh.size());
  int64_t globalAxisIndex = 0;
  for (int64_t localAxisSize : result.localMesh) {
    SmallVector<AxisRefAttr>& globalAxes =
        localAxisIndexToGlobalAxes.emplace_back();
    int64_t product = 1;
    while (product < localAxisSize) {
      MeshAxisAttr axisAttr = globalMesh.getAxes()[globalAxisIndex++];
      if (axisAttr.getSize() == 1) {
        continue;
      }
      globalAxes.push_back(AxisRefAttr::get(ctx, axisAttr.getName()));
      product *= axisAttr.getSize();
    }
    CHECK_EQ(product, localAxisSize);
  }
  SmallVector<SmallVector<int64_t>> dimToSubDimToLocalAxisIndex(rank);
  for (auto [localAxisIndex, subDimInfo] : llvm::enumerate(result.subDims)) {
    if (subDimInfo.tileDimIndex >= rank) {
      continue;
    }
    SmallVector<int64_t>& subDimToLocalAxisIndex =
        dimToSubDimToLocalAxisIndex[subDimInfo.tileDimIndex];
    if (subDimInfo.tileSubDimIndex >= subDimToLocalAxisIndex.size()) {
      subDimToLocalAxisIndex.resize(subDimInfo.tileSubDimIndex + 1);
    }
    subDimToLocalAxisIndex[subDimInfo.tileSubDimIndex] = localAxisIndex;
  }
  SmallVector<DimensionShardingAttr> dimShardings;
  dimShardings.reserve(rank);
  for (ArrayRef<int64_t> subDimToLocalAxisIndex : dimToSubDimToLocalAxisIndex) {
    SmallVector<AxisRefAttr> axes;
    for (int64_t localAxisIndex : llvm::reverse(subDimToLocalAxisIndex)) {
      absl::c_copy(localAxisIndexToGlobalAxes[localAxisIndex],
                   std::back_inserter(axes));
    }
    dimShardings.push_back(
        DimensionShardingAttr::get(ctx, axes, !openDims));
  }
  return TensorShardingAttr::get(ctx, StringAttr::get(ctx, kGlobalMeshName),
                                 dimShardings, {});
}
namespace {
bool shouldOpenDims(ArrayRef<bool> allowPropagationToTensors, int64_t index) {
  if (allowPropagationToTensors.empty()) {
    return false;
  }
  if (allowPropagationToTensors.size() == 1) {
    return allowPropagationToTensors.front();
  }
  CHECK_LT(index, allowPropagationToTensors.size());
  return allowPropagationToTensors[index];
}
LogicalResult importShardings(
    FuncOp funcOp, MeshAttr globalMesh,
    const SmallDenseMap<int64_t, StringRef>& deviceIdToMaximalMeshName,
    ArrayRef<bool> allowPropagationToArgs,
    ArrayRef<bool> allowPropagationToResults) {
  for (auto [argNum, argType] : llvm::enumerate(funcOp.getArgumentTypes())) {
    if (auto oldSharding =
            funcOp.getArgAttrOfType<StringAttr>(argNum, kXlaShardingAttr)) {
      funcOp.setArgAttr(
          argNum, kShardingAttr,
          convertToSdySharding(parseShardingFromString(oldSharding), globalMesh,
                               deviceIdToMaximalMeshName,
                               mlir::cast<ShapedType>(argType).getRank(),
                               shouldOpenDims(allowPropagationToArgs, argNum)));
      funcOp.removeArgAttr(argNum, kXlaShardingAttr);
    }
  }
  for (auto [resNum, resType] : llvm::enumerate(funcOp.getResultTypes())) {
    if (auto oldSharding =
            funcOp.getResultAttrOfType<StringAttr>(resNum, kXlaShardingAttr)) {
      funcOp.setResultAttr(
          resNum, kShardingAttr,
          convertToSdySharding(
              parseShardingFromString(oldSharding), globalMesh,
              deviceIdToMaximalMeshName,
              mlir::cast<ShapedType>(resType).getRank(),
              shouldOpenDims(allowPropagationToResults, resNum)));
      funcOp.removeResultAttr(
          resNum, StringAttr::get(funcOp.getContext(), kXlaShardingAttr));
    }
  }
  funcOp.front().walk([&](mlir::Operation* op) {
    if (auto oldSharding = op->getAttrOfType<StringAttr>(kXlaShardingAttr)) {
      const xla::HloSharding hloSharding = parseShardingFromString(oldSharding);
      ArrayRef<xla::HloSharding> flatHloSharding = hloSharding;
      if (hloSharding.IsTuple()) {
        flatHloSharding = hloSharding.tuple_elements();
      }
      SmallVector<TensorShardingAttr> newShardings;
      newShardings.reserve(op->getNumResults());
      for (const auto& [resHloSharding, resType] :
           llvm::zip_equal(flatHloSharding, op->getResultTypes())) {
        newShardings.push_back(convertToSdySharding(
            resHloSharding, globalMesh, deviceIdToMaximalMeshName,
            mlir::cast<ShapedType>(resType).getRank(),
            false));
      }
      op->setAttr(kShardingAttr, TensorShardingPerValueAttr::get(
                                     globalMesh.getContext(), newShardings));
      op->removeAttr(kXlaShardingAttr);
    }
  });
  return mlir::success();
}
class ImportShardingsPass
    : public PassWrapper<ImportShardingsPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImportShardingsPass)
  ImportShardingsPass(ArrayRef<bool> allowPropagationToArgs,
                      ArrayRef<bool> allowPropagationToResults)
      : allowPropagationToArgs(allowPropagationToArgs),
        allowPropagationToResults(allowPropagationToResults) {}
  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    auto [namedAxes, deviceIdsForMaximalMesh] = findMeshAxesAndIds(moduleOp);
    if (namedAxes.empty() && deviceIdsForMaximalMesh.empty()) {
      return;
    }
    mlir::SymbolTableCollection symbolTableCollection;
    mlir::SymbolTable& symbolTable =
        symbolTableCollection.getSymbolTable(moduleOp);
    OpBuilder opBuilder = mlir::OpBuilder::atBlockBegin(moduleOp.getBody());
    symbolTable.insert(opBuilder.create<MeshOp>(
        moduleOp.getLoc(), kGlobalMeshName,
        MeshAttr::get(moduleOp.getContext(), namedAxes)));
    SmallDenseMap<int64_t, StringRef> deviceIdToMaximalMeshName;
    for (int64_t deviceId : deviceIdsForMaximalMesh) {
      std::string meshName = absl::StrCat("maximal_mesh_", deviceId);
      auto meshOp = opBuilder.create<MeshOp>(
          moduleOp.getLoc(), meshName,
          MeshAttr::get(moduleOp.getContext(), deviceId));
      symbolTable.insert(meshOp);
      deviceIdToMaximalMeshName[deviceId] = meshOp.getSymName();
    }
    for (FuncOp funcOp : moduleOp.getOps<FuncOp>()) {
      bool isMain = funcOp.getSymName() == "main";
      MeshAttr globalMesh = MeshAttr::get(moduleOp.getContext(), namedAxes);
      if (mlir::failed(importShardings(
              funcOp, globalMesh, deviceIdToMaximalMeshName,
              isMain ? allowPropagationToArgs : ArrayRef<bool>(),
              isMain ? allowPropagationToResults : ArrayRef<bool>()))) {
        signalPassFailure();
      }
    }
  }
  StringRef getArgument() const override { return "xla-sdy-import-shardings"; }
  StringRef getDescription() const override {
    return "Builds the mesh and converts the shardings from kXlaShardingAttr "
           "to kShardingAttr.";
  }
  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<SdyDialect>();
  }
 private:
  ArrayRef<bool> allowPropagationToArgs;
  ArrayRef<bool> allowPropagationToResults;
};
std::unique_ptr<mlir::Pass> createImportShardingsPass(
    ArrayRef<bool> allowPropagationToArgs,
    ArrayRef<bool> allowPropagationToResults) {
  return std::make_unique<ImportShardingsPass>(allowPropagationToArgs,
                                               allowPropagationToResults);
}
}  
void registerMhloImportShardingsPass() {
  mlir::registerPass(
      std::bind(createImportShardingsPass, ArrayRef<bool>(), ArrayRef<bool>()));
}
void addMhloImportPipeline(mlir::OpPassManager& pm,
                           ArrayRef<bool> allowPropagationToArgs,
                           ArrayRef<bool> allowPropagationToResults) {
  addCommonPreImportPasses(pm);
  pm.addPass(createImportShardingsPass(allowPropagationToArgs,
                                       allowPropagationToResults));
  pm.addPass(createMhloRoundTripShardMapImportPass());
  addCommonPostImportPasses(pm);
}
void registerMhloImportPipeline() {
  mlir::PassPipelineRegistration<> importPipeline(
      "xla-sdy-mhlo-import-pipeline",
      "Run passes to import an mhlo module with `mhlo.shardings` into the SDY "
      "(Shardy) dialect.",
      std::bind(addMhloImportPipeline, std::placeholders::_1, ArrayRef<bool>(),
                ArrayRef<bool>()));
}
}  
}  