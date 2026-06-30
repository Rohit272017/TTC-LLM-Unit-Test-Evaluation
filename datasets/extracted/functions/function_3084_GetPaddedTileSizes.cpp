#include "xla/service/gpu/model/triton_emitter_constraints.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/AffineMap.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/model/affine_map_evaluator.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/model/symbolic_tile.h"
#include "xla/service/gpu/model/symbolic_tile_analysis.h"
#include "xla/service/gpu/model/symbolic_tiled_hlo_instruction.h"
#include "xla/stream_executor/device_description.h"
namespace xla {
namespace gpu {
namespace {
constexpr int64_t kMaxTensorNumElements = 1048576;
llvm::SmallVector<int64_t> GetPaddedTileSizes(
    llvm::SmallVector<int64_t> tile_sizes) {
  llvm::SmallVector<int64_t> result;
  result.reserve(tile_sizes.size());
  for (int64_t value : tile_sizes) {
    result.push_back(llvm::PowerOf2Ceil(value));
  }
  return result;
}
}  
 std::vector<TritonEmitterConstraints::CustomConstraints>
TritonEmitterConstraints::DeriveCustomConstraints(
    const std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>&
        instructions,
    const HloFusionAdaptor& fusion_adaptor) {
  std::vector<CustomConstraints> result;
  for (const auto& instruction : instructions) {
    const HloInstruction* hlo = instruction->hlo();
    if (hlo->opcode() == HloOpcode::kReshape ||
        hlo->opcode() == HloOpcode::kBitcast) {
      if (!fusion_adaptor.ContainsInstruction(hlo)) {
        continue;
      }
      mlir::MLIRContext* ctx =
          instruction->symbolic_tile().size_map().getContext();
      IndexingMap reshape_indexing_map =
          *ComputeOutputToInputIndexing(hlo, 0, ctx)
               .indexing_maps[0]
               .begin();
      std::optional<SymbolicTile> reshape_symbolic_tile =
          SymbolicTile::FromIndexingMap(reshape_indexing_map);
      CHECK(reshape_symbolic_tile.has_value());
      ConstraintExpression reshape_constraints =
          reshape_symbolic_tile->constraints();
      result.push_back(
          CustomConstraints{instruction->symbolic_tile().size_map(),
                            std::move(reshape_constraints)});
    }
  }
  return result;
}
 EmitterSpecificConstraintsBuilder
TritonEmitterConstraints::GetBuilder(
    const se::DeviceDescription& device_description) {
  return [=](const std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>&
                 instructions,
             const HloFusionAdaptor& fusion_adaptor) {
    llvm::DenseSet<mlir::AffineMap> unique_tile_size_maps;
    for (const auto& tiled_hlo_instruction : instructions) {
      unique_tile_size_maps.insert(
          tiled_hlo_instruction->symbolic_tile().size_map());
    }
    std::vector<CustomConstraints> custom_constraints =
        DeriveCustomConstraints(instructions, fusion_adaptor);
    llvm::SmallVector<mlir::AffineMap, 4> tile_size_maps(
        unique_tile_size_maps.begin(), unique_tile_size_maps.end());
    return std::unique_ptr<TritonEmitterConstraints>(
        absl::WrapUnique(new TritonEmitterConstraints(
            std::move(tile_size_maps), std::move(custom_constraints),
            instructions.back()->hlo()->shape(),
            device_description)));
  };
}
absl::StatusOr<bool> TritonEmitterConstraints::ParametersSatisfyConstraints(
    absl::Span<const int64_t> tile_parameters) const {
  for (const auto& tile_size_map : tile_size_maps_) {
    int64_t tile_size = 1;
    for (auto expr : tile_size_map.getResults()) {
      tile_size *= llvm::PowerOf2Ceil(
          EvaluateAffineExpr(expr, tile_parameters));
    }
    if (tile_size > kMaxTensorNumElements) {
      return false;
    }
  }
  int64_t num_tiles = 1;
  for (auto [dim_size, tile_size] :
       llvm::zip(root_shape_.dimensions(), tile_parameters)) {
    num_tiles *= (dim_size + tile_size - 1) / tile_size;
  }
  if (num_tiles >= device_info_.block_dim_limit().x) {
    return false;
  }
  for (const auto& custom_constraint : custom_constraints_) {
    llvm::SmallVector<int64_t> transformed_tile_parameters =
        EvaluateAffineMap(custom_constraint.tile_parameters_transform,
                          tile_parameters);
    if (!custom_constraint.constraints.IsSatisfiedBy(
            GetPaddedTileSizes(transformed_tile_parameters))) {
      return false;
    }
  }
  return true;
}
}  
}  