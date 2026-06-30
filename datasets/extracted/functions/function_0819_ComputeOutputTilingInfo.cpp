#include "xla/service/gpu/model/symbolic_tile_analysis.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/model/indexing_map_serialization.h"
#include "xla/service/gpu/model/symbolic_tile.h"
#include "xla/service/gpu/model/symbolic_tiled_hlo_instruction.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/model/tiled_hlo_instruction.h"
#include "xla/service/instruction_fusion.h"
#include "xla/service/name_uniquer.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace gpu {
namespace {
using ::mlir::AffineExpr;
using ::mlir::MLIRContext;
struct OutputTilingInfo {
  llvm::SmallVector<int64_t> num_output_tiles_per_dim;
  IndexingMap output_tile_offset_indexing;
};
OutputTilingInfo ComputeOutputTilingInfo(absl::Span<const int64_t> dimensions,
                                         absl::Span<const int64_t> tile_sizes,
                                         mlir::MLIRContext* mlir_context) {
  CHECK_EQ(dimensions.size(), tile_sizes.size());  
  int64_t num_tiles = 1;
  llvm::SmallVector<int64_t> outer_loop_bounds;
  outer_loop_bounds.reserve(dimensions.size());
  for (auto [dim_size, tile_size] : llvm::zip(dimensions, tile_sizes)) {
    int64_t num_tiles_per_dim = (dim_size + tile_size - 1) / tile_size;
    num_tiles *= num_tiles_per_dim;
    outer_loop_bounds.push_back(num_tiles_per_dim);
  }
  llvm::SmallVector<AffineExpr> tiled_dims;
  tiled_dims.reserve(dimensions.size());
  for (auto [dim_id, tile_size] : llvm::enumerate(tile_sizes)) {
    tiled_dims.push_back(tile_size *
                         mlir::getAffineDimExpr(dim_id, mlir_context));
  }
  IndexingMap output_tile_offset_indexing = IndexingMap::FromTensorSizes(
      mlir::AffineMap::get(
          dimensions.size(), 0, tiled_dims,
          mlir_context),
      outer_loop_bounds, {});
  return {outer_loop_bounds, output_tile_offset_indexing};
}
absl::StatusOr<IndexingMap> ComputeTileOffsetIndexing(
    const SymbolicTiledHloInstruction& tiled_hlo,
    const IndexingMap& output_tile_offset_indexing,
    mlir::MLIRContext* mlir_context) {
  IndexingMap tile_offset_indexing = ComposeIndexingMaps(
      output_tile_offset_indexing, tiled_hlo.indexing_map());
  if (absl::c_any_of(tile_offset_indexing.GetSymbolBounds(),
                     [](const Interval& symbol_bound) {
                       return symbol_bound.lower != 0;
                     })) {
    return absl::FailedPreconditionError(
        absl::StrCat("Symbol lower bound is not zero. ",
                     ToString(tiled_hlo.indexing_map())));
  }
  std::vector<AffineExpr> symbol_lower_bounds(
      tile_offset_indexing.GetRangeVarsCount(),
      mlir::getAffineConstantExpr(0, mlir_context));
  symbol_lower_bounds.reserve(tile_offset_indexing.GetSymbolCount());
  for (int i = 0; i < tile_offset_indexing.GetRTVarsCount(); ++i) {
    symbol_lower_bounds.push_back(mlir::getAffineSymbolExpr(i, mlir_context));
  }
  mlir::AffineMap simplified_affine_map =
      tile_offset_indexing.GetAffineMap().replaceDimsAndSymbols(
          {},
          symbol_lower_bounds,
          tile_offset_indexing.GetDimVarsCount(),
          tile_offset_indexing.GetRTVarsCount());
  IndexingMap simplified_indexing_map =
      IndexingMap{simplified_affine_map, tile_offset_indexing.GetDimVars(),
                  {}, tile_offset_indexing.GetRTVars()};
  simplified_indexing_map.Simplify();
  simplified_indexing_map.RescaleSymbols();
  simplified_indexing_map.RemoveUnusedSymbols();
  return simplified_indexing_map;
}
template <typename T>
class OrderedUniquePtrValueHashSet {
 public:
  std::pair<T*, bool> Insert(std::unique_ptr<T> elem) {
    auto [it, inserted] = hash_set_.insert(elem.get());
    if (inserted) {
      data_.push_back(std::move(elem));
    }
    return {*it, inserted};
  }
  void Reserve(int64_t n) {
    hash_set_.reserve(n);
    data_.reserve(n);
  }
  std::vector<std::unique_ptr<T>> ExtractData() { return std::move(data_); }
 private:
  struct PtrHash {
    size_t operator()(const T* v) const { return absl::HashOf(*v); }
  };
  struct PtrEqual {
    bool operator()(const T* lhs, const T* rhs) const {
      return lhs == rhs || *lhs == *rhs;
    }
  };
  absl::flat_hash_set<T*, PtrHash, PtrEqual> hash_set_;
  std::vector<std::unique_ptr<T>> data_;
};
FusionDecision ShouldProceedWithSymbolicTileDerivation(
    const SymbolicTiledHloInstruction& tiled_hlo_instruction) {
  const HloInstruction* hlo = tiled_hlo_instruction.hlo();
  const IndexingMap& indexing_map = tiled_hlo_instruction.indexing_map();
  if (hlo->opcode() == HloOpcode::kConcatenate) {
    return FusionDecision::Forbid("Bailing out on ") << hlo->ToString();
  }
  if (hlo->opcode() == HloOpcode::kReshape ||
      hlo->opcode() == HloOpcode::kBitcast) {
    mlir::MLIRContext* ctx = indexing_map.GetMLIRContext();
    IndexingMap reshape_indexing_map =
        *ComputeOutputToInputIndexing(hlo, 0, ctx)
             .indexing_maps[0]
             .begin();
    std::optional<SymbolicTile> reshape_symbolic_tile =
        SymbolicTile::FromIndexingMap(reshape_indexing_map);
    if (!reshape_symbolic_tile.has_value()) {
      return FusionDecision::Forbid("Bailing out on reshape ")
             << hlo->ToString() << " with indexing map "
             << ToString(reshape_indexing_map);
    }
  }
  return FusionDecision::Allow();
}
std::variant<ConstraintExpression, FusionDecision>
SetSymbolicTilesAndComputeConstraints(
    std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>&
        tiled_hlo_instructions,
    const HloFusionAdaptor& fusion_adaptor) {
  ConstraintExpression constraints;
  for (const std::unique_ptr<SymbolicTiledHloInstruction>&
           tiled_hlo_instruction : tiled_hlo_instructions) {
    const HloInstruction* hlo = tiled_hlo_instruction->hlo();
    const IndexingMap& indexing_map = tiled_hlo_instruction->indexing_map();
    if (fusion_adaptor.ContainsInstruction(hlo)) {
      FusionDecision should_proceed =
          ShouldProceedWithSymbolicTileDerivation(*tiled_hlo_instruction);
      if (!should_proceed) {
        return should_proceed;
      }
    }
    auto symbolic_tile = SymbolicTile::FromIndexingMap(indexing_map);
    if (!symbolic_tile.has_value()) {
      return FusionDecision::Forbid("Failed to compute symbolic tile for ")
             << ToString(indexing_map) << " for HLO " << hlo->ToString();
    }
    if (!symbolic_tile->is_satisfiable()) {
      return FusionDecision::Forbid("Symbolic tile ")
             << symbolic_tile->ToString() << " is not satisfiable for "
             << ToString(indexing_map) << " for HLO " << hlo->ToString();
    }
    constraints = ConstraintExpression::And(std::move(constraints),
                                            symbolic_tile->constraints());
    constraints.Simplify();
    if (!constraints.is_satisfiable()) {
      return FusionDecision::Forbid("Fusion has unsatisfiable constraints");
    }
    tiled_hlo_instruction->set_symbolic_tile(*std::move(symbolic_tile));
  }
  return constraints;
}
void SortTiledHloInstructionsInPostOrder(
    std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>&
        tiled_hlo_instructions,
    const SymbolicTiledHloInstruction* root_tiled_hlo) {
  absl::flat_hash_map<const SymbolicTiledHloInstruction*, int64_t>
      topological_order;
  std::function<void(const SymbolicTiledHloInstruction*)> visit_instruction;
  visit_instruction = [&](const SymbolicTiledHloInstruction* instruction) {
    if (topological_order.contains(instruction)) {
      return;
    }
    for (const SymbolicTiledHloInstruction* operand : instruction->operands()) {
      visit_instruction(operand);
    }
    topological_order[instruction] = topological_order.size();
  };
  visit_instruction(root_tiled_hlo);
  absl::c_sort(tiled_hlo_instructions,
               [&](const std::unique_ptr<SymbolicTiledHloInstruction>& t1,
                   const std::unique_ptr<SymbolicTiledHloInstruction>& t2) {
                 return topological_order.at(t1.get()) <
                        topological_order.at(t2.get());
               });
}
}  
 SymbolicTileAnalysisOrError SymbolicTileAnalysis::AnalyzeComputation(
    const HloComputation& computation, MLIRContext* ctx,
    EmitterSpecificConstraintsBuilder emitter_specific_constraints_builder) {
  auto fusion = HloFusionAdaptor::ForComputation(&computation);
  return SymbolicTileAnalysis::AnalyzeFusion(
      *fusion, ctx, emitter_specific_constraints_builder);
}
 SymbolicTileAnalysisOrError SymbolicTileAnalysis::AnalyzeFusion(
    const HloFusionAdaptor& fusion, MLIRContext* ctx,
    EmitterSpecificConstraintsBuilder emitter_specific_constraints_builder) {
  OrderedUniquePtrValueHashSet<SymbolicTiledHloInstruction>
      tiled_hlo_instructions_set;
  auto roots = fusion.GetRoots();
  if (roots.size() > 1) {
    return FusionDecision::Forbid("Multi-output fusions are not supported. ")
           << fusion.ToString();
  }
  auto& root = roots[0];
  auto [root_tiled_hlo, _] = tiled_hlo_instructions_set.Insert(
      std::make_unique<SymbolicTiledHloInstruction>(
          &root.instruction(), CreateIdentityMap(root.shape(), ctx)));
  std::vector<SymbolicTiledHloInstruction*> worklist = {root_tiled_hlo};
  while (!worklist.empty()) {
    auto tiled_hlo_instruction = worklist.back();
    worklist.pop_back();
    HloInstructionAdaptor instruction_adaptor(*tiled_hlo_instruction->hlo(),
                                              &fusion);
    if (!fusion.ContainsInstruction(instruction_adaptor)) {
      continue;
    }
    HloInstructionIndexing operands_indexing =
        ComputeOutputToInputIndexing(tiled_hlo_instruction->hlo(),
                                     0, ctx);
    for (auto [operand, operand_indexing_map_set] :
         llvm::zip(instruction_adaptor.GetOperands(),
                   operands_indexing.indexing_maps)) {
      CHECK_EQ(operand_indexing_map_set.size(), 1);  
      IndexingMap operand_indexing_map =
          ComposeIndexingMaps(tiled_hlo_instruction->indexing_map(),
                              *operand_indexing_map_set.begin());
      if (operand_indexing_map.IsUndefined()) {
        return FusionDecision::Forbid(
                   "Couldn't derive indexing map for instruction ")
               << tiled_hlo_instruction->hlo()->ToString() << " and operand "
               << operand.instruction().ToString();
      }
      operand_indexing_map.Simplify();
      operand_indexing_map.RescaleSymbols();
      operand_indexing_map.RemoveUnusedSymbols();
      auto [operand_tiled_hlo, inserted] = tiled_hlo_instructions_set.Insert(
          std::make_unique<SymbolicTiledHloInstruction>(
              &operand.instruction(), std::move(operand_indexing_map)));
      tiled_hlo_instruction->AppendOperand(operand_tiled_hlo);
      if (inserted) {
        worklist.push_back(operand_tiled_hlo);
      }
    }
  }
  std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>
      tiled_hlo_instructions = tiled_hlo_instructions_set.ExtractData();
  SortTiledHloInstructionsInPostOrder(tiled_hlo_instructions, root_tiled_hlo);
  std::variant<ConstraintExpression, FusionDecision> constraints_or =
      SetSymbolicTilesAndComputeConstraints(tiled_hlo_instructions, fusion);
  if (std::holds_alternative<FusionDecision>(constraints_or)) {
    return std::get<FusionDecision>(constraints_or);
  }
  std::unique_ptr<EmitterSpecificConstraints> emitter_specific_constraints;
  if (emitter_specific_constraints_builder != nullptr) {
    emitter_specific_constraints =
        emitter_specific_constraints_builder(tiled_hlo_instructions, fusion);
  }
  return SymbolicTileAnalysis(
      std::move(tiled_hlo_instructions),
      std::get<ConstraintExpression>(std::move(constraints_or)),
      std::move(emitter_specific_constraints), ctx);
}
absl::StatusOr<bool> SymbolicTileAnalysis::ParametersSatisfyConstraints(
    absl::Span<const int64_t> tile_parameters) const {
  if (!constraints_.is_satisfiable()) {
    return absl::FailedPreconditionError(
        "SymbolicTileAnalysis's constraints are not satisfiable. "
        "This should never happen.");
  }
  if (tile_parameters.size() != num_tile_parameters()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Failed to check if tile parameters satisfy constraints. Number of "
        "provided parameters doesn't match number of expected parameters "
        "(%d != %d)",
        tile_parameters.size(), num_tile_parameters()));
  }
  if (emitter_specific_constraints_ != nullptr) {
    TF_ASSIGN_OR_RETURN(
        bool constraints_are_satisfied,
        emitter_specific_constraints_->ParametersSatisfyConstraints(
            tile_parameters));
    if (!constraints_are_satisfied) {
      return false;
    }
  }
  return constraints_.IsSatisfiedBy(tile_parameters);
}
absl::StatusOr<TiledHloComputation>
SymbolicTileAnalysis::ComputeTiledHloInstructions(
    absl::Span<const int64_t> tile_parameters,
    bool constraints_are_known_satisfied,
    bool compute_all_tile_offset_indexing_maps) const {
  if (!constraints_are_known_satisfied) {
    TF_ASSIGN_OR_RETURN(bool constraints_are_satisfied,
                        ParametersSatisfyConstraints(tile_parameters));
    if (!constraints_are_satisfied) {
      return absl::InvalidArgumentError(
          absl::StrCat("Tile parameters ", absl::StrJoin(tile_parameters, ", "),
                       " do not satisfy constraints."));
    }
  }
  llvm::SmallPtrSet<const HloInstruction*, 8> parameters_with_offset_indexing;
  absl::flat_hash_map<const SymbolicTiledHloInstruction*,
                      llvm::SmallVector<int64_t>>
      tile_sizes_map;
  if (!compute_all_tile_offset_indexing_maps) {
    absl::flat_hash_set<size_t> hashes;
    for (const std::unique_ptr<SymbolicTiledHloInstruction>&
             symbolic_tiled_hlo : symbolic_tiled_hlo_instructions_) {
      if (!symbolic_tiled_hlo->operands().empty()) {
        continue;
      }
      llvm::SmallVector<int64_t> tile_sizes =
          symbolic_tiled_hlo->TileSizes(tile_parameters);
      size_t hash_value = absl::HashOf(symbolic_tiled_hlo->hlo(),
                                       absl::Span<const int64_t>(tile_sizes));
      tile_sizes_map.emplace(symbolic_tiled_hlo.get(), std::move(tile_sizes));
      auto [it, inserted] = hashes.insert(hash_value);
      if (!inserted) {
        parameters_with_offset_indexing.insert(symbolic_tiled_hlo->hlo());
      }
    }
  }
  OutputTilingInfo output_tiling_info = ComputeOutputTilingInfo(
      GetRoot()->hlo()->shape().dimensions(), tile_parameters, context_);
  OrderedUniquePtrValueHashSet<TiledHloInstruction> tiled_hlo_instructions_set;
  absl::flat_hash_map<const SymbolicTiledHloInstruction*, TiledHloInstruction*>
      symbolic_to_tiled_hlo_map;
  tiled_hlo_instructions_set.Reserve(symbolic_tiled_hlo_instructions_.size());
  std::function<absl::StatusOr<TiledHloInstruction*>(
      const SymbolicTiledHloInstruction*)>
      get_tiled_hlo_instruction;
  for (const std::unique_ptr<SymbolicTiledHloInstruction>& symbolic_tiled_hlo :
       symbolic_tiled_hlo_instructions_) {
    llvm::SmallVector<int64_t> tile_sizes;
    auto it = tile_sizes_map.find(symbolic_tiled_hlo.get());
    if (it != tile_sizes_map.end()) {
      tile_sizes = it->second;
    } else {
      tile_sizes = symbolic_tiled_hlo->TileSizes(tile_parameters);
    }
    llvm::SmallVector<int64_t> tile_strides =
        symbolic_tiled_hlo->TileStrides(tile_parameters);
    std::optional<IndexingMap> tile_offset_indexing;
    if (compute_all_tile_offset_indexing_maps ||
        parameters_with_offset_indexing.contains(symbolic_tiled_hlo->hlo()) ||
        symbolic_tiled_hlo->hlo()->opcode() == HloOpcode::kIota) {
      TF_ASSIGN_OR_RETURN(
          tile_offset_indexing,
          ComputeTileOffsetIndexing(
              *symbolic_tiled_hlo,
              output_tiling_info.output_tile_offset_indexing, context_));
    }
    llvm::SmallVector<const TiledHloInstruction*> operands;
    for (const SymbolicTiledHloInstruction* operand :
         symbolic_tiled_hlo->operands()) {
      operands.push_back(symbolic_to_tiled_hlo_map.at(operand));
    }
    TF_ASSIGN_OR_RETURN(std::unique_ptr<TiledHloInstruction> tiled_hlo_holder,
                        TiledHloInstruction::Create(
                            symbolic_tiled_hlo->hlo(), std::move(operands),
                            std::move(tile_sizes), std::move(tile_strides),
                            std::move(tile_offset_indexing)));
    auto [tiled_hlo, inserted] =
        tiled_hlo_instructions_set.Insert(std::move(tiled_hlo_holder));
    symbolic_to_tiled_hlo_map[symbolic_tiled_hlo.get()] = tiled_hlo;
  }
  return TiledHloComputation::FromSortedTiledHloInstructions(
      tiled_hlo_instructions_set.ExtractData(),
      output_tiling_info.num_output_tiles_per_dim);
}
std::string SymbolicTileAnalysis::ToString() const {
  std::stringstream ss;
  NameUniquer name_uniquer("_");
  absl::flat_hash_map<SymbolicTiledHloInstruction*, std::string> tile_names;
  for (const auto& tiled_hlo : symbolic_tiled_hlo_instructions_) {
    std::string tile_name = name_uniquer.GetUniqueName(
        absl::StrCat(tiled_hlo->hlo()->name(), ".tile_0"));
    tile_names[tiled_hlo.get()] = tile_name;
    absl::InlinedVector<std::string, 4> operand_names;
    for (const auto& operand : tiled_hlo->operands()) {
      operand_names.push_back(tile_names.at(operand));
    }
    ss << tile_name << " = " << HloOpcodeString(tiled_hlo->hlo()->opcode())
       << "(" << absl::StrJoin(operand_names, ", ") << ")\n";
    ss << tiled_hlo->ToString();
  }
  return ss.str();
}
namespace {
std::vector<int64_t> PossibleTileSizesForOneDimension(int64_t dim_size) {
  CHECK_GE(dim_size, 1);
  std::vector<int64_t> result;
  result.reserve(absl::bit_width(static_cast<uint64_t>(dim_size)));
  for (int64_t tile_size = 1; tile_size < dim_size; tile_size *= 2) {
    result.push_back(tile_size);
  }
  result.push_back(dim_size);
  return result;
}
}  
namespace detail {
std::vector<SymbolicTileAnalysis::Tiling> GetGoodTilings(
    absl::Span<const int64_t> dim_sizes,
    std::function<bool(absl::Span<const int64_t>)> is_valid) {
  CHECK(is_valid != nullptr);
  std::vector<SymbolicTileAnalysis::Tiling> tilings;
  tilings.push_back({});
  for (int dim_size : dim_sizes) {
    std::vector<int64_t> possible_tile_sizes =
        PossibleTileSizesForOneDimension(dim_size);
    std::vector<SymbolicTileAnalysis::Tiling> extended_tilings;
    extended_tilings.reserve(tilings.size() * possible_tile_sizes.size());
    for (const SymbolicTileAnalysis::Tiling& tiling : tilings) {
      for (int64_t tile_size : possible_tile_sizes) {
        SymbolicTileAnalysis::Tiling extended_tiling = tiling;
        extended_tiling.push_back(tile_size);
        extended_tilings.push_back(extended_tiling);
      }
    }
    tilings = std::move(extended_tilings);
  }
  tilings.erase(
      std::remove_if(tilings.begin(), tilings.end(), std::not_fn(is_valid)),
      tilings.end());
  return tilings;
}
}  
absl::StatusOr<std::vector<SymbolicTileAnalysis::Tiling>>
SymbolicTileAnalysis::GetGoodTilings() const {
  TF_RET_CHECK(!symbolic_tiled_hlo_instructions_.empty());
  TF_RET_CHECK(symbolic_tiled_hlo_instructions_.back() != nullptr);
  const SymbolicTiledHloInstruction& instr =
      *symbolic_tiled_hlo_instructions_.back();
  TF_RET_CHECK(instr.hlo() != nullptr);
  const Shape& shape = instr.hlo()->shape();
  if (!absl::c_all_of(shape.dimensions(),
                      [](int64_t dim_size) { return dim_size >= 1; })) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Shape %s has zero or negative dimensions.", shape.ToString()));
  }
  absl::Status status = absl::OkStatus();
  std::vector<SymbolicTileAnalysis::Tiling> result = detail::GetGoodTilings(
      shape.dimensions(), [&](absl::Span<const int64_t> tile_sizes) {
        absl::StatusOr<bool> is_valid =
            ParametersSatisfyConstraints(tile_sizes);
        if (!is_valid.ok()) {
          status = is_valid.status();
          return false;
        }
        return is_valid.value();
      });
  if (status.ok()) {
    return result;
  }
  return status;
}
}  
}  