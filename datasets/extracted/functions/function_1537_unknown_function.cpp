#include "xla/service/gpu/model/symbolic_tiled_hlo_instruction.h"
#include <cstdint>
#include <sstream>
#include <string>
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/service/gpu/model/affine_map_evaluator.h"
#include "xla/service/gpu/model/symbolic_tile.h"
namespace xla {
namespace gpu {
llvm::SmallVector<int64_t> SymbolicTiledHloInstruction::TileOffsets(
    absl::Span<int64_t const> tile_parameters) const {
  return EvaluateAffineMap(symbolic_tile().offset_map(),
                           tile_parameters);
}
llvm::SmallVector<int64_t> SymbolicTiledHloInstruction::TileSizes(
    absl::Span<int64_t const> tile_parameters) const {
  return EvaluateAffineMap(symbolic_tile().size_map(),
                           tile_parameters);
}
llvm::SmallVector<int64_t> SymbolicTiledHloInstruction::TileStrides(
    absl::Span<int64_t const> tile_parameters) const {
  return EvaluateAffineMap(symbolic_tile().stride_map(),
                           tile_parameters);
}
std::string SymbolicTiledHloInstruction::ToString() const {
  std::stringstream ss;
  ss << "\thlo: " << hlo_->ToString() << "\n";
  ss << "\t" << symbolic_tile().ToString() << "\n";
  ss << "\tindexing map: " << indexing_map_ << "\n";
  return ss.str();
}
}  
}  