#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include <sstream>
#include <string>
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/tiled_hlo_instruction.h"
#include "xla/service/name_uniquer.h"
#include "xla/util.h"
namespace xla {
namespace gpu {
std::string TiledHloComputation::ToString() const {
  std::stringstream ss;
  NameUniquer name_uniquer("_");
  absl::flat_hash_map<const TiledHloInstruction*, std::string> tile_names;
  for (const auto* tiled_hlo : instructions()) {
    std::string tile_name = name_uniquer.GetUniqueName(
        absl::StrCat(tiled_hlo->hlo()->name(), ".tile_0"));
    tile_names[tiled_hlo] = tile_name;
    absl::InlinedVector<std::string, 4> operand_names;
    for (const auto& operand : tiled_hlo->operands()) {
      operand_names.push_back(tile_names.at(operand));
    }
    ss << tile_name << " = " << HloOpcodeString(tiled_hlo->hlo()->opcode())
       << "(" << absl::StrJoin(operand_names, ", ") << ")\n";
    ss << tiled_hlo->ToString() << "\n";
  }
  return ss.str();
}
}  
}  