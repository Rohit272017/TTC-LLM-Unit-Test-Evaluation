#include "xla/service/gpu/transforms/scatter_expander.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
namespace xla {
bool GpuScatterExpander::InstructionMatchesPattern(HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kScatter &&
         (inst->shape().IsTuple() ||
          primitive_util::BitWidth(inst->shape().element_type()) > 64);
}
}  