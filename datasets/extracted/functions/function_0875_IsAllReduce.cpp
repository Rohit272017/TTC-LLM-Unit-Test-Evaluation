#include "xla/service/all_reduce_promotion.h"
#include <memory>
#include <string>
#include <utility>
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
namespace xla {
namespace {
bool IsAllReduce(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kAllReduce ||
         inst->opcode() == HloOpcode::kReduceScatter;
}
std::unique_ptr<HloInstruction> CloneAllReduce(
    const HloInstruction* inst, const Shape& shape,
    absl::Span<HloInstruction* const> operands) {
  std::unique_ptr<HloInstruction> new_inst =
      inst->CloneWithNewOperands(shape, operands);
  HloComputation* to_apply = new_inst->to_apply();
  HloComputation* to_apply_promoted = [&]() {
    PrimitiveType type = shape.element_type();
    std::string name = absl::StrCat(to_apply->name(), "_promoted");
    HloComputation::Builder promoted(name);
    auto x = promoted.AddInstruction(HloInstruction::CreateParameter(
        0, ShapeUtil::MakeShape(type, {}), "x"));
    auto y = promoted.AddInstruction(HloInstruction::CreateParameter(
        1, ShapeUtil::MakeShape(type, {}), "y"));
    promoted.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(type, {}), to_apply->root_instruction()->opcode(),
        x, y));
    return inst->GetModule()->AddEmbeddedComputation(promoted.Build());
  }();
  new_inst->set_to_apply(to_apply_promoted);
  to_apply_promoted->SetCollectiveCallInstruction(new_inst.get());
  return new_inst;
}
}  
AllReducePromotion::AllReducePromotion(
    absl::Span<std::pair<PrimitiveType, PrimitiveType> const> from_to_types)
    : pass_(from_to_types, IsAllReduce, CloneAllReduce) {}
absl::StatusOr<bool> AllReducePromotion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return pass_.Run(module, execution_threads);
}
}  