#include "xla/service/all_reduce_contiguous.h"
#include <vector>
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
namespace xla {
namespace {
absl::Status ReplaceWithContiguousAllReduce(
    HloAllReduceInstruction* all_reduce) {
  TF_RET_CHECK(all_reduce);
  TF_RET_CHECK(!all_reduce->has_sharding());
  HloComputation& computation = *all_reduce->parent();  
  PrimitiveType element_type = all_reduce->operand(0)->shape().element_type();
  std::vector<HloInstruction*> flat_operands;
  flat_operands.reserve(all_reduce->operand_count());
  int64_t total_size = 0;
  for (HloInstruction* operand : all_reduce->operands()) {
    TF_RET_CHECK(operand->shape().IsArray());
    int64_t num_elements = ShapeUtil::ElementsIn(operand->shape());
    Shape flat_shape = ShapeUtil::MakeShape(element_type, {num_elements});
    flat_operands.push_back(computation.AddInstruction(
        HloInstruction::CreateBitcast(flat_shape, operand)));
    total_size += num_elements;
  }
  Shape concat_shape = ShapeUtil::MakeShape(element_type, {total_size});
  HloInstruction* concatenated =
      computation.AddInstruction(HloInstruction::CreateConcatenate(
          concat_shape, flat_operands, 0));
  HloInstruction* new_all_reduce =
      computation.AddInstruction(HloInstruction::CreateAllReduce(
          concat_shape, {concatenated}, all_reduce->to_apply(),
          all_reduce->device_list(),
          false, all_reduce->channel_id(),
          all_reduce->use_global_device_ids()));
  std::vector<HloInstruction*> outputs;
  outputs.reserve(all_reduce->operand_count());
  int64_t offset = 0;
  for (int64_t i = 0; i < all_reduce->operand_count(); ++i) {
    const Shape& flat_shape = flat_operands[i]->shape();
    int64_t end = offset + flat_shape.dimensions(0);
    HloInstruction* sliced = computation.AddInstruction(
        HloInstruction::CreateSlice(flat_shape, new_all_reduce,
                                    {offset},
                                    {end},
                                    {1}));
    outputs.push_back(computation.AddInstruction(HloInstruction::CreateBitcast(
        all_reduce->operand(i)->shape(), sliced)));
    offset = end;
  }
  TF_RETURN_IF_ERROR(computation.ReplaceWithNewInstruction(
      all_reduce, HloInstruction::CreateTuple(outputs)));
  return absl::OkStatus();
}
}  
absl::StatusOr<bool> AllReduceContiguous::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(1) << "Running AllReduceContiguous";
  if (hlo_query::ContainsLayoutConstrainedAllReduce(*module)) {
    VLOG(1)
        << "Skip AllReduceContiguous because the module contains all-reduce "
           "with constrained layouts";
    return false;
  }
  std::vector<HloAllReduceInstruction*> all_reduces;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kAllReduce &&
          instruction->operand_count() > 1) {
        all_reduces.push_back(Cast<HloAllReduceInstruction>(instruction));
      }
    }
  }
  for (HloAllReduceInstruction* all_reduce : all_reduces) {
    TF_RETURN_IF_ERROR(ReplaceWithContiguousAllReduce(all_reduce));
  }
  return !all_reduces.empty();
}
}  