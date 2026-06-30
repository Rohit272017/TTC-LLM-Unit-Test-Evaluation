#include "xla/service/all_gather_broadcast_reorder.h"
#include <cstdint>
#include <vector>
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
namespace xla {
absl::StatusOr<bool> AllGatherBroadcastReorder::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  if (hlo_query::ContainsLayoutConstrainedCollective(*module,
                                                     HloOpcode::kAllGather)) {
    VLOG(1) << "Skip AllGatherBroadcastReorder because the module contains "
               "all-gather with constrained layouts";
    return false;
  }
  int64_t next_channel_id = hlo_query::NextChannelId(*module);
  bool changed = false;
  for (auto computation : module->computations(execution_threads)) {
    for (HloInstruction *inst : computation->MakeInstructionPostOrder()) {
      if (inst->opcode() != HloOpcode::kAllGather || !inst->shape().IsArray() ||
          inst->operand(0)->opcode() != HloOpcode::kBroadcast) {
        continue;
      }
      HloAllGatherInstruction *ag = Cast<HloAllGatherInstruction>(inst);
      HloBroadcastInstruction *bcast =
          Cast<HloBroadcastInstruction>(inst->mutable_operand(0));
      absl::flat_hash_set<int64_t> non_uniform_dims;
      non_uniform_dims.insert(bcast->dimensions().begin(),
                              bcast->dimensions().end());
      const bool all_gather_along_uniform_dim =
          non_uniform_dims.insert(ag->all_gather_dimension()).second;
      int64_t uniform_dim_size = 1;
      for (int64_t i = 0; i < ag->shape().rank(); ++i) {
        if (non_uniform_dims.count(i) == 0) {
          uniform_dim_size *= ag->shape().dimensions(i);
        }
      }
      if (uniform_dim_size == 1) {
        continue;
      }
      HloInstruction *replacement;
      const int64_t ag_dim = ag->all_gather_dimension();
      if (!all_gather_along_uniform_dim) {
        VLOG(2) << "All-gather along non uniform dimension";
        auto ag_dim_index = PositionInContainer(bcast->dimensions(), ag_dim);
        Shape new_ag_shape = bcast->operand(0)->shape();
        new_ag_shape.set_dimensions(ag_dim_index,
                                    ag->shape().dimensions(ag_dim));
        auto *new_ag =
            Cast<HloAllGatherInstruction>(computation->AddInstruction(
                ag->CloneWithNewOperands(new_ag_shape, bcast->operands())));
        if (ag->channel_id()) {
          new_ag->set_channel_id(next_channel_id++);
        }
        new_ag->set_all_gather_dimension(ag_dim_index);
        replacement = computation->AddInstruction(
            bcast->CloneWithNewOperands(ag->shape(), {new_ag}));
      } else {
        VLOG(2) << "All-gather along uniform dimension";
        HloInstruction *x = bcast->mutable_operand(0);
        std::vector<int64_t> shape_dims{1};
        absl::Span<const int64_t> x_dims = x->shape().dimensions();
        shape_dims.insert(shape_dims.end(), x_dims.begin(), x_dims.end());
        Shape shape =
            ShapeUtil::MakeShape(x->shape().element_type(), shape_dims);
        HloInstruction *rs0 = computation->AddInstruction(
            HloInstruction::CreateReshape(shape, x));
        const int64_t ag_factor = ag->shape().dimensions(ag_dim) /
                                  ag->operand(0)->shape().dimensions(ag_dim);
        shape.set_dimensions(0, ag_factor);
        auto *new_ag =
            Cast<HloAllGatherInstruction>(computation->AddInstruction(
                ag->CloneWithNewOperands(shape, {rs0})));
        if (ag->channel_id()) {
          new_ag->set_channel_id(next_channel_id++);
        }
        new_ag->set_all_gather_dimension(0);
        std::vector<int64_t> bcast_shape_dims =
            SpanToVector(ag->shape().dimensions());
        bcast_shape_dims[ag_dim] = ag_factor;
        bcast_shape_dims.insert(bcast_shape_dims.begin() + ag_dim + 1,
                                ag->shape().dimensions(ag_dim) / ag_factor);
        Shape bcast_shape =
            ShapeUtil::MakeShape(x->shape().element_type(), bcast_shape_dims);
        std::vector<int64_t> bcast_dims;
        bcast_dims.push_back(ag_dim);
        for (int64_t d : bcast->dimensions()) {
          bcast_dims.push_back(d + (d > ag_dim));
        }
        HloInstruction *bcast = computation->AddInstruction(
            HloInstruction::CreateBroadcast(bcast_shape, new_ag, bcast_dims));
        replacement = computation->AddInstruction(
            HloInstruction::CreateReshape(ag->shape(), bcast));
      }
      TF_RETURN_IF_ERROR(ag->ReplaceAllUsesWith(replacement));
      TF_RETURN_IF_ERROR(computation->RemoveInstructionAndUnusedOperands(ag));
      changed = true;
    }
  }
  return changed;
}
}  