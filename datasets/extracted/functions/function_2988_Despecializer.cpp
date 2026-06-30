#include "xla/service/despecializer.h"
#include <iterator>
#include <utility>
#include <vector>
#include "xla/service/defuser.h"
#include "xla/service/float_normalization.h"
#include "xla/service/hlo_memory_scheduler.h"
#include "xla/service/sub_byte_normalization.h"
namespace xla {
Despecializer::Despecializer() : pipeline_("despecializer") {
  pipeline_.AddPass<HloDescheduler>();
  pipeline_.AddPass<ControlDepRemover>();
  pipeline_.AddPass<Defuser>();
  pipeline_.AddPass<BFloat16MixedPrecisionRemoval>();
  pipeline_.AddPass<SubByteNormalization>(
      SubByteNormalization::REMOVE_ELEMENT_SIZE);
}
void Despecializer::AddAssumeGatherIndicesInBoundRewriteToCopy() {
  pipeline_.AddPass<AssumeGatherIndicesInBoundRewriteToCopy>();
}
void Despecializer::AddReduceWindowToReduceBroadcastDeconstruct() {
  pipeline_.AddPass<DeconstructReduceWindowToReduceBroadcast>();
}
absl::StatusOr<bool> Despecializer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return pipeline_.Run(module, execution_threads);
}
absl::StatusOr<bool> AssumeGatherIndicesInBoundRewriteToCopy::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloInstruction*> candidates;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->IsCustomCall("AssumeGatherIndicesInBound")) {
        candidates.push_back(instruction);
      }
    }
  }
  for (HloInstruction* gather_indices : candidates) {
    auto computation = gather_indices->parent();
    auto copy = computation->AddInstruction(
        HloInstruction::CreateUnary(gather_indices->shape(), HloOpcode::kCopy,
                                    gather_indices->mutable_operand(0)));
    TF_CHECK_OK(computation->ReplaceInstruction(gather_indices, copy));
  }
  return !candidates.empty();
}
absl::StatusOr<bool> DeconstructReduceWindowToReduceBroadcast::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  std::vector<std::pair<HloInstruction*, int64_t>> candidate_rw;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kReduceWindow) {
        continue;
      }
      auto* reduce_window = CastOrNull<HloReduceWindowInstruction>(instruction);
      if (reduce_window == nullptr) {
        continue;
      }
      if (reduce_window->operand(0)->shape() != reduce_window->shape()) {
        continue;
      }
      const Window& window = reduce_window->window();
      int64_t num_stride_dilations = absl::c_count_if(
          window.dimensions(), [](const WindowDimension& win_dim) {
            return (
                win_dim.stride() != 1 || win_dim.window_reversal() == true ||
                win_dim.window_dilation() != 1 || win_dim.base_dilation() != 1);
          });
      if (num_stride_dilations != 0) {
        continue;
      }
      int64_t num_dimensions_reduced = absl::c_count_if(
          window.dimensions(),
          [](const WindowDimension& win_dim) { return (win_dim.size() != 1); });
      if (num_dimensions_reduced != 1) {
        continue;
      }
      auto reduce_dim = absl::c_find_if(
          window.dimensions(),
          [](const WindowDimension& win_dim) { return (win_dim.size() != 1); });
      if (reduce_dim == window.dimensions().end()) {
        continue;
      }
      int64_t reduce_dim_index =
          std::distance(window.dimensions().begin(), reduce_dim);
      auto input_dim_size =
          reduce_window->operand(0)->shape().dimensions(reduce_dim_index);
      if (reduce_dim->size() != 2 * input_dim_size - 1) {
        continue;
      }
      if (reduce_dim->padding_low() != input_dim_size - 1) {
        continue;
      }
      if (reduce_dim->padding_high() != input_dim_size - 1) {
        continue;
      }
      VLOG(2) << "Adding Candidate ReduceWindow:" << reduce_window->ToString();
      candidate_rw.push_back(std::make_pair(reduce_window, reduce_dim_index));
    }
  }
  for (const auto& rw : candidate_rw) {
    auto reduce_window = rw.first;
    auto reduce_dim_index = rw.second;
    if (reduce_window == nullptr || reduce_dim_index < 0 ||
        reduce_dim_index >= reduce_window->operand(0)->shape().rank()) {
      continue;
    }
    std::vector<int64_t> reduce_instr_dimensions;
    std::vector<int64_t> broadcast_dimensions;
    const Window& window = reduce_window->window();
    for (int64_t index = 0; index < window.dimensions().size(); ++index) {
      const auto& window_dimension = window.dimensions(index);
      if (window_dimension.size() == 1) {
        reduce_instr_dimensions.push_back(
            reduce_window->operand(0)->shape().dimensions(index));
        broadcast_dimensions.push_back(index);
      }
    }
    Shape reduce_shape = ShapeUtil::MakeShape(
        reduce_window->shape().element_type(), reduce_instr_dimensions);
    auto reduce_instr =
        reduce_window->AddInstruction(HloInstruction::CreateReduce(
            reduce_shape, reduce_window->mutable_operand(0),
            reduce_window->mutable_operand(1), {reduce_dim_index},
            reduce_window->called_computations()[0]));
    auto broadcast_instr =
        reduce_window->AddInstruction(HloInstruction::CreateBroadcast(
            reduce_window->shape(), reduce_instr, broadcast_dimensions));
    VLOG(2) << "reduce_window:" << reduce_window->ToString();
    VLOG(2) << "reduce:" << reduce_instr->ToString();
    VLOG(2) << "broadcast:" << broadcast_instr->ToString();
    TF_CHECK_OK(reduce_window->parent()->ReplaceInstruction(reduce_window,
                                                            broadcast_instr));
    changed = true;
  }
  return changed;
}
}  