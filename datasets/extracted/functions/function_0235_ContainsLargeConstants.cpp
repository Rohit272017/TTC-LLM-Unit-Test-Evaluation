#include "xla/service/hlo_computation_deduplicator.h"
#include <string>
#include <utility>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape_util.h"
#include "tsl/platform/logging.h"
namespace xla {
bool HloComputationDeduplicator::ContainsLargeConstants(HloComputation* comp) {
  int total_size = 0;
  for (HloInstruction* instruction : comp->instructions()) {
    if (instruction->IsConstant()) {
      total_size += ShapeUtil::ArrayDataSize(instruction->literal().shape());
      if (total_size > 1024) {
        return true;
      }
    }
  }
  return false;
}
absl::StatusOr<bool> HloComputationDeduplicator::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  absl::flat_hash_map<std::string, HloComputation*> unique_comps;
  absl::flat_hash_map<HloComputation*, HloComputation*> replacement;
  HloPrintOptions options = HloPrintOptions::Canonical();
  options.set_print_subcomputation_mode(
      HloPrintOptions::PrintSubcomputationMode::kOff);
  options.set_print_infeed_outfeed_config(false);
  options.set_print_only_essential_constants(true);
  options.set_print_operand_shape(true);
  options.set_print_ids(false);
  options.set_canonicalize_computations(true);
  auto comp_eq = [&replacement](const HloComputation* a,
                                const HloComputation* b) {
    if (a->unique_id() == b->unique_id()) return true;
    if (replacement.contains(a) &&
        replacement.at(a)->unique_id() == b->unique_id()) {
      return true;
    }
    if (replacement.contains(b) &&
        replacement.at(b)->unique_id() == a->unique_id()) {
      return true;
    }
    if (replacement.contains(a) && replacement.contains(b) &&
        replacement.at(a)->unique_id() == replacement.at(b)->unique_id()) {
      return true;
    }
    return false;
  };
  for (HloComputation* comp :
       module->MakeComputationPostOrder(execution_threads)) {
    if (comp->IsEntryComputation() || comp->instruction_count() > 128 ||
        ContainsLargeConstants(comp) || comp->IsCollectiveCalledComputation()) {
      continue;
    }
    std::string comp_str = comp->ToString(options);
    auto poss_dup = unique_comps.find(comp_str);
    if (poss_dup != unique_comps.end() &&
        poss_dup->second->Equal(*comp,  true,
                                comp_eq)) {
      VLOG(2) << "Replacing " << comp->name() << " with "
              << poss_dup->second->name();
      replacement[comp] = poss_dup->second;
    } else {
      unique_comps[std::move(comp_str)] = comp;
    }
  }
  if (mark_fusion_duplications_) {
    module->MarkFusionDuplications(replacement);
  } else {
    module->ReplaceComputations(replacement);
  }
  return !replacement.empty();
}
}  