#include "xla/service/add_original_value.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/shape_util.h"
namespace xla {
absl::StatusOr<bool> AddOriginalValue::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (const auto computation : module->computations()) {
    for (const auto instruction : computation->instructions()) {
      auto original_value =
          std::make_shared<OriginalValue>(instruction->shape());
      if (instruction->opcode() == HloOpcode::kGetTupleElement) {
        const auto* tuple = instruction->operand(0);
        original_value->CopySubtreeFrom(*tuple->original_value(),
                                        {instruction->tuple_index()}, {});
      } else if (instruction->opcode() == HloOpcode::kTuple) {
        for (int64_t operand_number = 0;
             operand_number < instruction->operand_count(); ++operand_number) {
          original_value->CopySubtreeFrom(
              *instruction->operand(operand_number)->original_value(), {},
              {operand_number});
        }
      } else {
        for (auto& leaf : original_value->leaves()) {
          leaf.second = {std::string(instruction->name()), leaf.first};
        }
      }
      instruction->set_original_value(original_value);
      changed = true;
    }
  }
  return changed;
}
}  