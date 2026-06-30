#include "xla/service/gpu/transforms/scatter_slice_simplifier.h"
#include <cstdint>
#include <iterator>
#include <optional>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace {
bool IsValidIntermediaryUser(const HloInstruction* instruction) {
  return instruction->IsElementwise() ||
         instruction->opcode() == HloOpcode::kGetTupleElement;
}
class ScatterSliceMatcher {
 public:
  explicit ScatterSliceMatcher(const HloScatterInstruction* scatter)
      : scatter_(scatter),
        operand_dimensions_(
            scatter->scatter_operands()[0]->shape().dimensions()),
        result_dimensions_(operand_dimensions_.begin(),
                           operand_dimensions_.end()) {}
  std::optional<Shape> InferShape() {
    VLOG(10) << "Evaluating scatter " << scatter_->name();
    if (!AreAllUsersValid(scatter_)) {
      return std::nullopt;
    }
    std::vector<Shape> result_shapes;
    absl::c_transform(scatter_->scatter_operands(),
                      std::back_inserter(result_shapes),
                      [&](const HloInstruction* op) {
                        return ShapeUtil::MakeShape(op->shape().element_type(),
                                                    result_dimensions_);
                      });
    return ShapeUtil::MakeMaybeTupleShape(result_shapes);
  }
 private:
  bool UpdateDimensions(const HloSliceInstruction* slice) {
    int64_t rank = slice->shape().rank();
    for (int64_t i = 0; i < rank; ++i) {
      if (slice->slice_starts(i) != 0 || slice->slice_strides(i) != 1) {
        return false;  
      }
      if (slice->slice_limits(i) != result_dimensions_[i]) {
        if (result_dimensions_[i] != operand_dimensions_[i]) {
          return false;  
        }
        auto& update_window_dims =
            scatter_->scatter_dimension_numbers().update_window_dims();
        if (absl::c_binary_search(update_window_dims, i)) {
          return false;  
        }
        result_dimensions_[i] = slice->slice_limits(i);
        VLOG(10) << "Dimension " << i << " truncated to size "
                 << result_dimensions_[i];
      }
    }
    return true;
  }
  bool IsUserValid(const HloInstruction* op) {
    VLOG(10) << "Visiting user " << op->name();
    if (auto* slice = DynCast<HloSliceInstruction>(op)) {
      return UpdateDimensions(slice);
    }
    bool is_valid = visited_set_.contains(op) ||
                    (IsValidIntermediaryUser(op) && AreAllUsersValid(op));
    if (is_valid) {
      visited_set_.emplace(op);
    }
    return is_valid;
  }
  bool AreAllUsersValid(const HloInstruction* op) {
    if (op->user_count() == 0) {
      return !op->IsRoot();
    }
    return absl::c_all_of(op->users(), [this](const HloInstruction* user) {
      return IsUserValid(user);
    });
  }
  const HloScatterInstruction* scatter_;
  absl::flat_hash_set<const HloInstruction*> visited_set_;
  absl::Span<const int64_t> operand_dimensions_;
  DimensionVector result_dimensions_;
};
HloInstruction* CreateSliceFrom(HloInstruction* operand, const Shape& shape) {
  std::vector<int64_t> start_indices(shape.rank(), 0);
  std::vector<int64_t> limit_indices(shape.rank());
  std::vector<int64_t> strides(shape.rank(), 1);
  for (int64_t i = 0; i < shape.rank(); ++i) {
    limit_indices[i] = shape.dimensions(i);
  }
  return operand->AddInstruction(HloInstruction::CreateSlice(
      shape, operand, start_indices, limit_indices, strides));
}
HloInstruction* CreateScatterFrom(HloScatterInstruction* scatter,
                                  const Shape& shape) {
  std::vector<HloInstruction*> operands(scatter->scatter_operand_count());
  for (int64_t i = 0; i < operands.size(); ++i) {
    operands[i] =
        CreateSliceFrom(scatter->scatter_operands()[i],
                        shape.IsTuple() ? shape.tuple_shapes(i) : shape);
  }
  return scatter->AddInstruction(HloInstruction::CreateScatter(
      shape, absl::MakeSpan(operands), scatter->scatter_indices(),
      scatter->scatter_updates(), scatter->called_computations()[0],
      scatter->scatter_dimension_numbers(), scatter->indices_are_sorted(),
      scatter->unique_indices()));
}
class ScatterSliceSimplifierVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleScatter(HloInstruction* instruction) override {
    auto* scatter = Cast<HloScatterInstruction>(instruction);
    std::optional<Shape> result_shape =
        ScatterSliceMatcher(scatter).InferShape();
    if (!result_shape.has_value()) {
      return absl::OkStatus();
    }
    VLOG(2) << "Matched scatter " << scatter->name() << " with shape "
            << scatter->shape().ToString() << ", inferred result shape "
            << result_shape->ToString() << " (from the slice users)";
    HloInstruction* new_scatter = CreateScatterFrom(scatter, *result_shape);
    return ReplaceAllUsersRecursive(scatter, new_scatter);
  }
 private:
  absl::Status ReplaceAllUsersRecursive(HloInstruction* old_instruction,
                                        HloInstruction* new_instruction) {
    replacements_[old_instruction] = new_instruction;
    std::vector<HloInstruction*> users = old_instruction->users();
    for (HloInstruction* user : users) {
      if (user->parent() == nullptr) {
        VLOG(3) << "Skipping user " << user->name() << " (already replaced)";
        continue;
      }
      TF_RETURN_IF_ERROR(ReplaceUserRecursive(user, new_instruction));
    }
    return absl::OkStatus();
  }
  absl::Status ReplaceUserRecursive(HloInstruction* user,
                                    HloInstruction* operand) {
    VLOG(3) << "Replacing scatter user " << user->name();
    if (user->opcode() == HloOpcode::kSlice) {
      return ReplaceInstruction(user, operand);
    }
    HloInstruction* new_user = nullptr;
    if (user->IsElementwise()) {
      auto new_shape = [operand](HloInstruction* from) {
        return ShapeUtil::MakeShape(from->shape().element_type(),
                                    operand->shape().dimensions());
      };
      std::vector<HloInstruction*> new_operands;
      absl::c_transform(user->operands(), std::back_inserter(new_operands),
                        [&](HloInstruction* op) {
                          auto it = replacements_.find(op);
                          return it != replacements_.end()
                                     ? it->second
                                     : CreateSliceFrom(op, new_shape(op));
                        });
      new_user = user->AddInstruction(
          user->CloneWithNewOperands(new_shape(user), new_operands));
    } else {
      auto* gte = Cast<HloGetTupleElementInstruction>(user);
      TF_ASSIGN_OR_RETURN(new_user,
                          MakeGetTupleElementHlo(operand, gte->tuple_index(),
                                                 &user->metadata()));
    }
    return ReplaceAllUsersRecursive(user, new_user);
  }
  absl::flat_hash_map<HloInstruction*, HloInstruction*> replacements_;
};
}  
absl::StatusOr<bool> ScatterSliceSimplifier::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return ScatterSliceSimplifierVisitor{}.RunOnModule(module, execution_threads);
}
}  