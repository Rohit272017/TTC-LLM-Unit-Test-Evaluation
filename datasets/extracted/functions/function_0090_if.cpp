#include "xla/service/reshape_decomposer.h"
#include "absl/status/status.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/service/hlo_creation_utils.h"
namespace xla {
namespace {
class ReshapeDecomposerVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleReshape(HloInstruction* reshape) override {
    HloInstruction* operand = reshape->mutable_operand(0);
    auto s = reshape->shape();
    auto s0 = operand->shape();
    if (ShapeUtil::ReshapeIsBitcast(s, s0)) {
      auto b = MakeBitcastHlo(operand, s, &operand->metadata());
      return ReplaceInstruction(reshape, b);
    } else if (auto output_aligned_input_shape =
                   ShapeUtil::AlignLayouts(s, s0)) {
      Shape new_input_shape = *output_aligned_input_shape;
      HloInstruction* copied_operand = MakeCopyHlo(operand, new_input_shape);
      VLOG(3) << "Decomposing reshape into reshape-bitcast and a physical "
                 "transpose on the operand: "
              << copied_operand->ToString();
      auto b = MakeBitcastHlo(copied_operand, s, &copied_operand->metadata());
      TF_RETURN_IF_ERROR(ReplaceInstruction(reshape, b));
      DCHECK(ShapeUtil::ReshapeIsBitcast(b->shape(), b->operand(0)->shape()));
    } else if (auto input_aligned_output_shape =
                   ShapeUtil::AlignLayouts(s0, s)) {
      Shape new_output_shape = *input_aligned_output_shape;
      auto b = MakeBitcastHlo(operand, new_output_shape, &operand->metadata());
      DCHECK(ShapeUtil::ReshapeIsBitcast(b->shape(), b->operand(0)->shape()));
      HloInstruction* copied_result = MakeCopyHlo(b, s);
      VLOG(3) << "Decomposing reshape into reshape-bitcast and a physical "
                 "transposition on the result: "
              << copied_result->ToString();
      TF_RETURN_IF_ERROR(ReplaceInstruction(reshape, copied_result));
    } else {
      VLOG(3) << "Both input and output of reshape are not alignable, create "
                 "two physical transposes";
      auto s0_normalized = ShapeUtil::MakeShapeWithDescendingLayout(
          s0.element_type(), s0.dimensions());
      auto c1 = MakeCopyHlo(reshape->mutable_operand(0), s0_normalized);
      auto s_normalized = ShapeUtil::MakeShapeWithDescendingLayout(
          s.element_type(), s.dimensions());
      auto b = MakeBitcastHlo(c1, s_normalized, &c1->metadata());
      DCHECK(ShapeUtil::ReshapeIsBitcast(b->shape(), b->operand(0)->shape()));
      auto c2 = MakeCopyHlo(b, s);
      TF_RETURN_IF_ERROR(ReplaceInstruction(reshape, c2));
    }
    return absl::OkStatus();
  }
};
}  
absl::StatusOr<bool> ReshapeDecomposer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return ReshapeDecomposerVisitor{}.RunOnModule(module, execution_threads);
}
}  