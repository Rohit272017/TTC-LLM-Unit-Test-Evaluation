#include "xla/service/gpu/transforms/gemv_rewriter.h"
#include <cstdint>
#include <vector>
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/shape.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace gpu {
namespace {
absl::StatusOr<Layout> GetLayoutWithNewMinorMostDimension(
    const Layout& layout) {
  if (!LayoutUtil::IsMonotonicWithDim0Major(layout)) {
    return absl::InvalidArgumentError("Layout is not normalized.");
  }
  return LayoutUtil::MakeDescendingLayout(layout.minor_to_major_size() + 1);
}
class GemvRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleDot(HloInstruction* instr) override {
    HloDotInstruction* dot = Cast<HloDotInstruction>(instr);
    const DotDimensionNumbers& dim_numbers = dot->dot_dimension_numbers();
    HloInstruction* lhs = dot->mutable_operand(0);
    HloInstruction* rhs = dot->mutable_operand(1);
    bool lhs_has_non_contracting_dim =
        lhs->shape().rank() ==
        dim_numbers.lhs_batch_dimensions_size() +
            dim_numbers.lhs_contracting_dimensions_size() + 1;
    bool rhs_has_non_contracting_dim =
        rhs->shape().rank() ==
        dim_numbers.rhs_batch_dimensions_size() +
            dim_numbers.rhs_contracting_dimensions_size() + 1;
    if (lhs_has_non_contracting_dim && rhs_has_non_contracting_dim) {
      return absl::OkStatus();
    }
    if (!lhs_has_non_contracting_dim && !rhs_has_non_contracting_dim) {
      return absl::OkStatus();
    }
    if (dot->shape().is_dynamic()) {
      return absl::OkStatus();
    }
    changed_ = true;
    HloComputation* computation = dot->parent();
    HloInstruction* new_lhs = lhs;
    if (!lhs_has_non_contracting_dim) {
      const Shape& lhs_shape = lhs->shape();
      absl::Span<const int64_t> lhs_dimensions = lhs_shape.dimensions();
      std::vector<int64_t> new_lhs_dimensions(lhs_dimensions.begin(),
                                              lhs_dimensions.end());
      new_lhs_dimensions.push_back(1);
      Shape new_lhs_shape(
          lhs_shape.element_type(), new_lhs_dimensions,
          absl::InlinedVector<bool, 4>(new_lhs_dimensions.size(), false),
          {});
      TF_ASSIGN_OR_RETURN(
          *new_lhs_shape.mutable_layout(),
          GetLayoutWithNewMinorMostDimension(lhs_shape.layout()));
      new_lhs = computation->AddInstruction(
          HloInstruction::CreateBitcast(new_lhs_shape, lhs));
    }
    HloInstruction* new_rhs = rhs;
    if (!rhs_has_non_contracting_dim) {
      const Shape& rhs_shape = rhs->shape();
      absl::Span<const int64_t> rhs_dimensions = rhs_shape.dimensions();
      std::vector<int64_t> new_rhs_dimensions(rhs_dimensions.begin(),
                                              rhs_dimensions.end());
      new_rhs_dimensions.push_back(1);
      Shape new_rhs_shape(
          rhs_shape.element_type(), new_rhs_dimensions,
          absl::InlinedVector<bool, 4>(new_rhs_dimensions.size(), false),
          {});
      TF_ASSIGN_OR_RETURN(
          *new_rhs_shape.mutable_layout(),
          GetLayoutWithNewMinorMostDimension(rhs_shape.layout()));
      new_rhs = computation->AddInstruction(
          HloInstruction::CreateBitcast(new_rhs_shape, rhs));
    }
    std::vector<int64_t> new_out_dimensions;
    new_out_dimensions.reserve(dot->shape().dimensions().size() + 1);
    for (int64_t dim_size : dot->shape().dimensions()) {
      new_out_dimensions.push_back(dim_size);
    }
    if (!lhs_has_non_contracting_dim) {
      int non_contracting_dim_size = new_out_dimensions.back();
      new_out_dimensions[new_out_dimensions.size() - 1] = 1;
      new_out_dimensions.push_back(non_contracting_dim_size);
    } else {
      new_out_dimensions.push_back(1);
    }
    Shape new_out_shape(
        dot->shape().element_type(), new_out_dimensions,
        absl::InlinedVector<bool, 4>(new_out_dimensions.size(), false),
        {});
    TF_ASSIGN_OR_RETURN(
        *new_out_shape.mutable_layout(),
        GetLayoutWithNewMinorMostDimension(dot->shape().layout()));
    HloInstruction* new_dot =
        computation->AddInstruction(HloInstruction::CreateDot(
            new_out_shape, new_lhs, new_rhs, dot->dot_dimension_numbers(),
            dot->precision_config()));
    HloInstruction* bitcast = computation->AddInstruction(
        HloInstruction::CreateBitcast(dot->shape(), new_dot));
    return computation->ReplaceInstruction(dot, bitcast);
  }
  bool changed() const { return changed_; }
 private:
  bool changed_ = false;
};
}  
absl::StatusOr<bool> GemvRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  GemvRewriterVisitor gemv_rewriter;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_RETURN_IF_ERROR(computation->Accept(&gemv_rewriter));
  }
  return gemv_rewriter.changed();
}
}  
}  