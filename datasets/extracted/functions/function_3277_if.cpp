#include "xla/service/gpu/transforms/move_copy_to_users.h"
#include <vector>
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/service/hlo_creation_utils.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace {
class MoveCopyToUsersVisitor : public DfsHloRewriteVisitor {
  absl::Status HandlePad(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    HloInstruction* c = hlo->mutable_operand(1);
    if (operand->opcode() == HloOpcode::kCopy) {
      HloInstruction* copied = operand->mutable_operand(0);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * earlier_pad,
          MakePadHlo(copied, c, hlo->padding_config(), &hlo->metadata()));
      *earlier_pad->mutable_shape()->mutable_layout() =
          copied->shape().layout();
      HloInstruction* later_copy = MakeCopyHlo(earlier_pad, hlo->shape());
      TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
    }
    return absl::OkStatus();
  }
  absl::Status HandleSlice(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    if (operand->opcode() == HloOpcode::kCopy) {
      HloInstruction* copied = operand->mutable_operand(0);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * earlier_slice,
          MakeSliceHlo(copied, hlo->slice_starts(), hlo->slice_limits(),
                       hlo->slice_strides(), &hlo->metadata()));
      *earlier_slice->mutable_shape()->mutable_layout() =
          copied->shape().layout();
      HloInstruction* later_copy = MakeCopyHlo(earlier_slice, hlo->shape());
      TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
    }
    return absl::OkStatus();
  }
  absl::Status HandleDynamicSlice(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    if (operand->opcode() == HloOpcode::kCopy) {
      HloInstruction* copied = operand->mutable_operand(0);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * earlier_slice,
          MakeDynamicSliceHlo(
              copied,
              absl::Span<HloInstruction* const>(hlo->operands()).subspan(1),
              hlo->dynamic_slice_sizes(), &hlo->metadata()));
      *earlier_slice->mutable_shape()->mutable_layout() =
          copied->shape().layout();
      HloInstruction* later_copy = MakeCopyHlo(earlier_slice, hlo->shape());
      TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
    }
    return absl::OkStatus();
  }
  absl::Status HandleReduceWindow(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    if (operand->opcode() == HloOpcode::kCopy) {
      HloInstruction* copied = operand->mutable_operand(0);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * earlier_reduce_window,
          MakeReduceWindowHlo(copied, hlo->mutable_operand(1), hlo->window(),
                              hlo->called_computations()[0], &hlo->metadata()));
      *earlier_reduce_window->mutable_shape()->mutable_layout() =
          copied->shape().layout();
      HloInstruction* later_copy =
          MakeCopyHlo(earlier_reduce_window, hlo->shape());
      TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
    }
    return absl::OkStatus();
  }
  absl::Status HandleReduce(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    if (operand->opcode() == HloOpcode::kCopy && !hlo->shape().IsTuple()) {
      HloInstruction* new_reduce = hlo->AddInstruction(
          hlo->CloneWithNewOperands(hlo->shape(), {operand->mutable_operand(0),
                                                   hlo->mutable_operand(1)}));
      TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, new_reduce));
    }
    return absl::OkStatus();
  }
  absl::Status HandleBitcastConvert(HloInstruction* hlo) override {
    return absl::OkStatus();
  }
  absl::Status HandleElementwiseUnary(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    if (hlo->opcode() == HloOpcode::kReducePrecision) {
      return absl::OkStatus();
    }
    if (operand->opcode() == HloOpcode::kCopy) {
      HloInstruction* copied = operand->mutable_operand(0);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * earlier_elementwise,
          MakeUnaryHlo(hlo->opcode(), copied, &hlo->metadata()));
      HloInstruction* later_copy =
          MakeCopyHlo(earlier_elementwise, hlo->shape());
      TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
    }
    return absl::OkStatus();
  }
  absl::Status HandleReverse(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    if (operand->opcode() == HloOpcode::kCopy) {
      HloInstruction* copied = operand->mutable_operand(0);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * earlier_reverse,
          MakeReverseHlo(copied, hlo->dimensions(), &hlo->metadata()));
      HloInstruction* later_copy = MakeCopyHlo(earlier_reverse, hlo->shape());
      TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
    }
    return absl::OkStatus();
  }
  absl::Status HandleConvert(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    if (operand->opcode() == HloOpcode::kCopy) {
      HloInstruction* copied = operand->mutable_operand(0);
      HloInstruction* earlier_convert = MakeConvertToHlo(
          copied, hlo->shape().element_type(), &hlo->metadata());
      HloInstruction* later_copy = MakeCopyHlo(earlier_convert, hlo->shape());
      TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
    }
    return absl::OkStatus();
  }
  absl::Status HandleElementwiseBinary(HloInstruction* hlo) override {
    HloInstruction* a = hlo->mutable_operand(0);
    HloInstruction* b = hlo->mutable_operand(1);
    if (a->opcode() == HloOpcode::kCopy && b->opcode() == HloOpcode::kCopy) {
      HloInstruction* copied_a = a->mutable_operand(0);
      HloInstruction* copied_b = b->mutable_operand(0);
      if (copied_a->shape() == copied_b->shape()) {
        HloInstruction* earlier_elementwise;
        if (hlo->opcode() == HloOpcode::kCompare) {
          TF_ASSIGN_OR_RETURN(
              earlier_elementwise,
              MakeCompareHlo(hlo->comparison_direction(), copied_a, copied_b,
                             &hlo->metadata()));
        } else {
          TF_ASSIGN_OR_RETURN(earlier_elementwise,
                              MakeBinaryHlo(hlo->opcode(), copied_a, copied_b,
                                            &hlo->metadata()));
        }
        HloInstruction* later_copy =
            MakeCopyHlo(earlier_elementwise, hlo->shape());
        TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
      }
    }
    return absl::OkStatus();
  }
  absl::Status HandleConcatenate(HloInstruction* hlo) override {
    const HloInstruction* first = hlo->operand(0);
    if (first->opcode() != HloOpcode::kCopy) {
      return absl::OkStatus();
    }
    const HloInstruction* inner_op = first->operand(0);
    const Layout& inner_op_layout = inner_op->shape().layout();
    std::vector<HloInstruction*> new_operands;
    new_operands.reserve(hlo->operand_count());
    for (HloInstruction* op : hlo->mutable_operands()) {
      if (op->opcode() != HloOpcode::kCopy ||
          op->operand(0)->shape().layout() != inner_op_layout) {
        VLOG(3) << "Mismatch between " << op->ToString()
                << " and expected op layout " << inner_op_layout.ToString();
        return absl::OkStatus();
      }
      new_operands.push_back(op->mutable_operand(0));
    }
    TF_ASSIGN_OR_RETURN(
        HloInstruction * new_concat,
        MakeConcatHlo(new_operands, hlo->concatenate_dimension()));
    *new_concat->mutable_shape()->mutable_layout() = inner_op_layout;
    HloInstruction* new_copy = MakeCopyHlo(new_concat, hlo->shape());
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, new_copy));
    return absl::OkStatus();
  }
};
}  
absl::StatusOr<bool> MoveCopyToUsers::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return MoveCopyToUsersVisitor{}.RunOnModule(module, execution_threads);
}
}  