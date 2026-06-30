#include "xla/service/host_memory_transfer_asyncifier.h"
#include <cstdint>
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace {
class HostMemoryTransferAsyncifierVisitor : public DfsHloVisitorWithDefault {
 public:
  explicit HostMemoryTransferAsyncifierVisitor(int64_t host_memory_space_color)
      : kHostMemorySpaceColor(host_memory_space_color) {}
  bool Changed() const { return changed_; }
  absl::Status DefaultAction(HloInstruction* hlo_instruction) override {
    return absl::OkStatus();
  }
  absl::Status HandleDynamicSlice(HloInstruction* dynamic_slice) override {
    HloInstruction* dynamic_slice_operand = dynamic_slice->mutable_operand(0);
    if (!dynamic_slice->shape().has_layout()) {
      return InternalStrCat(dynamic_slice->name(), " does not have a layout.");
    }
    if (!dynamic_slice_operand->shape().has_layout()) {
      return InternalStrCat(dynamic_slice->name(), "'s operand, ",
                            dynamic_slice_operand->name(),
                            ", does not have a layout.");
    }
    VLOG(3) << absl::StreamFormat(
        "\"%s\" from S(%d) to S(%d)", dynamic_slice->name(),
        dynamic_slice_operand->shape().layout().memory_space(),
        dynamic_slice->shape().layout().memory_space());
    if (dynamic_slice_operand->shape().layout().memory_space() !=
        kHostMemorySpaceColor) {
      return absl::OkStatus();
    }
    if (dynamic_slice->shape().layout().memory_space() !=
        xla::Layout::kDefaultMemorySpace) {
      return absl::OkStatus();
    }
    const Shape context_shape = ShapeUtil::MakeScalarShape(U32);
    const Shape transfer_bytes_shape = ShapeUtil::MakeScalarShape(S32);
    TF_ASSIGN_OR_RETURN(
        HloInstruction * async_done,
        dynamic_slice->parent()->CreateAsyncInstructions(
            dynamic_slice, {context_shape, transfer_bytes_shape}));
    VLOG(1) << "DynamicSlice \"" << dynamic_slice->ToString()
            << "\" is slicing from host memory. Converting to async "
            << async_done->ToString();
    MarkAsChanged();
    return absl::OkStatus();
  }
  absl::Status HandleDynamicUpdateSlice(
      HloInstruction* dynamic_update_slice) override {
    HloInstruction* dynamic_update_slice_operand =
        dynamic_update_slice->mutable_operand(0);
    HloInstruction* dynamic_update_slice_update =
        dynamic_update_slice->mutable_operand(1);
    if (!dynamic_update_slice->shape().has_layout()) {
      return InternalStrCat(dynamic_update_slice->name(),
                            " does not have a layout.");
    }
    if (!dynamic_update_slice_operand->shape().has_layout()) {
      return InternalStrCat(dynamic_update_slice->name(), "'s operand, ",
                            dynamic_update_slice_operand->name(),
                            ", does not have a layout.");
    }
    if (!dynamic_update_slice_update->shape().has_layout()) {
      return InternalStrCat(dynamic_update_slice->name(), "'s update, ",
                            dynamic_update_slice_update->name(),
                            ", does not have a layout.");
    }
    if (dynamic_update_slice_update->shape().layout().memory_space() !=
        xla::Layout::kDefaultMemorySpace) {
      return absl::OkStatus();
    }
    if (dynamic_update_slice->shape().layout().memory_space() !=
        kHostMemorySpaceColor) {
      return absl::OkStatus();
    }
    if (dynamic_update_slice_operand->shape().layout().memory_space() !=
        dynamic_update_slice->shape().layout().memory_space()) {
      return InternalStrCat(
          "Unexpected that ", dynamic_update_slice_operand->name(),
          "'s memory space is not the same as the dynamic-update-slice.");
    }
    const Shape context_shape = ShapeUtil::MakeScalarShape(U32);
    TF_ASSIGN_OR_RETURN(HloInstruction * async_done,
                        dynamic_update_slice->parent()->CreateAsyncInstructions(
                            dynamic_update_slice, {context_shape}));
    VLOG(1) << "DynamicUpdateSlice \"" << dynamic_update_slice->ToString()
            << "\" is slicing into host memory space. Converting to async "
            << async_done->ToString();
    MarkAsChanged();
    return absl::OkStatus();
  }
  absl::Status HandleCopy(HloInstruction* copy) override {
    HloInstruction* operand = copy->mutable_operand(0);
    if (!operand->shape().has_layout()) {
      return InternalStrCat(operand->name(), " does not have a layout.");
    }
    if (!copy->shape().has_layout()) {
      return InternalStrCat(copy->name(), " does not have a layout.");
    }
    const auto copy_src_memory_space = operand->shape().layout().memory_space();
    const auto copy_dst_memory_space = copy->shape().layout().memory_space();
    if (!((copy_src_memory_space == kHostMemorySpaceColor &&
           copy_dst_memory_space == xla::Layout::kDefaultMemorySpace) ||
          (copy_src_memory_space == xla::Layout::kDefaultMemorySpace &&
           copy_dst_memory_space == kHostMemorySpaceColor))) {
      VLOG(2)
          << "Skipping copy because it is not a copy between device memory and "
             "host memory: "
          << copy->ToString();
      return absl::OkStatus();
    }
    const Shape context_shape = ShapeUtil::MakeScalarShape(U32);
    TF_ASSIGN_OR_RETURN(
        HloInstruction * async_done,
        copy->parent()->CreateAsyncInstructions(copy, {context_shape}));
    VLOG(1)
        << "Copy \"" << copy->name()
        << "\" is between device and host memory space. Converting to async "
        << async_done->ToString();
    MarkAsChanged();
    return absl::OkStatus();
  }
 private:
  const int64_t kHostMemorySpaceColor;
  bool changed_ = false;
  void MarkAsChanged() { changed_ = true; }
};
}  
absl::StatusOr<bool> HostMemoryTransferAsyncifier::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  HostMemoryTransferAsyncifierVisitor visitor(kHostMemorySpaceColor);
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  }
  return visitor.Changed();
}
}  