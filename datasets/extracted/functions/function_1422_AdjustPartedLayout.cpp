#include "tensorflow/dtensor/mlir/spmd_expander.h"
#include <climits>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/IR/BuiltinTypes.h"  
#include "mlir/IR/Operation.h"  
#include "mlir/IR/OperationSupport.h"  
#include "mlir/Support/LLVM.h"  
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/framework/registration/registration.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/dtensor_utils.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/expansions/replicated_spmd_expander.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/proto/layout.pb.h"
namespace tensorflow {
namespace dtensor {
namespace {
Status AdjustPartedLayout(const llvm::DenseMap<int, Layout>& input_layouts,
                          llvm::DenseMap<int, Layout>* computed_layouts) {
  bool input_has_parted_layout = false;
  for (const auto& input_layout : input_layouts) {
    if (input_layout.second.type() == Layout::LayoutType::kParted) {
      input_has_parted_layout = true;
      break;
    }
  }
  if (input_has_parted_layout) {
    for (auto& computed_layout : *computed_layouts) {
      TF_ASSIGN_OR_RETURN(Layout parted, computed_layout.second.ToParted());
      computed_layout.getSecond() = parted;
    }
  }
  return absl::OkStatus();
}
bool SkipExpansionForPartedLayout(mlir::Operation* op) {
  if (llvm::isa<mlir::func::ReturnOp, mlir::tf_device::ReturnOp>(op)) {
    return false;
  }
  auto status_or_input_layouts = ExtractRequiredLayoutFromOperands(op);
  if (!status_or_input_layouts.ok()) {
    return false;
  }
  bool operand_uses_parted_layout = false;
  for (const auto& layout : status_or_input_layouts.value()) {
    if (layout.type() == Layout::LayoutType::kParted) {
      operand_uses_parted_layout = true;
      break;
    }
  }
  return operand_uses_parted_layout;
}
}  
SPMDExpanderRegistry* SPMDExpanderRegistry::Global() {
  static SPMDExpanderRegistry* registry = new SPMDExpanderRegistry();
  return registry;
}
SPMDExpanderBase* SPMDExpanderRegistry::GetPropagateFnForFullOpName(
    const std::string& full_op_name) {
  auto key = full_op_name;
  auto fn = op_to_propagate_fn_map_.find(key);
  if (fn == op_to_propagate_fn_map_.end()) {
    if (EnableReplicatedSpmdAsDefault(key)) {
      LOG(WARNING)
          << full_op_name << " is defaulting to ReplicatedOpSPMDExpander. This "
          << " has performance implications as all inputs and outputs "
          << " will be replicated if they are not already. Please file a "
          << " feature request to TF DTensor to implement an efficient "
          << " SPMD for this operation.";
      RegisterPropagateFn(key, std::make_unique<ReplicatedOpSPMDExpander>(
                                   true));
      return op_to_propagate_fn_map_.find(key)->second.get();
    } else {
      return nullptr;
    }
  }
  return fn->second.get();
}
SPMDExpanderBase* SPMDExpanderRegistry::GetPropagateFnForOp(
    mlir::Operation* op) {
  return GetPropagateFnForFullOpName(OpName(op));
}
InitOnStartupMarker SPMDExpanderRegistry::RegisterPropagateFn(
    std::string opName, std::unique_ptr<SPMDExpanderBase> prop) {
  CHECK(op_to_propagate_fn_map_  
            .insert_or_assign(opName, std::move(prop))
            .second);
  return {};
}
Status SPMDExpanderBase::ExpandOpAndSetLayout(mlir::Operation* op,
                                              mlir::Operation** output) {
  TF_ASSIGN_OR_RETURN(std::vector<std::optional<Layout>> computed_layout,
                      ExtractLayoutFromOp(op));
  if (computed_layout.empty() && op->getNumResults() != 0) {
    return errors::InvalidArgument(
        absl::StrCat("No attached layout found for op : ", OpName(op),
                     " This might be due to an error in layout propagation.")
            .c_str());
  }
  TF_ASSIGN_OR_RETURN(const Mesh& mesh, ExtractDeviceMeshEnclosingCluster(op));
  bool skip_expansion_for_parted_layout = SkipExpansionForPartedLayout(op);
  if (mesh.IsSingleDevice() || mesh.use_xla_spmd() ||
      skip_expansion_for_parted_layout) {
    if (skip_expansion_for_parted_layout) {
      *output = InferSPMDExpandedLocalShape(op);
    } else {
      *output = op;
    }
    SetLayoutOnOp(*output, absl::Span<std::optional<Layout>>(
                               computed_layout.data(), computed_layout.size()));
    return absl::OkStatus();
  }
  llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> global_output_shapes;
  global_output_shapes.reserve(op->getNumResults());
  for (auto output_value : op->getResults()) {
    auto maybe_ranked =
        mlir::dyn_cast<mlir::RankedTensorType>(output_value.getType());
    if (llvm::isa<mlir::TF::RestoreV2Op, mlir::TF::DTensorRestoreV2Op>(op) &&
        (!maybe_ranked || !maybe_ranked.hasStaticShape()))
      continue;
    TF_ASSIGN_OR_RETURN(auto global_shape,
                        ExtractGlobalOutputShape(output_value));
    global_output_shapes.emplace_back(llvm::SmallVector<int64_t, 4>{
        global_shape.begin(), global_shape.end()});
  }
  TF_ASSIGN_OR_RETURN(*output, this->ExpandOp(op));
  SetLayoutOnOp(*output, absl::Span<std::optional<Layout>>(
                             computed_layout.data(), computed_layout.size()));
  for (const auto& output_layout_and_index :
       llvm::enumerate(llvm::zip((*output)->getResults(), computed_layout))) {
    const int index = output_layout_and_index.index();
    const auto& output_and_layout = output_layout_and_index.value();
    auto output_value = std::get<0>(output_and_layout);
    auto local_expanded_shape_or_status = GetShapeOfValue(output_value);
    if (!local_expanded_shape_or_status.ok()) continue;
    const auto local_expanded_shape = local_expanded_shape_or_status.value();
    const auto& layout = std::get<1>(output_and_layout);
    const auto expected_global_shape =
        layout->GlobalShapeFromLocalShape(local_expanded_shape);
    for (const auto& expanded_and_true_global_shape :
         llvm::zip(global_output_shapes[index], expected_global_shape)) {
      const auto expanded_shape = std::get<0>(expanded_and_true_global_shape);
      const auto expected_shape = std::get<1>(expanded_and_true_global_shape);
      if (expanded_shape <= 0 || expected_shape <= 0) continue;
      if (expanded_shape != expected_shape) {
        return errors::Internal(
            "SPMD expansion resulted in op output inconsistent with the "
            "provided layout. Expected shape: <",
            absl::StrJoin(expected_global_shape, ","), "> got shape: <",
            absl::StrJoin(global_output_shapes[index], ","), ">");
      }
    }
  }
  return absl::OkStatus();
}
StatusOr<llvm::DenseMap<int, Layout>> SPMDExpanderBase::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  return errors::Unimplemented(
      "ComputeLayoutForward API must be implemented via the subclass.");
}
StatusOr<llvm::DenseMap<int, Layout>> SPMDExpanderBase::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts,
    const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(const Mesh& mesh, ExtractDeviceMeshEnclosingCluster(op));
  if (mesh.IsSingleDevice()) {
    TF_ASSIGN_OR_RETURN(
        Layout layout,
        Layout::GetLayout(Layout::LayoutType::kSingleDevice, {}, mesh));
    auto layouts = llvm::DenseMap<int, Layout>{};
    for (int i = 0; i < op->getNumResults(); ++i) {
      layouts.insert({i, layout});
    }
    return layouts;
  }
  TF_ASSIGN_OR_RETURN(auto layouts, ComputeLayoutForward(op, input_layouts));
  TF_RETURN_IF_ERROR(AdjustPartedLayout(input_layouts, &layouts));
  return layouts;
}
StatusOr<llvm::DenseMap<int, Layout>> SPMDExpanderBase::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  return errors::Unimplemented(
      "ComputeLayoutBackward API must be implemented via the subclass.");
}
StatusOr<llvm::DenseMap<int, Layout>> SPMDExpanderBase::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts,
    const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(const Mesh& mesh, ExtractDeviceMeshEnclosingCluster(op));
  if (mesh.IsSingleDevice()) {
    TF_ASSIGN_OR_RETURN(
        Layout layout,
        Layout::GetLayout(Layout::LayoutType::kSingleDevice, {}, mesh));
    auto layouts = llvm::DenseMap<int, Layout>{};
    for (int i = 0; i < op->getNumOperands(); ++i) {
      layouts.insert({i, layout});
    }
    return layouts;
  }
  return ComputeLayoutBackward(op, output_layouts);
}
Status RunSPMDExpansion(mlir::Operation* op, mlir::Operation** output) {
  SPMDExpanderBase* expander =
      SPMDExpanderRegistry::Global()->GetPropagateFnForOp(op);
  if (expander != nullptr) {
    return expander->ExpandOpAndSetLayout(op, output);
  } else {
    VLOG(1) << "No expansion found for " << OpName(op) << "\n";
    *output = op;
  }
  return absl::OkStatus();
}
}  
}  