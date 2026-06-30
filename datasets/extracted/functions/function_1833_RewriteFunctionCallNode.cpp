#include "tensorflow/core/common_runtime/lower_function_call_op.h"
#include <utility>
#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "tensorflow/core/common_runtime/function_def_utils.h"
#include "tensorflow/core/common_runtime/inline_function_utils.h"
#include "tensorflow/core/common_runtime/lower_function_call_inline_policy.h"
#include "tensorflow/core/config/flag_defs.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/refcount.h"
namespace tensorflow {
using KeepCallerNode = InlineFunctionBodyOptions::KeepCallerNode;
using OutputControlSrc = InlineFunctionBodyOptions::OutputControlSource;
Status RewriteFunctionCallNode(Node* n, Graph* g,
                               const FunctionLibraryDefinition& flib_def,
                               bool keep_caller_fetchable) {
  VLOG(2) << "Lower function call node: " << SummarizeNode(*n);
  InlineFunctionBodyOptions inline_options;
  inline_options.keep_caller_node = keep_caller_fetchable
                                        ? KeepCallerNode::kFetchable
                                        : KeepCallerNode::kTargetable;
  FunctionCallInlinePolicy policy = GetFunctionCallInlinePolicy(n);
  if (policy == FunctionCallInlinePolicy::kMultiDevicePlacer) {
    inline_options.output_control_src = OutputControlSrc::kControlOutputs;
    inline_options.inlined_function_body_placer =
        InlinedFunctionBodyPlacer::MultiDevice();
  } else if (policy == FunctionCallInlinePolicy::kSingleDevicePlacer) {
    inline_options.output_control_src = OutputControlSrc::kDataOutputs;
    inline_options.inlined_function_body_placer =
        InlinedFunctionBodyPlacer::SingleDevice();
  } else {
    return errors::InvalidArgument("Unsupported function inlining policy");
  }
  core::RefCountPtr<FunctionRecord> fdef;
  if (n->IsPartitionedCall()) {
    NameAttrList func;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "f", &func));
    fdef = flib_def.FindRecord(func.name());
  } else if (n->type_string() == FunctionLibraryDefinition::kGradientOp) {
    VLOG(2) << "Skip SymbolicGradient lowering";
    return absl::OkStatus();
  } else {
    fdef = flib_def.FindRecord(n->type_string());
  }
  if (fdef == nullptr) {
    return errors::Internal("Can't find a function: node=", SummarizeNode(*n));
  }
  std::unique_ptr<FunctionBody> fbody;
  TF_RETURN_IF_ERROR(
      FunctionDefToBodyHelper(std::move(fdef), n->attrs(), &flib_def, &fbody));
  if (flags::Global().enable_function_pruning_before_inlining.value()) {
    VLOG(2) << "Pruning enabled before inlining";
    PruneFunctionBody(
        fbody->record->fdef(), fbody->graph,
        absl::Span<Node*>(fbody->arg_nodes.data(), fbody->arg_nodes.size()));
  } else {
    VLOG(2) << "Pruning disabled before inlining";
  }
  Status can_inline_function_call =
      ValidateInlining(n, fbody.get(), inline_options);
  if (can_inline_function_call.ok()) {
    TF_RETURN_IF_ERROR(
        InlineFunctionBody(flib_def, g, n, fbody.get(), inline_options));
  } else {
    VLOG(2) << "Failed to inline function call node: "
            << can_inline_function_call.message();
  }
  return absl::OkStatus();
}
}  