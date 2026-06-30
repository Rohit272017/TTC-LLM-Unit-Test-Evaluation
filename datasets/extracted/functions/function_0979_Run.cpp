#include "tensorflow/compiler/jit/force_xla_constants_on_host_pass.h"
#include "tensorflow/compiler/jit/compilability_check_util.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
namespace tensorflow {
Status ForceXlaConstantsOnHostPass::Run(
    const GraphOptimizationPassOptions& options) {
  Graph* graph = options.graph->get();
  OptimizerOptions opts;
  auto pflr = std::make_unique<ProcessFunctionLibraryRuntime>(
      nullptr, options.session_options->env, nullptr,
      TF_GRAPH_DEF_VERSION, options.flib_def, opts);
  FunctionLibraryRuntime* flr =
      pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);
  for (Node* node : graph->nodes()) {
    if (CanCreateXlaKernel(node->def())) {
      const FunctionBody* fbody = nullptr;
      std::vector<int> constant_arg_indices;
      std::vector<int> resource_arg_indices;
      NameAttrList function;
      TF_RETURN_IF_ERROR(NameAndAttrsFromFunctionCall(node->def(), &function));
      TF_RETURN_IF_ERROR(GetBodyAndConstantsAndResources(
          flr, function, &fbody, &constant_arg_indices, &resource_arg_indices));
      VLOG(3) << "Found constant arg indices: "
              << absl::StrJoin(constant_arg_indices, ", ");
      node->AddAttr("_input_hostmem", constant_arg_indices);
    }
  }
  return absl::OkStatus();
}
}  