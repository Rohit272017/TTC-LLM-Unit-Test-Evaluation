#include "tensorflow/core/common_runtime/function_optimization_registry.h"
#include <string>
#include "tensorflow/core/framework/metrics.h"
namespace tensorflow {
void FunctionOptimizationPassRegistry::Init(
    std::unique_ptr<FunctionOptimizationPass> pass) {
  DCHECK(!pass_) << "Only one pass should be set.";
  pass_ = std::move(pass);
}
Status FunctionOptimizationPassRegistry::Run(
    const std::string& function_name, const DeviceSet& device_set,
    const ConfigProto& config_proto,
    const FunctionOptimizationPass::FunctionOptions& function_options,
    std::unique_ptr<Graph>* graph, FunctionLibraryDefinition* flib_def,
    std::vector<std::string>* control_ret_node_names,
    bool* control_rets_updated) {
  if (!pass_) return absl::OkStatus();
  tensorflow::metrics::ScopedCounter<2> timings(
      tensorflow::metrics::GetGraphOptimizationCounter(),
      {"GraphOptimizationPass", "FunctionOptimizationPassRegistry"});
  return pass_->Run(function_name, device_set, config_proto, function_options,
                    graph, flib_def, control_ret_node_names,
                    control_rets_updated);
}
FunctionOptimizationPassRegistry& FunctionOptimizationPassRegistry::Global() {
  static FunctionOptimizationPassRegistry* kGlobalRegistry =
      new FunctionOptimizationPassRegistry;
  return *kGlobalRegistry;
}
}  