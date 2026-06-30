#include "tensorflow/core/grappler/optimizers/tfg_optimizer_hook.h"
#include <string>
#include <utility>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"  
#include "mlir/IR/Dialect.h"  
#include "mlir/IR/MLIRContext.h"  
#include "mlir/Pass/PassManager.h"  
#include "mlir/Pass/PassRegistry.h"  
#include "mlir/Support/LogicalResult.h"  
#include "tensorflow/c/tf_status.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_debug_info.pb.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/importexport/graphdef_export.h"
#include "tensorflow/core/ir/importexport/graphdef_import.h"
#include "tensorflow/core/ir/tf_op_registry.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/util/dump_graph.h"
using tensorflow::Status;
using tensorflow::errors::InvalidArgument;
namespace mlir {
namespace tfg {
class TFGGrapplerOptimizer::Impl {
 public:
  explicit Impl(TFGPassPipelineBuilder builder, unsigned num_tfg_threads)
      : ctx_(MLIRContext::Threading::DISABLED), mgr_(&ctx_) {
    DialectRegistry registry;
    registry.addExtension(+[](MLIRContext* ctx, TFGraphDialect* dialect) {
      dialect->addInterfaces<TensorFlowOpRegistryInterface>();
    });
    ctx_.appendDialectRegistry(registry);
    builder(mgr_);
    if (num_tfg_threads) {
      llvm::ThreadPoolStrategy strategy;
      strategy.ThreadsRequested = num_tfg_threads;
      threadpool_ = std::make_unique<llvm::DefaultThreadPool>(strategy);
      ctx_.setThreadPool(*threadpool_);
    }
  }
  LogicalResult RunPipeline(ModuleOp module) { return mgr_.run(module); }
  MLIRContext* GetContext() { return &ctx_; }
  std::string GetPipelineString() {
    std::string pipeline;
    llvm::raw_string_ostream os(pipeline);
    mgr_.printAsTextualPipeline(os);
    return os.str();
  }
 private:
  std::unique_ptr<llvm::DefaultThreadPool> threadpool_;
  MLIRContext ctx_;
  PassManager mgr_;
};
TFGGrapplerOptimizer::TFGGrapplerOptimizer(TFGPassPipelineBuilder builder,
                                           unsigned num_tfg_threads)
    : impl_(std::make_unique<Impl>(std::move(builder), num_tfg_threads)) {}
TFGGrapplerOptimizer::~TFGGrapplerOptimizer() = default;
std::string TFGGrapplerOptimizer::name() const {
  return absl::StrCat("tfg_optimizer{", impl_->GetPipelineString(), "}");
}
Status TFGGrapplerOptimizer::Optimize(
    tensorflow::grappler::Cluster* cluster,
    const tensorflow::grappler::GrapplerItem& item,
    tensorflow::GraphDef* optimized_graph) {
  if (VLOG_IS_ON(4)) {
    tensorflow::DumpGraphDefToFile(
        absl::StrCat("tfg_before_graph_", item.id, "_",
                     std::hash<std::string>()(name())),
        item.graph);
  }
  VLOG(5) << "TFG Before Graph: \n" << item.graph.DebugString();
  tensorflow::GraphDebugInfo debug_info;
  tensorflow::metrics::ScopedCounter<2> metrics(
      tensorflow::metrics::GetGraphOptimizationCounter(),
      {"TfgOptimizer", "convert_graphdef_to_tfg"});
  auto error_or_module =
      ImportGraphDef(impl_->GetContext(), debug_info, item.graph);
  if (!error_or_module.ok()) {
    auto status = error_or_module.status();
    tensorflow::errors::AppendToMessage(
        &status, "when importing GraphDef to MLIR module in GrapplerHook");
    LOG(ERROR) << name() << " failed: " << status.ToString();
    return absl::AbortedError(status.message());
  }
  metrics.ReportAndStop();
  ModuleOp module = (*error_or_module).get();
  if (failed(impl_->RunPipeline(module))) {
    return absl::InvalidArgumentError("MLIR Graph Optimizer failed: ");
  }
  tensorflow::GraphDef graphdef;
  metrics.Reset({"TfgOptimizer", "convert_tfg_to_graphdef"});
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      ConvertToGraphDef(module, &graphdef),
      "when exporting MLIR module to GraphDef in GrapplerHook");
  (void)graphdef.mutable_library();
  metrics.ReportAndStop();
  *optimized_graph = std::move(graphdef);
  if (VLOG_IS_ON(4)) {
    tensorflow::DumpGraphDefToFile(
        absl::StrCat("tfg_after_graph_", item.id, "_",
                     std::hash<std::string>()(name())),
        *optimized_graph);
  }
  if (VLOG_IS_ON(5)) {
    VLOG(5) << "TFG After Graph: \n"
            << optimized_graph->DebugString() << "\nMLIR module: \n";
    module.dump();
  }
  return absl::OkStatus();
}
}  
}  