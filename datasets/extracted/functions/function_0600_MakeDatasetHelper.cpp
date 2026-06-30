#include "tensorflow/core/kernels/data/optimize_dataset_op.h"
#if !defined(IS_MOBILE_PLATFORM)
#include <map>
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/rewrite_utils.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
namespace tensorflow {
namespace data {
 constexpr const char* const OptimizeDatasetOp::kDatasetType;
 constexpr const char* const OptimizeDatasetOp::kInputDataset;
 constexpr const char* const OptimizeDatasetOp::kOptimizations;
 constexpr const char* const
    OptimizeDatasetOp::kOptimizationsEnabled;
 constexpr const char* const
    OptimizeDatasetOp::kOptimizationsDisabled;
 constexpr const char* const
    OptimizeDatasetOp::kOptimizationsDefault;
 constexpr const char* const OptimizeDatasetOp::kOutputTypes;
 constexpr const char* const OptimizeDatasetOp::kOutputShapes;
 constexpr const char* const
    OptimizeDatasetOp::kOptimizationConfigs;
 constexpr const char* const OptimizeDatasetOp::kOptimizeDatasetV1;
 constexpr const char* const OptimizeDatasetOp::kOptimizeDatasetV2;
namespace {
void MakeDatasetHelper(OpKernelContext* ctx,
                       absl::flat_hash_set<tstring>& optimizations,
                       const absl::flat_hash_set<tstring>& optimization_configs,
                       DatasetBase* input, DatasetBase** output) {
  std::vector<string> graduated_experiments = {
    "disable_intra_op_parallelism",
    "use_private_thread_pool"
  };
  for (auto& experiment : graduated_experiments) {
    if (!optimizations.contains(experiment)) {
      optimizations.insert(experiment);
    }
    VLOG(1) << "The graduated experiment \"" << experiment << "\" is applied.";
  }
  if (optimizations.empty()) {
    *output = input;
    input->Ref();
    return;
  }
  auto config_factory = [&optimizations, &optimization_configs]() {
    return CreateRewriterConfig(optimizations, optimization_configs);
  };
  core::RefCountPtr<DatasetBase> rewritten;
  Status s = RewriteDataset(ctx, input, std::move(config_factory),
                            false, &rewritten);
  *output = rewritten.release();
  if (errors::IsDeadlineExceeded(s)) {
    LOG(WARNING) << s.ToString();
    *output = input;
    input->Ref();
    return;
  }
  OP_REQUIRES_OK(ctx, s);
}
}  
void OptimizeDatasetOp::MakeDatasetFromOptions(
    OpKernelContext* ctx, DatasetBase* input,
    const absl::flat_hash_set<tstring>& optimizations_enabled,
    const absl::flat_hash_set<tstring>& optimizations_disabled,
    const absl::flat_hash_set<tstring>& optimizations_default,
    const absl::flat_hash_set<tstring>& optimization_configs,
    DatasetBase** output) {
  auto experiments = GetExperiments();
  LogAndRecordExperiments(experiments);
  auto optimizations =
      SelectOptimizations(experiments, optimizations_enabled,
                          optimizations_disabled, optimizations_default);
  MakeDatasetHelper(ctx, optimizations, optimization_configs, input, output);
}
OptimizeDatasetOp::OptimizeDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
  auto& op_name = ctx->def().op();
  if (op_name == kOptimizeDatasetV1) {
    op_version_ = 1;
  } else if (op_name == kOptimizeDatasetV2) {
    op_version_ = 2;
  }
  std::vector<tstring> optimization_configs;
  OP_REQUIRES_OK(ctx,
                 ctx->GetAttr(kOptimizationConfigs, &optimization_configs));
  optimization_configs_.insert(optimization_configs.begin(),
                               optimization_configs.end());
}
void OptimizeDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                    DatasetBase** output) {
  absl::flat_hash_set<tstring> optimizations;
  if (op_version_ == 1) {
    std::vector<tstring> optimizations_enabled;
    OP_REQUIRES_OK(ctx, ParseVectorArgument<tstring>(ctx, kOptimizations,
                                                     &optimizations_enabled));
    optimizations.insert(optimizations_enabled.begin(),
                         optimizations_enabled.end());
  } else if (op_version_ == 2) {
    std::vector<tstring> optimizations_enabled, optimizations_disabled,
        optimizations_default;
    OP_REQUIRES_OK(ctx, ParseVectorArgument<tstring>(ctx, kOptimizationsEnabled,
                                                     &optimizations_enabled));
    OP_REQUIRES_OK(ctx,
                   ParseVectorArgument<tstring>(ctx, kOptimizationsDisabled,
                                                &optimizations_disabled));
    OP_REQUIRES_OK(ctx, ParseVectorArgument<tstring>(ctx, kOptimizationsDefault,
                                                     &optimizations_default));
    auto experiments = GetExperiments();
    LogAndRecordExperiments(experiments);
    optimizations = SelectOptimizations(
        experiments,
        {optimizations_enabled.begin(), optimizations_enabled.end()},
        {optimizations_disabled.begin(), optimizations_disabled.end()},
        {optimizations_default.begin(), optimizations_default.end()});
  }
  MakeDatasetHelper(
      ctx, optimizations,
      {optimization_configs_.begin(), optimization_configs_.end()}, input,
      output);
}
namespace {
REGISTER_KERNEL_BUILDER(Name("OptimizeDataset").Device(DEVICE_CPU),
                        OptimizeDatasetOp);
REGISTER_KERNEL_BUILDER(Name("OptimizeDatasetV2").Device(DEVICE_CPU),
                        OptimizeDatasetOp);
}  
}  
}  
#else   
namespace tensorflow {
namespace data {
void OptimizeDatasetOp::MakeDatasetFromOptions(
    OpKernelContext* ctx, DatasetBase* input,
    const absl::flat_hash_set<tstring>& optimizations_enabled,
    const absl::flat_hash_set<tstring>& optimizations_disabled,
    const absl::flat_hash_set<tstring>& optimizations_default,
    const absl::flat_hash_set<tstring>& optimization_configs,
    DatasetBase** output) {
  input->Ref();
  *output = input;
}
OptimizeDatasetOp::OptimizeDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {}
void OptimizeDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                    DatasetBase** output) {
  input->Ref();
  *output = input;
}
namespace {
REGISTER_KERNEL_BUILDER(Name("OptimizeDataset").Device(DEVICE_CPU),
                        OptimizeDatasetOp);
REGISTER_KERNEL_BUILDER(Name("OptimizeDatasetV2").Device(DEVICE_CPU),
                        OptimizeDatasetOp);
}  
}  
}  
#endif  