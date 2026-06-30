#include "tensorflow/core/kernels/data/rewrite_dataset_op.h"
#if !defined(IS_MOBILE_PLATFORM)
#include <map>
#include <string>
#include "tensorflow/core/data/rewrite_utils.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
namespace tensorflow {
namespace data {
 constexpr const char* const RewriteDatasetOp::kDatasetType;
 constexpr const char* const RewriteDatasetOp::kInputDataset;
 constexpr const char* const RewriteDatasetOp::kRewriteName;
 constexpr const char* const RewriteDatasetOp::kOutputTypes;
 constexpr const char* const RewriteDatasetOp::kOutputShapes;
RewriteDatasetOp::RewriteDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {}
void RewriteDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                   DatasetBase** output) {
  tstring rewrite_name;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kRewriteName, &rewrite_name));
  auto config_factory = [rewrite_name]() {
    RewriterConfig rewriter_config;
    rewriter_config.add_optimizers(std::string(rewrite_name));
    rewriter_config.set_meta_optimizer_iterations(RewriterConfig::ONE);
    rewriter_config.set_fail_on_optimizer_errors(true);
    return rewriter_config;
  };
  core::RefCountPtr<DatasetBase> rewritten;
  OP_REQUIRES_OK(ctx, RewriteDataset(ctx, input, std::move(config_factory),
                                     false, &rewritten));
  *output = rewritten.release();
}
namespace {
REGISTER_KERNEL_BUILDER(Name("RewriteDataset").Device(DEVICE_CPU),
                        RewriteDatasetOp);
}  
}  
}  
#else   
namespace tensorflow {
namespace data {
RewriteDatasetOp::RewriteDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {}
void RewriteDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                   DatasetBase** output) {
  input->Ref();
  *output = input;
}
namespace {
REGISTER_KERNEL_BUILDER(Name("RewriteDataset").Device(DEVICE_CPU),
                        RewriteDatasetOp);
}  
}  
}  
#endif  