#include "absl/log/check.h"
#include "tensorflow/compiler/tf2xla/kernels/tensor_list_utils.h"
#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.pb.h"
namespace tensorflow {
namespace {
class IdentityOp : public XlaOpKernel {
 public:
  explicit IdentityOp(OpKernelConstruction* context) : XlaOpKernel(context) {}
  void Compile(XlaOpKernelContext* ctx) override {
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      if (IsTensorListInput(ctx, i)) {
        ctx->SetTensorListOutput(i, ctx->Input(i));
      } else {
        DCHECK(ctx->input_type(i) != DT_VARIANT);
        ctx->op_kernel_context()->set_output(
            i, ctx->op_kernel_context()->input(i));
      }
    }
  }
 private:
  IdentityOp(const IdentityOp&) = delete;
  void operator=(const IdentityOp&) = delete;
};
REGISTER_XLA_OP(
    Name("Identity").AllowResourceTypes().AllowVariantTypes().CompilationOnly(),
    IdentityOp);
REGISTER_XLA_OP(Name("IdentityN")
                    .AllowResourceTypes()
                    .AllowVariantTypes()
                    .CompilationOnly(),
                IdentityOp);
REGISTER_XLA_OP(Name("PlaceholderWithDefault"), IdentityOp);
REGISTER_XLA_OP(Name("PreventGradient"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("StopGradient").AllowVariantTypes(), IdentityOp);
REGISTER_XLA_OP(Name("Snapshot"), IdentityOp);
REGISTER_XLA_OP(Name("_EagerConst"), IdentityOp);
}  
}  