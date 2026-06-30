#include <algorithm>
#include <vector>
#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/lib/matrix.h"
#include "xla/hlo/builder/lib/pooling.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
namespace tensorflow {
namespace {
xla::XlaOp CreateDiagonal(xla::XlaOp input, int64_t last_dim_size,
                          absl::Span<const int64_t> other_dims) {
  xla::XlaBuilder* builder = input.builder();
  xla::XlaOp iota = xla::Iota(builder, xla::S32, last_dim_size);
  xla::XlaOp iota_broadcast = xla::Broadcast(iota, {last_dim_size});
  xla::XlaOp mask = xla::Eq(iota_broadcast, iota, {0});
  if (!other_dims.empty()) {
    mask = xla::Broadcast(mask, other_dims);
  }
  std::vector<int64_t> out_dim_sizes(other_dims.begin(), other_dims.end());
  out_dim_sizes.push_back(last_dim_size);
  out_dim_sizes.push_back(last_dim_size);
  std::vector<int64_t> broadcast_dimensions(other_dims.size() + 1);
  absl::c_iota(broadcast_dimensions, 0);
  ++broadcast_dimensions.back();
  xla::XlaOp input_broadcast =
      xla::BroadcastInDim(input, out_dim_sizes, broadcast_dimensions);
  return xla::Select(mask, input_broadcast, xla::ZerosLike(input_broadcast));
}
class DiagOp : public XlaOpKernel {
 public:
  explicit DiagOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES(ctx, ctx->num_inputs() >= 1,
                errors::InvalidArgument("Diag op must have at an input"));
    const TensorShape input_shape = ctx->InputShape(0);
    auto dims = input_shape.dim_sizes();
    OP_REQUIRES(ctx, !dims.empty(),
                errors::InvalidArgument("Expected 1 <= dims, got shape ",
                                        input_shape.DebugString()));
    xla::XlaOp input = ctx->Input(0);
    int64_t size = input_shape.num_elements();
    input = xla::Reshape(input, {size});
    xla::XlaOp diag = CreateDiagonal(input, size, {});
    std::vector<int64_t> new_dims(dims.size() * 2);
    std::copy(dims.begin(), dims.end(), new_dims.begin());
    std::copy(dims.begin(), dims.end(), new_dims.begin() + dims.size());
    diag = xla::Reshape(diag, new_dims);
    ctx->SetOutput(0, diag);
  }
};
REGISTER_XLA_OP(Name("Diag"), DiagOp);
REGISTER_XLA_OP(Name("DiagPart"), MlirXlaOpKernel);
}  
}  