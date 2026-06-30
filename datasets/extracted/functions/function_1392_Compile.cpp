#include "tensorflow/compiler/tf2xla/kernels/cwise_ops.h"
#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2xla/lib/broadcast.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/util/bcast.h"
namespace tensorflow {
void XlaBinaryOp::Compile(XlaOpKernelContext* ctx) {
  TensorShape lhs_shape = ctx->InputShape(0);
  TensorShape rhs_shape = ctx->InputShape(1);
  xla::Shape lhs_xla_shape = ctx->InputXlaShape(0).value();
  xla::Shape rhs_xla_shape = ctx->InputXlaShape(1).value();
  auto lhs_handle = ctx->Input(0);
  auto rhs_handle = ctx->Input(1);
  if (lhs_shape.dims() == rhs_shape.dims()) {
    auto reconcile_tensor_mismatched_dims = [ctx](
                                                xla::XlaOp lhs, xla::XlaOp rhs,
                                                const xla::Shape& lhs_xla_shape,
                                                const xla::Shape& rhs_xla_shape,
                                                TensorShape* lhs_tensor_shape) {
      for (int64_t i = 0; i < lhs_xla_shape.rank(); ++i) {
        if (lhs_xla_shape.is_dynamic_dimension(i)) {
          if (!rhs_xla_shape.is_dynamic_dimension(i) &&
              lhs_xla_shape.dimensions(i) > rhs_xla_shape.dimensions(i) &&
              rhs_xla_shape.dimensions(i) != 1) {
            auto size = xla::GetDimensionSize(lhs, i);
            lhs = xla::SliceInDim(lhs, 0, rhs_xla_shape.dimensions(i), 1,
                                  i);
            lhs_tensor_shape->set_dim(i, rhs_xla_shape.dimensions(i));
            lhs = xla::SetDimensionSize(lhs, size, i);
          }
          if (rhs_xla_shape.is_dynamic_dimension(i) &&
              lhs_xla_shape.dimensions(i) < rhs_xla_shape.dimensions(i) &&
              rhs_xla_shape.dimensions(i) != 1 &&
              lhs_xla_shape.dimensions(i) != 1) {
            auto size = xla::GetDimensionSize(lhs, i);
            int64_t diff =
                rhs_xla_shape.dimensions(i) - lhs_xla_shape.dimensions(i);
            lhs = xla::PadInDim(
                lhs, xla::Zero(ctx->builder(), lhs_xla_shape.element_type()), i,
                0, diff);
            lhs_tensor_shape->set_dim(i, rhs_xla_shape.dimensions(i));
            lhs = xla::SetDimensionSize(lhs, size, i);
          }
          if (lhs_xla_shape.dimensions(i) == 1 &&
              rhs_xla_shape.dimensions(i) != 1) {
            auto size = xla::GetDimensionSize(lhs, i);
            lhs = xla::RemoveDynamicDimension(lhs, i);
            std::vector<int64_t> dimensions(lhs_xla_shape.dimensions().begin(),
                                            lhs_xla_shape.dimensions().end());
            dimensions[i] = rhs_xla_shape.dimensions(i);
            std::vector<int64_t> broadcast_dimensions(lhs_xla_shape.rank());
            absl::c_iota(broadcast_dimensions, 0);
            lhs = xla::BroadcastInDim(lhs, dimensions, broadcast_dimensions);
            xla::XlaOp rhs_size;
            if (rhs_xla_shape.is_dynamic_dimension(i)) {
              rhs_size = xla::GetDimensionSize(rhs, i);
            } else {
              rhs_size = xla::ConstantR0<int32_t>(lhs.builder(),
                                                  rhs_xla_shape.dimensions(i));
            }
            size = xla::Mul(size, rhs_size);
            lhs = xla::SetDimensionSize(lhs, size, i);
            lhs_tensor_shape->set_dim(i, rhs_xla_shape.dimensions(i));
          }
        }
      }
      return lhs;
    };
    lhs_handle = reconcile_tensor_mismatched_dims(
        lhs_handle, rhs_handle, lhs_xla_shape, rhs_xla_shape, &lhs_shape);
    rhs_handle = reconcile_tensor_mismatched_dims(
        rhs_handle, lhs_handle, rhs_xla_shape, lhs_xla_shape, &rhs_shape);
  }
  BCast bcast(BCast::FromShape(lhs_shape), BCast::FromShape(rhs_shape),
              false);
  if (!bcast.IsValid()) {
    ctx->SetStatus(absl::InvalidArgumentError(
        absl::StrCat("Incompatible shapes: ", lhs_shape.DebugString(), " vs. ",
                     rhs_shape.DebugString())));
    return;
  }
  std::vector<int64_t> extend_dimension;
  int max_rank = std::max(lhs_shape.dims(), rhs_shape.dims());
  int min_rank = std::min(lhs_shape.dims(), rhs_shape.dims());
  if (min_rank != max_rank) {
    for (int i = 0; i < min_rank; ++i) {
      extend_dimension.push_back(max_rank - min_rank + i);
    }
  }
  xla::XlaOp output =
      Computation(ctx, lhs_handle, lhs_shape.dim_sizes(), rhs_handle,
                  rhs_shape.dim_sizes(), bcast, extend_dimension);
  ctx->SetOutput(0, output);
}
 std::pair<xla::XlaOp, xla::XlaOp> XlaBinaryOp::Broadcast(
    xla::XlaOp lhs, xla::XlaOp rhs, const BCast& broadcast_helper) {
  auto lhs_output = BroadcastTo(lhs, broadcast_helper.output_shape());
  if (!lhs_output.ok()) {
    xla::XlaOp error = lhs.builder()->ReportError(lhs_output.status());
    return {error, error};
  }
  auto rhs_output = BroadcastTo(rhs, broadcast_helper.output_shape());
  if (!rhs_output.ok()) {
    xla::XlaOp error = rhs.builder()->ReportError(rhs_output.status());
    return {error, error};
  }
  return {lhs_output.value(), rhs_output.value()};
}
}  