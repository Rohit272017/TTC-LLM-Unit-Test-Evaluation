#include <tuple>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/comparison_util.h"
#include "xla/hlo/builder/lib/arithmetic.h"
#include "xla/hlo/builder/lib/comparators.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/tpu/tpu_defs.h"
namespace tensorflow {
namespace {
class DynamicPartitionOp : public XlaOpKernel {
 public:
  explicit DynamicPartitionOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_partitions", &num_partitions_));
  }
  xla::XlaOp CountS32(XlaOpKernelContext* ctx, xla::XlaOp input,
                      int64_t target) {
    xla::XlaOp equal_dim =
        xla::Compare(input, xla::ConstantR0<int32>(ctx->builder(), target), {},
                     xla::ComparisonDirection::kEq);
    xla::XlaOp casted = xla::ConvertElementType(equal_dim, xla::S32);
    return xla::ReduceAll(
        casted, xla::Zero(ctx->builder(), xla::S32),
        xla::CreateScalarAddComputation(xla::S32, ctx->builder()));
  }
  std::pair<std::vector<xla::XlaOp>, std::vector<xla::XlaOp>>
  DynamicPartition1D(XlaOpKernelContext* ctx, xla::XlaOp data_1d,
                     xla::XlaOp partitions_1d, const xla::Shape& data_1d_shape,
                     const xla::Shape& partition_1d_shape) {
    int64_t input_count = data_1d_shape.dimensions(0);
    std::vector<xla::XlaOp> to_sort = {partitions_1d, data_1d};
    std::vector<xla::PrimitiveType> types_to_sort = {
        partition_1d_shape.element_type(), data_1d_shape.element_type()};
    xla::XlaOp sorted = xla::Sort(
        to_sort, xla::CreateScalarLtComputation(types_to_sort, ctx->builder()),
        0,
        true);
    xla::XlaOp sorted_partitions = xla::GetTupleElement(sorted, 0);
    xla::XlaOp sorted_data = xla::GetTupleElement(sorted, 1);
    std::vector<xla::XlaOp> partition_length(num_partitions_);
    std::vector<xla::XlaOp> partition_start(num_partitions_);
    xla::XlaOp count_so_far = xla::Zero(ctx->builder(), xla::S32);
    for (int64_t i = 0; i < num_partitions_; ++i) {
      xla::XlaOp count = CountS32(ctx, sorted_partitions, i);
      partition_length[i] = count;
      partition_start[i] = count_so_far;
      count_so_far = xla::Add(count_so_far, count);
    }
    xla::PaddingConfig padding_config;
    auto* dims = padding_config.add_dimensions();
    dims->set_edge_padding_low(0);
    dims->set_edge_padding_high(input_count);
    dims->set_interior_padding(0);
    auto padded_data =
        xla::Pad(sorted_data, xla::Zero(ctx->builder(), ctx->input_xla_type(0)),
                 padding_config);
    std::vector<xla::XlaOp> output(num_partitions_);
    for (int64_t i = 0; i < num_partitions_; ++i) {
      padded_data = xla::RemoveDynamicDimension(padded_data, 0);
      auto sliced =
          xla::DynamicSlice(padded_data, {partition_start[i]}, {input_count});
      output[i] = sliced;
    }
    return {output, partition_length};
  }
  void Compile(XlaOpKernelContext* ctx) override {
    xla::Shape data_shape = ctx->InputXlaShape(0).value();
    xla::Shape partition_shape = ctx->InputXlaShape(1).value();
    xla::XlaOp data = ctx->Input(0);
    xla::XlaOp partitions = ctx->Input(1);
    std::vector<int64_t> partitions_static;
    bool partitions_are_static =
        ctx->ConstantInputReshapedToIntVector(1, &partitions_static).ok();
    if (data_shape.rank() > partition_shape.rank()) {
      std::vector<int64_t> broadcasted_dims;
      auto rank = partition_shape.rank();
      broadcasted_dims.reserve(rank);
      for (int64_t i = 0; i < rank; ++i) {
        broadcasted_dims.push_back(i);
      }
      partitions = xla::BroadcastInDim(partitions, data_shape.dimensions(),
                                       broadcasted_dims);
    }
    std::vector<int64_t> output_shape_bound_dims;
    output_shape_bound_dims.push_back(
        xla::ShapeUtil::ElementsIn(partition_shape));
    int64_t count_diff = 1;
    for (int64_t i = partition_shape.rank(); i < data_shape.rank(); ++i) {
      output_shape_bound_dims.push_back(data_shape.dimensions(i));
      count_diff *= data_shape.dimensions(i);
    }
    int64_t input_count = xla::ShapeUtil::ElementsIn(data_shape);
    auto data_1d = xla::Reshape(data, {input_count});
    auto partitions_1d = xla::Reshape(partitions, {input_count});
    xla::Shape data_1d_shape =
        xla::ShapeUtil::MakeShape(data_shape.element_type(), {input_count});
    xla::Shape partitions_1d_shape = xla::ShapeUtil::MakeShape(
        partition_shape.element_type(), {input_count});
    std::vector<xla::XlaOp> output, partition_length;
    std::tie(output, partition_length) = DynamicPartition1D(
        ctx, data_1d, partitions_1d, data_1d_shape, partitions_1d_shape);
    for (int64_t i = 0; i < num_partitions_; ++i) {
      auto reshape = xla::Reshape(output[i], output_shape_bound_dims);
      if (partitions_are_static) {
        int64_t size = absl::c_count(partitions_static, i);
        ctx->SetOutput(i, xla::SliceInDim(reshape, 0, size, 1, 0));
      } else {
        xla::XlaOp length;
        if (count_diff != 0) {
          length = xla::Div(partition_length[i],
                            xla::ConstantR0<int32>(ctx->builder(), count_diff));
        } else {
          length = CountS32(ctx, ctx->Input(1), i);
        }
        ctx->SetOutput(i, xla::SetDimensionSize(reshape, length, 0));
      }
    }
  }
 private:
  int64_t num_partitions_;
};
REGISTER_XLA_OP(Name("DynamicPartition"), DynamicPartitionOp);
}  
}  