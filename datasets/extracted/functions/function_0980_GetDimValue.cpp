#include <algorithm>
#include <numeric>
#include <string>
#include "absl/algorithm/container.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Support/LLVM.h"  
#include "tensorflow/compiler/mlir/lite/core/c/builtin_op_data.h"
#include "tensorflow/compiler/mlir/lite/kernels/padding.h"
#include "tensorflow/compiler/mlir/quantization/common/attrs_and_constraints.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/constant_fold.h"
#include "xla/xla_data.pb.h"
namespace mlir::quant {
namespace {
Value GetDimValue(OpBuilder &builder, Location loc, Value shape_value,
                  int32_t dim) {
  Type attribute_type = builder.getI64Type();
  return builder.create<TF::StridedSliceOp>(
      loc,
      RankedTensorType::get(
          {}, mlir::cast<ShapedType>(shape_value.getType()).getElementType()),
      shape_value,
      Create1DConstValue<int32_t>(builder, loc, {dim}),
      Create1DConstValue<int32_t>(builder, loc, {dim + 1}),
      Create1DConstValue<int32_t>(builder, loc, {1}),
      builder.getIntegerAttr(attribute_type, 0),
      builder.getIntegerAttr(attribute_type, 0),
      builder.getIntegerAttr(attribute_type, 0),
      builder.getIntegerAttr(attribute_type, 0),
      builder.getIntegerAttr(attribute_type, 1));
}
void GetSamePaddingValues(OpBuilder &builder, Location loc, Value input_size,
                          int64_t filter_sz, int64_t dilation_rate,
                          int64_t stride, Value &padding_low,
                          Value &padding_high) {
  Value zero = CreateScalarConstValue<int32_t>(builder, loc, 0);
  Value one = CreateScalarConstValue<int32_t>(builder, loc, 1);
  Value two = CreateScalarConstValue<int32_t>(builder, loc, 2);
  Value filter_size = CreateScalarConstValue<int32_t>(builder, loc, filter_sz);
  Type int32_scalar_type = zero.getType();
  auto scalar_add = [&](Value lhs, Value rhs) {
    return builder.create<TF::AddOp>(loc, int32_scalar_type, lhs, rhs);
  };
  auto scalar_mul = [&](Value lhs, Value rhs) {
    return builder.create<TF::MulOp>(loc, int32_scalar_type, lhs, rhs);
  };
  auto scalar_sub = [&](Value lhs, Value rhs) {
    return builder.create<TF::SubOp>(loc, int32_scalar_type, lhs, rhs);
  };
  auto scalar_div = [&](Value lhs, Value rhs) {
    return builder.create<TF::DivOp>(loc, int32_scalar_type, lhs, rhs);
  };
  Value stride_value = CreateScalarConstValue<int32_t>(builder, loc, stride);
  Value dilation_rate_value =
      CreateScalarConstValue<int32_t>(builder, loc, dilation_rate);
  Value effective_filter_size_op = scalar_add(
      scalar_mul(dilation_rate_value, scalar_sub(filter_size, one)), one);
  Value output_size = scalar_div(
      scalar_add(input_size, scalar_sub(stride_value, one)), stride_value);
  Value padding_needed = scalar_sub(
      scalar_add(effective_filter_size_op,
                 scalar_mul(stride_value, scalar_sub(output_size, one))),
      input_size);
  padding_needed = builder.create<TF::MaximumOp>(loc, padding_needed, zero);
  padding_low = scalar_div(padding_needed, two);
  padding_high = scalar_sub(padding_needed, padding_low);
}
Value PadForDynamicShapedInputSamePadding(
    OpBuilder &builder, Location loc, Value input, Value filter,
    int8_t input_zp_value, ArrayAttr strides, ArrayAttr dilations,
    StringAttr conv_padding, Value &padding, int num_dims) {
  Value zero_rank1 = CreateConstValue<int32_t>(builder, loc, {1}, {0});
  SmallVector<Value> temp_padding_values{zero_rank1, zero_rank1};
  auto reshape_op = [&](Value value, const SmallVector<int64_t> &shape) {
    const int64_t rank = shape.size();
    return builder.create<TF::ReshapeOp>(
        loc, RankedTensorType::get(shape, builder.getI32Type()), value,
        CreateConstValue<int64_t>(builder, loc, {rank}, shape));
  };
  ShapedType filter_shape = mlir::cast<ShapedType>(filter.getType());
  Value input_shape_value = builder.create<TF::ShapeOp>(
      loc, RankedTensorType::get({num_dims}, builder.getI32Type()), input);
  auto scalar_to_rank1 = [&](Value value) { return reshape_op(value, {1}); };
  for (int i : llvm::seq<int>(1, num_dims - 1)) {
    Value input_size_i = GetDimValue(builder, loc, input_shape_value, i);
    const int stride_i = mlir::cast<IntegerAttr>(strides[i]).getInt();
    const int dilation_i = mlir::cast<IntegerAttr>(dilations[i]).getInt();
    const int filter_i = filter_shape.getDimSize(i - 1);
    Value pad_i_low, pad_i_high;
    GetSamePaddingValues(builder, loc, input_size_i, filter_i, dilation_i,
                         stride_i, pad_i_low, pad_i_high);
    temp_padding_values.push_back(scalar_to_rank1(pad_i_low));
    temp_padding_values.push_back(scalar_to_rank1(pad_i_high));
  }
  temp_padding_values.push_back(zero_rank1);
  temp_padding_values.push_back(zero_rank1);
  padding = CreateConstValue<int32_t>(
      builder, loc, {num_dims - 2, 2},
      SmallVector<int32_t>(2 * (num_dims - 2), 0));
  Value zero = CreateScalarConstValue(builder, loc, 0);
  Value temp_padding_rank1 = builder.create<TF::ConcatOp>(
      loc, RankedTensorType::get({2 * num_dims}, builder.getI32Type()), zero,
      temp_padding_values);
  Value temp_padding = reshape_op(temp_padding_rank1, {num_dims, 2});
  return builder.create<TF::PadV2Op>(
      loc, input.getType(), input, temp_padding,
      CreateScalarConstValue<int8_t>(builder, loc, input_zp_value));
}
}  
Value CalculatePaddingAndPadIfNeeded(OpBuilder &builder, Location loc,
                                     Value input, Value filter,
                                     int8_t input_zp_value, ArrayAttr strides,
                                     ArrayAttr dilations,
                                     StringAttr conv_padding,
                                     ArrayAttr explicit_paddings,
                                     Value &padding, int num_dims) {
  ShapedType input_shape = mlir::cast<ShapedType>(input.getType());
  SmallVector<int64_t> spatial_dims(num_dims - 2);
  absl::c_iota(spatial_dims, 1);
  bool has_dynamic_spatial_dim = absl::c_any_of(
      spatial_dims,
      [&input_shape](int64_t dim) { return input_shape.isDynamicDim(dim); });
  if (conv_padding.strref() == "SAME" && has_dynamic_spatial_dim) {
    return PadForDynamicShapedInputSamePadding(
        builder, loc, input, filter, input_zp_value, strides, dilations,
        conv_padding, padding, num_dims);
  }
  ShapedType filter_shape = mlir::cast<ShapedType>(filter.getType());
  SmallVector<int32_t> padding_values(2 * num_dims, 0);
  if (conv_padding.strref() == "EXPLICIT") {
    if (explicit_paddings.size() != 2 * num_dims) {
      emitError(loc,
                absl::StrFormat(
                    "explicit_paddings are expected to be %d-element arrays",
                    2 * num_dims));
      return {};
    }
    for (int i : spatial_dims) {
      padding_values[2 * i] =
          mlir::cast<IntegerAttr>(explicit_paddings[2 * i]).getInt();
      padding_values[2 * i + 1] =
          mlir::cast<IntegerAttr>(explicit_paddings[2 * i + 1]).getInt();
    }
  } else if (conv_padding.strref() == "SAME") {
    for (int i : spatial_dims) {
      int input_size = input_shape.getDimSize(i);
      int filter_size = filter_shape.getDimSize(i - 1);
      int stride_i = mlir::cast<IntegerAttr>(strides[i]).getInt();
      int dilation_i = mlir::cast<IntegerAttr>(dilations[i]).getInt();
      int out_size = tflite_migration::ComputeOutSize(
          kTfLitePaddingSame, input_size, filter_size, stride_i, dilation_i);
      int offset = 0;
      int padding_before = tflite_migration::ComputePaddingWithOffset(
          stride_i, dilation_i, input_size, filter_size, out_size, &offset);
      int padding_after = padding_before + offset;
      padding_values[2 * i] = padding_before;
      padding_values[2 * i + 1] = padding_after;
    }
  }
  if (input_zp_value == 0 ||
      absl::c_all_of(padding_values, [](int v) { return v == 0; })) {
    padding = CreateConstValue<int32_t>(
        builder, loc, {num_dims - 2, 2},
        SmallVector<int32_t>(padding_values.begin() + 2,
                             padding_values.end() - 2));
    return input;
  }
  padding =
      CreateConstValue<int32_t>(builder, loc, {num_dims - 2, 2},
                                SmallVector<int32_t>(2 * (num_dims - 2), 0));
  Value temp_padding =
      CreateConstValue<int32_t>(builder, loc, {num_dims, 2}, padding_values);
  SmallVector<int64_t> output_shape(input_shape.getShape().begin(),
                                    input_shape.getShape().end());
  for (int i : spatial_dims) {
    output_shape[i] += padding_values[2 * i] + padding_values[2 * i + 1];
  }
  return builder.create<TF::PadV2Op>(
      loc, RankedTensorType::get(output_shape, builder.getI8Type()), input,
      temp_padding,
      CreateScalarConstValue<int8_t>(builder, loc, input_zp_value));
}
Value PackOperand(OpBuilder &builder, Location loc, Value value, int pack_dim) {
  ShapedType value_type = mlir::cast<ShapedType>(value.getType());
  const int rank = value_type.getRank();
  SmallVector<int64_t> packed_shape(value_type.getShape().begin(),
                                    value_type.getShape().end());
  RankedTensorType shape_type =
      RankedTensorType::get({rank}, builder.getI64Type());
  Value shape_value = builder.create<TF::ShapeOp>(loc, shape_type, value);
  if (packed_shape[pack_dim] % 2 != 0) {
    packed_shape[pack_dim] += 1;
    SmallVector<int32_t> padding(rank * 2, 0);
    padding[pack_dim * 2 + 1] = 1;
    Value padding_value =
        CreateConstValue<int32_t>(builder, loc, {rank, 2}, padding);
    value = builder.create<TF::PadV2Op>(
        loc, RankedTensorType::get(packed_shape, builder.getI8Type()), value,
        padding_value, CreateScalarConstValue<int8_t>(builder, loc, 0));
    SmallVector<int64_t> shape_add(rank, 0);
    shape_add[pack_dim] = 1;
    shape_value = builder.create<TF::AddOp>(
        loc, shape_type, shape_value,
        CreateConstValue<int64_t>(builder, loc, {rank}, shape_add));
  }
  packed_shape[pack_dim] /= 2;
  SmallVector<int64_t> divisor(rank, 1);
  divisor[pack_dim] = 2;
  RankedTensorType packed_output_type =
      RankedTensorType::get(packed_shape, builder.getI8Type());
  Value packed_shape_value = builder.create<TF::DivOp>(
      loc, shape_type, shape_value,
      CreateConstValue<int64_t>(builder, loc, {rank}, divisor));
  Value packed_low_begin_value = CreateConstValue<int64_t>(
      builder, loc, {rank}, SmallVector<int64_t>(rank, 0));
  Value packed_low_value =
      builder.create<TF::SliceOp>(loc, packed_output_type, value,
                                  packed_low_begin_value, packed_shape_value);
  packed_low_value = builder.create<TF::BitwiseAndOp>(
      loc, packed_output_type, packed_low_value,
      CreateScalarConstValue<int8_t>(builder, loc, 0x0F));
  SmallVector<int64_t> packed_high_begin(rank, 0);
  packed_high_begin[pack_dim] = packed_shape[pack_dim];
  Value packed_high_begin_value =
      CreateConstValue<int64_t>(builder, loc, {rank}, packed_high_begin);
  Value packed_high_value =
      builder.create<TF::SliceOp>(loc, packed_output_type, value,
                                  packed_high_begin_value, packed_shape_value);
  packed_high_value = builder.create<TF::LeftShiftOp>(
      loc, packed_output_type, packed_high_value,
      CreateScalarConstValue<int8_t>(builder, loc, 4));
  Operation *packed = builder.create<TF::BitwiseOrOp>(
      loc, packed_output_type, packed_low_value, packed_high_value);
  return ConstantFoldOpIfPossible(packed).front();
}
}  