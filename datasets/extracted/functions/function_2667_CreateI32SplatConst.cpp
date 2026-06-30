#include "tensorflow/compiler/mlir/lite/utils/lstm_utils.h"
#include <algorithm>
#include <optional>
#include <vector>
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/Dialect/Tensor/IR/Tensor.h"  
#include "mlir/IR/Attributes.h"  
#include "mlir/IR/Builders.h"  
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/BuiltinTypes.h"  
#include "mlir/IR/Location.h"  
#include "mlir/IR/MLIRContext.h"  
#include "mlir/IR/OpDefinition.h"  
#include "mlir/IR/Operation.h"  
#include "mlir/IR/Types.h"  
#include "mlir/IR/Value.h"  
#include "mlir/Support/LLVM.h"  
#include "mlir/Support/LogicalResult.h"  
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/utils/utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
namespace mlir {
namespace TFL {
namespace {
Value CreateI32SplatConst(OpBuilder* builder, ArrayRef<int64_t> shape,
                          int32_t val, mlir::Location location) {
  auto type = RankedTensorType::get(shape, builder->getIntegerType(32));
  auto attr = DenseElementsAttr::get(type, val);
  return builder->create<arith::ConstantOp>(location, type, attr);
}
Value CreateF32SplatConst(OpBuilder* builder, ArrayRef<int64_t> shape,
                          float val, mlir::Location location) {
  auto type = RankedTensorType::get(shape, builder->getF32Type());
  auto attr = DenseElementsAttr::get(type, val);
  return builder->create<arith::ConstantOp>(location, type, attr);
}
Value CreatTfF32ConstOp(OpBuilder* builder, ArrayRef<int64_t> shape, float val,
                        mlir::Location location) {
  auto type = RankedTensorType::get(shape, builder->getF32Type());
  auto ele_type = RankedTensorType::get({1}, builder->getF32Type());
  auto attr = DenseElementsAttr::get(ele_type, val);
  return builder->create<TF::ConstOp>(location, type, attr);
}
Value CreateI64DenseConst(OpBuilder* builder, ArrayRef<int64_t> shape,
                          ArrayRef<int64_t> values, mlir::Location location) {
  auto type = RankedTensorType::get(static_cast<int>(shape.size()),
                                    builder->getIntegerType(64));
  auto attr = DenseElementsAttr::get(type, values);
  return builder->create<arith::ConstantOp>(location, type, attr);
}
Value CreateI32DenseConst(OpBuilder* builder, ArrayRef<int32_t> values,
                          mlir::Location location) {
  auto type = RankedTensorType::get(static_cast<int>(values.size()),
                                    builder->getIntegerType(32));
  auto attr = DenseElementsAttr::get(type, values);
  return builder->create<arith::ConstantOp>(location, type, attr);
}
Value CreateNoneValue(OpBuilder* builder, mlir::Location location) {
  return builder->create<TFL::NoValueOp>(location, builder->getNoneType(),
                                         builder->getUnitAttr());
}
Value Transpose(OpBuilder* builder, Value value_to_transpose,
                SmallVector<int32_t, 4> perm, RankedTensorType original_type,
                mlir::Location location) {
  auto perm_op = CreateI32DenseConst(builder, perm, location);
  auto transpose_type = original_type;
  auto transpose_shape =
      llvm::to_vector<8>(llvm::map_range(perm, [transpose_type](int32_t dim) {
        return transpose_type.getDimSize(dim);
      }));
  auto elem_type = transpose_type.getElementType();
  auto result_type = RankedTensorType::get(transpose_shape, elem_type);
  return builder->create<TF::TransposeOp>(location, result_type,
                                          value_to_transpose, perm_op);
}
Value Transpose2D(OpBuilder* builder, Value value_to_transpose,
                  RankedTensorType type, mlir::Location location) {
  SmallVector<int32_t, 4> perm = {1, 0};
  return Transpose(builder, value_to_transpose, perm, type, location);
}
Value Reverse(OpBuilder* builder, Value value_to_reverse, int axis,
              RankedTensorType type, mlir::Location location) {
  auto axis_op = CreateI32SplatConst(builder, {1}, axis, location);
  return builder->create<TF::ReverseV2Op>(location, type, value_to_reverse,
                                          axis_op);
}
ArrayRef<int64_t> GetRankedTensorShape(Value value) {
  return mlir::cast<RankedTensorType>(value.getType()).getShape();
}
Value SliceRankedTensor(OpBuilder* builder, Value input,
                        ArrayRef<int64_t> begin_shape,
                        ArrayRef<int64_t> begin_values,
                        ArrayRef<int64_t> size_shape,
                        ArrayRef<int64_t> size_values,
                        mlir::Location location) {
  ArrayRef<int64_t> input_shape = GetRankedTensorShape(input);
  for (int i = 0, end = input_shape.size(); i < end; i++) {
    if (begin_values[i] < 0 ||
        (begin_values[i] + size_values[i] > input_shape[i])) {
      return CreateF32SplatConst(builder, size_shape, 0, location);
    }
  }
  auto slice_i2c_begin =
      CreateI64DenseConst(builder, begin_shape, begin_values, location);
  auto slice_i2c_size =
      CreateI64DenseConst(builder, size_shape, size_values, location);
  return builder->create<TF::SliceOp>(
      location,
      RankedTensorType::get(
          size_values,
          mlir::cast<RankedTensorType>(input.getType()).getElementType()),
      input, slice_i2c_begin, slice_i2c_size);
}
Value CreateStridedSliceOp(mlir::Location loc, ArrayRef<int64_t> output_shape,
                           Value input, ArrayRef<int32_t> begin,
                           ArrayRef<int32_t> end, ArrayRef<int32_t> strides,
                           int64_t begin_mask, int64_t end_mask,
                           int64_t ellipsis_mask, int64_t new_axis_mask,
                           int64_t shrink_axis_mask, OpBuilder* builder) {
  auto output_type = RankedTensorType::get(
      output_shape,
      mlir::cast<RankedTensorType>(input.getType()).getElementType());
  auto begin_tensor = CreateI32DenseConst(builder, begin, loc);
  auto end_tensor = CreateI32DenseConst(builder, end, loc);
  auto strides_tensor = CreateI32DenseConst(builder, strides, loc);
  return builder->create<TF::StridedSliceOp>(
      loc, output_type, input, begin_tensor, end_tensor, strides_tensor,
      builder->getI64IntegerAttr(begin_mask),
      builder->getI64IntegerAttr(end_mask),
      builder->getI64IntegerAttr(ellipsis_mask),
      builder->getI64IntegerAttr(new_axis_mask),
      builder->getI64IntegerAttr(shrink_axis_mask));
}
}  
void ConvertLSTMCellSimpleToFusedLSTM::SetWeightForInputToCellGate() {
  SmallVector<int64_t, 2> begin_i2c_values = {0, 0};
  input2cell_ = SliceRankedTensor(
      &builder_, weight_transposed_, weight_slice_shape_, begin_i2c_values,
      weight_slice_shape_, weight_slice_size_input_values_,
      fused_func_op_.getLoc());
}
void ConvertLSTMCellSimpleToFusedLSTM::SetWeightForInputToInputGate() {
  SmallVector<int64_t, 2> begin_i2i_values = {n_cell_, 0};
  input2input_ = couple_input_forget_gates_
                     ? none_
                     : SliceRankedTensor(&builder_, weight_transposed_,
                                         weight_slice_shape_, begin_i2i_values,
                                         weight_slice_shape_,
                                         weight_slice_size_input_values_,
                                         fused_func_op_.getLoc());
}
void ConvertLSTMCellSimpleToFusedLSTM::SetWeightForInputToForgetGate() {
  int input_forget_start = couple_input_forget_gates_ ? n_cell_ : 2 * n_cell_;
  SmallVector<int64_t, 2> begin_i2f_values = {input_forget_start, 0};
  input2forget_ = SliceRankedTensor(
      &builder_, weight_transposed_, weight_slice_shape_, begin_i2f_values,
      weight_slice_shape_, weight_slice_size_input_values_,
      fused_func_op_.getLoc());
}
void ConvertLSTMCellSimpleToFusedLSTM::SetWeightForInputToOutputGate() {
  int input_output_start =
      couple_input_forget_gates_ ? 2 * n_cell_ : 3 * n_cell_;
  SmallVector<int64_t, 2> begin_i2o_values = {input_output_start, 0};
  input2output_ = SliceRankedTensor(
      &builder_, weight_transposed_, weight_slice_shape_, begin_i2o_values,
      weight_slice_shape_, weight_slice_size_input_values_,
      fused_func_op_.getLoc());
}
void ConvertLSTMCellSimpleToFusedLSTM::SetWeightForRecurrentToCellGate() {
  SmallVector<int64_t, 2> begin_rec2c_values = {0, n_input_};
  rec2cell_ = SliceRankedTensor(
      &builder_, weight_transposed_, weight_slice_shape_, begin_rec2c_values,
      weight_slice_shape_, weight_slice_size_recurrent_values_,
      fused_func_op_.getLoc());
}
void ConvertLSTMCellSimpleToFusedLSTM::SetWeightForRecurrentToInputGate() {
  SmallVector<int64_t, 2> begin_rec2i_values = {n_cell_, n_input_};
  rec2input_ = couple_input_forget_gates_
                   ? none_
                   : SliceRankedTensor(&builder_, weight_transposed_,
                                       weight_slice_shape_, begin_rec2i_values,
                                       weight_slice_shape_,
                                       weight_slice_size_recurrent_values_,
                                       fused_func_op_.getLoc());
}
void ConvertLSTMCellSimpleToFusedLSTM::SetWeightForRecurrentToForgetGate() {
  int rec_forget_start = couple_input_forget_gates_ ? n_cell_ : 2 * n_cell_;
  SmallVector<int64_t, 2> begin_rec2f_values = {rec_forget_start, n_input_};
  rec2forget_ = SliceRankedTensor(
      &builder_, weight_transposed_, weight_slice_shape_, begin_rec2f_values,
      weight_slice_shape_, weight_slice_size_recurrent_values_,
      fused_func_op_.getLoc());
}
void ConvertLSTMCellSimpleToFusedLSTM::SetWeightForRecurrentToOutputGate() {
  int rec_output_start = couple_input_forget_gates_ ? 2 * n_cell_ : 3 * n_cell_;
  SmallVector<int64_t, 2> begin_rec2o_values = {rec_output_start, n_input_};
  rec2output_ = SliceRankedTensor(
      &builder_, weight_transposed_, weight_slice_shape_, begin_rec2o_values,
      weight_slice_shape_, weight_slice_size_recurrent_values_,
      fused_func_op_.getLoc());
}
void ConvertLSTMCellSimpleToFusedLSTM::SetBiasToCellGate() {
  SmallVector<int64_t, 1> begin_bias2c_values = {0};
  bias2cell_ = SliceRankedTensor(&builder_, bias_, bias_slice_shape_,
                                 begin_bias2c_values, bias_slice_shape_,
                                 bias_size_values_, fused_func_op_.getLoc());
}
void ConvertLSTMCellSimpleToFusedLSTM::SetBiasToInputGate() {
  SmallVector<int64_t, 1> begin_bias2i_values = {n_cell_};
  bias2input_ =
      couple_input_forget_gates_
          ? none_
          : SliceRankedTensor(&builder_, bias_, bias_slice_shape_,
                              begin_bias2i_values, bias_slice_shape_,
                              bias_size_values_, fused_func_op_.getLoc());
}
void ConvertLSTMCellSimpleToFusedLSTM::SetBiasToForgetGate() {
  int bias_forget_start = couple_input_forget_gates_ ? n_cell_ : 2 * n_cell_;
  SmallVector<int64_t, 1> begin_bias2f_values = {bias_forget_start};
  bias2forget_ = SliceRankedTensor(&builder_, bias_, bias_slice_shape_,
                                   begin_bias2f_values, bias_slice_shape_,
                                   bias_size_values_, fused_func_op_.getLoc());
}
void ConvertLSTMCellSimpleToFusedLSTM::SetBiasToOutputGate() {
  int bias_output_start =
      couple_input_forget_gates_ ? 2 * n_cell_ : 3 * n_cell_;
  SmallVector<int64_t, 1> begin_bias2o_values = {bias_output_start};
  bias2output_ = SliceRankedTensor(&builder_, bias_, bias_slice_shape_,
                                   begin_bias2o_values, bias_slice_shape_,
                                   bias_size_values_, fused_func_op_.getLoc());
}
void ConvertLSTMCellSimpleToFusedLSTM::SetProjection() {
  SmallVector<int64_t, 2> projection_slice_shape = {
      1, num_cols_projection_transposed_};
  SmallVector<int64_t, 2> projection_slice_size_values = {n_output_, n_cell_};
  SmallVector<int64_t, 2> projection_slice_begin_values = {0, 0};
  proj_weight_ =
      !projection_
          ? none_
          : SliceRankedTensor(
                &builder_, projection_transposed_, projection_slice_shape,
                projection_slice_begin_values, projection_slice_shape,
                projection_slice_size_values, fused_func_op_.getLoc());
}
void ConvertLSTMCellSimpleToFusedLSTM::SetProjectionBias() {
  proj_bias_ = !projection_type_
                   ? none_
                   : CreateF32SplatConst(&builder_, {n_output_}, 0,
                                         fused_func_op_.getLoc());
}
void ConvertLSTMCellSimpleToFusedLSTM::SetInputActivationState() {
  input_activation_state_ = CreateF32SplatConst(&builder_, {1, n_output_}, 0,
                                                fused_func_op_.getLoc());
}
void ConvertLSTMCellSimpleToFusedLSTM::SetInputCellState() {
  input_cell_state_ =
      CreateF32SplatConst(&builder_, {1, n_cell_}, 0, fused_func_op_.getLoc());
}
void ConvertLSTMCellSimpleToFusedLSTM::SetCellLayerNormCoefficients() {
  cell_layer_norm_coefficients_ = none_;
}
void ConvertLSTMCellSimpleToFusedLSTM::SetInputLayerNormCoefficients() {
  input_layer_norm_coefficients_ = none_;
}
void ConvertLSTMCellSimpleToFusedLSTM::SetForgetLayerNormCoefficients() {
  forget_layer_norm_coefficients_ = none_;
}
void ConvertLSTMCellSimpleToFusedLSTM::SetOutputLayerNormCoefficients() {
  output_layer_norm_coefficients_ = none_;
}
void ConvertLSTMCellSimpleToFusedLSTM::GenerateFusedOpOperands() {
  weight_transposed_ =
      Transpose2D(&builder_, weight_, weight_type_, fused_func_op_.getLoc());
  projection_transposed_ = Transpose2D(&builder_, projection_, projection_type_,
                                       fused_func_op_.getLoc());
  none_ = CreateNoneValue(&builder_, fused_func_op_.getLoc());
  SetWeightForInputToCellGate();
  SetWeightForInputToInputGate();
  SetWeightForInputToForgetGate();
  SetWeightForInputToOutputGate();
  SetWeightForRecurrentToCellGate();
  SetWeightForRecurrentToInputGate();
  SetWeightForRecurrentToForgetGate();
  SetWeightForRecurrentToOutputGate();
  SetBiasToCellGate();
  SetBiasToInputGate();
  SetBiasToForgetGate();
  SetBiasToOutputGate();
  SetProjection();
  SetProjectionBias();
  SetInputActivationState();
  SetInputCellState();
  SetCellLayerNormCoefficients();
  SetInputLayerNormCoefficients();
  SetForgetLayerNormCoefficients();
  SetOutputLayerNormCoefficients();
}
void ConvertLSTMCellSimpleToFusedLSTM::UpdateFuncSignature() {
  SmallVector<int64_t, 2> output_shape{1, tensorflow::kTFDynamicSize};
  auto input_types = fused_func_op_.getFunctionType().getInputs();
  auto output_type = tensorflow::GetTypeFromTFTensorShape(
      output_shape,
      mlir::cast<RankedTensorType>(input_.getType()).getElementType());
  fused_func_op_.setType(mlir::FunctionType::get(fused_func_op_.getContext(),
                                                 input_types, output_type));
}
LogicalResult ConvertLSTMCellSimpleToFusedLSTM::RewriteFunc() {
  LogicalResult result = Initialize();
  if (failed(result)) {
    return result;
  }
  UpdateFuncSignature();
  GenerateFusedOpOperands();
  SmallVector<int64_t, 2> output_shape = {1, n_output_};
  auto result_type = mlir::RankedTensorType::get(
      output_shape,
      mlir::cast<RankedTensorType>(input_.getType()).getElementType());
  lstm_ = builder_.create<mlir::TFL::LSTMOp>(
      fused_func_op_.getLoc(), result_type, input_, input2input_, input2forget_,
      input2cell_, input2output_, rec2input_, rec2forget_, rec2cell_,
      rec2output_,  none_,
       none_,
       none_, bias2input_, bias2forget_, bias2cell_,
      bias2output_, proj_weight_, proj_bias_, input_activation_state_,
      input_cell_state_, input_layer_norm_coefficients_,
      forget_layer_norm_coefficients_, cell_layer_norm_coefficients_,
      output_layer_norm_coefficients_, builder_.getStringAttr("TANH"),
      builder_.getF32FloatAttr(10.0), builder_.getF32FloatAttr(0.0),
      mlir::TFL::LSTMKernelTypeAttr::get(builder_.getContext(),
                                         mlir::TFL::LSTMKernelType::FULL),
      mlir::BoolAttr(),
      mlir::TypeAttr(),
      mlir::TypeAttr(),
      mlir::TypeAttr(),
      mlir::TypeAttr(),
      mlir::TypeAttr());
  SmallVector<int64_t, 2> func_output_shape = {1, tensorflow::kTFDynamicSize};
  auto func_result_type = tensorflow::GetTypeFromTFTensorShape(
      func_output_shape,
      mlir::cast<RankedTensorType>(input_.getType()).getElementType());
  auto tensor_cast = builder_.create<mlir::tensor::CastOp>(
      fused_func_op_.getLoc(), func_result_type, lstm_.getResult());
  builder_.create<mlir::func::ReturnOp>(fused_func_op_.getLoc(),
                                        tensor_cast.getResult());
  return success();
}
LogicalResult ConvertLSTMCellSimpleToFusedLSTM::InitializeFromFuncAttributes() {
  auto attr = fused_func_op_->getAttrOfType<StringAttr>(kTFImplements);
  if (!attr) {
    return fused_func_op_.emitError()
           << "Invalid function attribute, expected " << kTFImplements
           << " attribute "
              "not found";
  }
  llvm::SmallVector<llvm::StringRef, 4> attr_tokens;
  attr.getValue().split(attr_tokens, ",");
  if (attr_tokens.empty()) {
    return fused_func_op_.emitError()
           << kTFImplements << " attribute should be set";
  }
  if (GetCompositeOpName().str() != attr_tokens[0]) {
    return fused_func_op_.emitError()
           << "Unexpected interface for the composite op. Expected: "
           << GetCompositeOpName() << " Actual: " << attr_tokens[0];
  }
  couple_input_forget_gates_ =
      std::find(attr_tokens.begin() + 1, attr_tokens.end(),
                kCoupleInputForgetGates) != attr_tokens.end();
  return success();
}
LogicalResult ConvertLSTMCellSimpleToFusedLSTM::Initialize() {
  if (failed(InitializeFromFuncAttributes())) {
    return fused_func_op_.emitError()
           << "Expected function attributes were not set on the function "
              "encapsulating the composite op";
  }
  num_gates_ = couple_input_forget_gates_ ? 3 : 4;
  input_ = fused_func_op_.getArgument(0);
  bias_ = fused_func_op_.getArgument(2);
  weight_ = fused_func_op_.getArgument(1);
  weight_type_ = mlir::cast<RankedTensorType>(weight_.getType());
  if (weight_type_.getRank() != 2) {
    return fused_func_op_.emitError() << "The weight tensor was not of rank 2";
  }
  if (weight_type_.getDimSize(1) % num_gates_ != 0) {
    return fused_func_op_.emitError()
           << "Invalid dimension 1 of weight tensor, "
              "should be divisible by the number of gates";
  }
  n_cell_ = weight_type_.getDimSize(1) / num_gates_;
  projection_ = fused_func_op_.getArgument(3);
  projection_type_ = mlir::cast<RankedTensorType>(projection_.getType());
  if (projection_type_.getRank() != 2) {
    n_output_ = n_cell_;
  } else {
    n_output_ = projection_type_.getDimSize(1);
  }
  n_input_ = weight_type_.getDimSize(0) - n_output_;
  num_cols_weight_transposed_ = weight_type_.getDimSize(0);
  num_cols_projection_transposed_ = projection_type_.getDimSize(0);
  bias_slice_shape_ = {n_cell_};
  bias_size_values_ = {n_cell_};
  weight_slice_shape_ = {1, num_cols_weight_transposed_};
  weight_slice_size_input_values_ = {n_cell_, n_input_};
  weight_slice_size_recurrent_values_ = {n_cell_, n_output_};
  return success();
}
LogicalResult ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM::Initialize() {
  if (failed(ConvertLSTMCellSimpleToFusedLSTM::Initialize())) {
    return fused_func_op_.emitError()
           << "Specified LayerNormalizedLSTMCellSimple was not of the expected "
              "interface and cannot not be converted to the fused LSTM op";
  }
  layer_norm_scale_ = fused_func_op_.getArgument(4);
  layer_norm_scale_type_ =
      mlir::cast<RankedTensorType>(layer_norm_scale_.getType());
  if (layer_norm_scale_type_.getRank() != 1) {
    return fused_func_op_.emitError()
           << "The layer_norm_scale tensor was not of rank 1";
  }
  layer_norm_slice_shape_ = {n_cell_};
  layer_norm_size_values_ = {n_cell_};
  return success();
}
void ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM::
    SetCellLayerNormCoefficients() {
  SmallVector<int64_t, 1> begin_cell_layer_norm_values = {0};
  cell_layer_norm_coefficients_ =
      SliceRankedTensor(&builder_, layer_norm_scale_, layer_norm_slice_shape_,
                        begin_cell_layer_norm_values, layer_norm_slice_shape_,
                        layer_norm_size_values_, fused_func_op_.getLoc());
}
void ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM::
    SetInputLayerNormCoefficients() {
  SmallVector<int64_t, 1> begin_input_layer_norm_values = {n_cell_};
  input_layer_norm_coefficients_ =
      couple_input_forget_gates_
          ? none_
          : SliceRankedTensor(
                &builder_, layer_norm_scale_, layer_norm_slice_shape_,
                begin_input_layer_norm_values, layer_norm_slice_shape_,
                layer_norm_size_values_, fused_func_op_.getLoc());
}
void ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM::
    SetForgetLayerNormCoefficients() {
  SmallVector<int64_t, 1> begin_forget_layer_norm_values = {2 * n_cell_};
  forget_layer_norm_coefficients_ =
      SliceRankedTensor(&builder_, layer_norm_scale_, layer_norm_slice_shape_,
                        begin_forget_layer_norm_values, layer_norm_slice_shape_,
                        layer_norm_size_values_, fused_func_op_.getLoc());
}
void ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM::
    SetOutputLayerNormCoefficients() {
  SmallVector<int64_t, 1> begin_output_layer_norm_values = {3 * n_cell_};
  output_layer_norm_coefficients_ =
      SliceRankedTensor(&builder_, layer_norm_scale_, layer_norm_slice_shape_,
                        begin_output_layer_norm_values, layer_norm_slice_shape_,
                        layer_norm_size_values_, fused_func_op_.getLoc());
}
TF::ConstOp Create1DConstantOp(const std::vector<int>& value, Location loc,
                               OpBuilder* builder) {
  auto type =
      mlir::RankedTensorType::get(value.size(), builder->getIntegerType(32));
  auto dense_values = mlir::DenseIntElementsAttr::get(type, value);
  return builder->create<TF::ConstOp>(loc, dense_values);
}
TF::ConstOp CreateScalarConstantOp(int value, Location loc,
                                   OpBuilder* builder) {
  return builder->create<TF::ConstOp>(loc, builder->getI32IntegerAttr(value));
}
TF::ReshapeOp CreateFlattenOP(const Value& input, Location loc,
                              OpBuilder* builder) {
  auto output_shape = Create1DConstantOp({-1}, loc, builder);
  return builder->create<mlir::TF::ReshapeOp>(
      loc,
      input,
      output_shape.getResult());
}
LogicalResult CreateEqualSizeSplitVOp(Value input, int axis, int splits,
                                      Location loc, OpBuilder* builder,
                                      Operation** result) {
  auto input_type = mlir::cast<RankedTensorType>(input.getType());
  SmallVector<int64_t, 4> output_shape;
  int size_of_splits;
  if (input_type.getRank() < axis || axis < 0) return failure();
  for (int i = 0; i < input_type.getRank(); ++i) {
    int64_t dim = input_type.getDimSize(i);
    if (i == axis) {
      if (dim % splits != 0) {
        return failure();
      }
      size_of_splits = dim / splits;
      output_shape.push_back(size_of_splits);
    } else {
      output_shape.push_back(dim);
    }
  }
  SmallVector<mlir::Type, 4> output_types;
  for (int i = 0; i < splits; ++i) {
    output_types.push_back(
        mlir::RankedTensorType::get(output_shape, input_type.getElementType()));
  }
  auto size_of_splits_op = Create1DConstantOp(
      {size_of_splits, size_of_splits, size_of_splits, size_of_splits}, loc,
      builder);
  auto axis_op = CreateScalarConstantOp(axis, loc, builder);
  *result = builder->create<TF::SplitVOp>(loc, output_types, input,
                                          size_of_splits_op.getResult(),
                                          axis_op.getResult());
  return success();
}
LogicalResult ConvertKerasLSTMLayer(mlir::func::FuncOp func_op,
                                    OpBuilder* builder) {
  return ConvertKerasLSTMLayer(func_op, builder, false);
}
LogicalResult ConvertKerasLSTMLayer(mlir::func::FuncOp func_op,
                                    OpBuilder* builder, bool indy) {
  Value input = func_op.getArgument(0);
  Value output_init_state = func_op.getArgument(1);
  Value hidden_init_state = func_op.getArgument(2);
  Value weight_kernel = func_op.getArgument(3);
  Value recurrent_kernel = func_op.getArgument(4);
  Value bias = func_op.getArgument(5);
  if (func_op.getNumResults() != 5) return failure();
  auto time_major_attr = func_op->getAttrOfType<BoolAttr>("tf.time_major");
  if (time_major_attr == nullptr) return failure();
  bool time_majored = time_major_attr.getValue();
  auto input_type = mlir::dyn_cast_or_null<RankedTensorType>(input.getType());
  if (!input_type) {
    func_op.emitError() << "Input type is not a ranked tensor type";
    return failure();
  }
  auto final_inputs = input;
  auto final_input_type = input_type;
  auto go_backwards_attr = func_op->getAttrOfType<BoolAttr>("tf.go_backwards");
  if (go_backwards_attr != nullptr && go_backwards_attr.getValue()) {
    int time_dim = time_majored ? 0 : 1;
    final_inputs = Reverse(builder, final_inputs, time_dim, final_input_type,
                           func_op.getLoc());
  }
  int64_t batch = time_majored ? final_input_type.getDimSize(1)
                               : final_input_type.getDimSize(0);
  int64_t time = time_majored ? final_input_type.getDimSize(0)
                              : final_input_type.getDimSize(1);
  RankedTensorType weight_type =
      mlir::cast<RankedTensorType>(weight_kernel.getType());
  if (weight_type.getRank() != 2)
    return func_op.emitError() << "The weight should be rank of 2";
  Value transposed_weight_kernel =
      Transpose2D(builder, weight_kernel, weight_type, func_op.getLoc());
  RankedTensorType recurrent_kernel_type =
      mlir::cast<RankedTensorType>(recurrent_kernel.getType());
  const int64_t n_output = recurrent_kernel_type.getDimSize(0);
  Value transpose_recurrent_kernel = Transpose2D(
      builder, recurrent_kernel, recurrent_kernel_type, func_op.getLoc());
  const int splits = 4;
  Operation* weights_array;
  if (failed(CreateEqualSizeSplitVOp(transposed_weight_kernel, 0, splits,
                                     func_op.getLoc(), builder,
                                     &weights_array)))
    return failure();
  Operation* recurrent_weights_array;
  if (failed(CreateEqualSizeSplitVOp(transpose_recurrent_kernel, 0, splits,
                                     func_op.getLoc(), builder,
                                     &recurrent_weights_array)))
    return failure();
  Value recurrent_to_input_weights =
      indy ? mlir::cast<Value>(
                 CreateFlattenOP(recurrent_weights_array->getResult(0),
                                 func_op.getLoc(), builder)
                     .getResult())
           : recurrent_weights_array->getResult(0);
  Value recurrent_to_forget_weights =
      indy ? mlir::cast<Value>(
                 CreateFlattenOP(recurrent_weights_array->getResult(1),
                                 func_op.getLoc(), builder)
                     .getResult())
           : recurrent_weights_array->getResult(1);
  Value recurrent_to_cell_weights =
      indy ? mlir::cast<Value>(
                 CreateFlattenOP(recurrent_weights_array->getResult(2),
                                 func_op.getLoc(), builder)
                     .getResult())
           : recurrent_weights_array->getResult(2);
  Value recurrent_to_output_weights =
      indy ? mlir::cast<Value>(
                 CreateFlattenOP(recurrent_weights_array->getResult(3),
                                 func_op.getLoc(), builder)
                     .getResult())
           : recurrent_weights_array->getResult(3);
  Operation* bias_array;
  if (failed(CreateEqualSizeSplitVOp(bias, 0, splits, func_op.getLoc(), builder,
                                     &bias_array)))
    return failure();
  SmallVector<int64_t, 3> output_shape;
  if (time_majored) {
    output_shape = {time, batch, n_output};
  } else {
    output_shape = {batch, time, n_output};
  }
  auto result_type = mlir::RankedTensorType::get(
      output_shape,
      mlir::cast<RankedTensorType>(final_inputs.getType()).getElementType());
  Value none = CreateNoneValue(builder, func_op.getLoc());
  auto lstm = builder->create<mlir::TFL::UnidirectionalSequenceLSTMOp>(
      func_op.getLoc(), result_type, final_inputs,
      weights_array->getResult(0),
      weights_array->getResult(1),
      weights_array->getResult(2),
      weights_array->getResult(3),
      recurrent_to_input_weights,
      recurrent_to_forget_weights,
      recurrent_to_cell_weights,
      recurrent_to_output_weights,
      none,
      none,
      none,
      bias_array->getResult(0),
      bias_array->getResult(1),
      bias_array->getResult(2),
      bias_array->getResult(3),
      none,
      none,
      output_init_state,
      hidden_init_state,
      none,
      none,
      none,
      none,
       builder->getStringAttr("TANH"),
       builder->getF32FloatAttr(10.0),
       builder->getF32FloatAttr(0.0),
       builder->getBoolAttr(time_majored),
      mlir::BoolAttr(),
      builder->getBoolAttr(indy),
      mlir::TypeAttr(),
      mlir::TypeAttr(),
      mlir::TypeAttr(),
      mlir::TypeAttr(),
      mlir::TypeAttr());
  auto final_output_full_sequences = lstm.getResult();
  SmallVector<int64_t, 2> last_output_shape({batch, n_output});
  SmallVector<int32_t, 3> end({0, 0, 0});
  SmallVector<int32_t, 3> strides({1, 1, 1});
  SmallVector<int32_t, 3> begin;
  int64_t new_axis_mask = 0;
  int64_t ellipsis_mask = 0;
  int64_t begin_mask;
  int64_t end_mask;
  int64_t shrink_axis_mask;
  if (time_majored) {
    begin_mask = 6;
    end_mask = 6;
    shrink_axis_mask = 1;
    begin = {-1, 0, 0};
  } else {
    begin_mask = 5;
    end_mask = 5;
    shrink_axis_mask = 2;
    begin = {0, -1, 0};
  }
  auto last_output = CreateStridedSliceOp(
      func_op.getLoc(), last_output_shape, final_output_full_sequences, begin,
      end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
      shrink_axis_mask, builder);
  SmallVector<Value, 5> outputs;
  SmallVector<Type, 5> output_types;
  outputs.push_back(last_output);
  output_types.push_back(last_output.getType());
  outputs.push_back(final_output_full_sequences);
  output_types.push_back(final_output_full_sequences.getType());
  for (int i = 2; i < 5; ++i) {
    auto result_type =
        mlir::dyn_cast<RankedTensorType>(func_op.getResultTypes()[i]);
    outputs.push_back(CreatTfF32ConstOp(builder, result_type.getShape(), 0.0f,
                                        func_op.getLoc()));
    output_types.push_back(result_type);
  }
  func_op.setType(mlir::FunctionType::get(func_op.getContext(),
                                          func_op.getFunctionType().getInputs(),
                                          output_types));
  builder->create<mlir::func::ReturnOp>(func_op.getLoc(), outputs);
  return success();
}
}  
}  