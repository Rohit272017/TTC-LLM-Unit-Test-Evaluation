#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/tf_quantize_op.h"
#include <functional>
#include <optional>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  
#include "mlir/IR/Block.h"  
#include "mlir/IR/Builders.h"  
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/BuiltinOps.h"  
#include "mlir/IR/BuiltinTypeInterfaces.h"  
#include "mlir/IR/BuiltinTypes.h"  
#include "mlir/IR/Diagnostics.h"  
#include "mlir/IR/Matchers.h"  
#include "mlir/IR/Operation.h"  
#include "mlir/IR/PatternMatch.h"  
#include "mlir/IR/SymbolTable.h"  
#include "mlir/IR/Types.h"  
#include "mlir/Support/LLVM.h"  
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/utils/tf_quantize_op_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
namespace mlir {
namespace quant {
namespace {
constexpr StringRef kDequantizeFunctionName = "composite_dequantize";
constexpr StringRef kUniformQuantizationFunctionName = "uniform";
func::FuncOp PrepareFunctionRegister(PatternRewriter& rewriter, Value input_val,
                                     ShapedType result_type,
                                     StringRef func_name,
                                     Value& func_input_arg) {
  Operation* input_op = input_val.getDefiningOp();
  Operation* insertion_point = input_op->getParentOfType<func::FuncOp>();
  if (!insertion_point) insertion_point = input_op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPointAfter(insertion_point);
  UnrankedTensorType create_unknown_input_shape =
      CreateUnknownShapeFromElementType(input_val.getType());
  UnrankedTensorType create_unknown_output_shape =
      CreateUnknownShapeFromElementType(result_type);
  FunctionType func_type =
      FunctionType::get(rewriter.getContext(), {create_unknown_input_shape},
                        {create_unknown_output_shape});
  func::FuncOp quantization_func =
      rewriter.create<func::FuncOp>(input_op->getLoc(), func_name, func_type);
  OpBuilder::InsertionGuard guard = OpBuilder::InsertionGuard(rewriter);
  ArrayRef<Type> inputs = quantization_func.getFunctionType().getInputs();
  Block* block = rewriter.createBlock(
      &quantization_func.getBody(), quantization_func.begin(), inputs,
      SmallVector<Location>(inputs.size(), quantization_func.getLoc()));
  func_input_arg = block->getArgument(0);
  return quantization_func;
}
TF::PartitionedCallOp FinalizeFunctionRegister(
    PatternRewriter& rewriter, Value input, Value output,
    func::FuncOp& quantization_func, Operation* quantized_op,
    StringRef func_name, IRRewriter::InsertPoint original_point,
    Type quantize_result_type) {
  rewriter.create<func::ReturnOp>(input.getLoc(), ArrayRef<Value>({output}));
  quantization_func.setVisibility(func::FuncOp::Visibility::Private);
  SymbolTable symbol_table(quantized_op->getParentOfType<ModuleOp>());
  symbol_table.insert(quantization_func);
  FlatSymbolRefAttr func_name_attr =
      FlatSymbolRefAttr::get(rewriter.getStringAttr(func_name));
  rewriter.restoreInsertionPoint(original_point);
  auto quantize_call = rewriter.create<TF::PartitionedCallOp>(
      quantized_op->getLoc(), quantize_result_type, input, func_name_attr,
      "", "", "");
  return quantize_call;
}
std::optional<TF::PartitionedCallOp> RegisterOperationsInFuncOp(
    StringRef func_name, PatternRewriter& rewriter, QuantizedType quant_type,
    Value input_val, ShapedType result_type,
    std::function<Operation*(PatternRewriter&, Operation*, Value, ShapedType,
                             QuantizedType)>
        quantization_operations_func) {
  Operation* input_op = input_val.getDefiningOp();
  auto original_point = rewriter.saveInsertionPoint();
  auto unique_func_name = func_name.str();
  SymbolTable symbol_table(input_op->getParentOfType<ModuleOp>());
  while (symbol_table.lookup(unique_func_name)) {
    absl::StrAppend(&unique_func_name, "_");
  }
  Value func_input_arg;
  func::FuncOp func_op = PrepareFunctionRegister(
      rewriter, input_val, result_type, unique_func_name, func_input_arg);
  Operation* last_op_in_func =
      quantization_operations_func(rewriter, func_op.getOperation(),
                                   func_input_arg, result_type, quant_type);
  auto end_call_op = FinalizeFunctionRegister(
      rewriter, input_val, last_op_in_func->getResult(0), func_op, input_op,
      unique_func_name, original_point, result_type);
  return end_call_op;
}
QuantizedType CalculateUniformQuantParams(
    PatternRewriter& rewriter, TF::ConstOp op,
    tensorflow::quantization::QuantizationComponentSpec& weight_spec) {
  const bool kIsNarrowRange = true;
  const bool kIsSigned = true;
  const int kBitWidth = 8;
  DenseFPElementsAttr attr;
  if (!matchPattern(op->getResult(0), m_Constant(&attr))) return nullptr;
  QuantizedType quant_type = mlir::dyn_cast<quant::QuantizedType>(
      quant::GetUniformQuantizedTypeForWeight(
          attr, kIsNarrowRange && kIsSigned, kBitWidth, kIsSigned,
          kIsNarrowRange,  false));
  return quant_type;
}
std::optional<Value> AddUniformQuantizeOps(PatternRewriter& rewriter,
                                           TF::ConstOp op,
                                           QuantizedType quant_type) {
  DenseFPElementsAttr attr;
  if (!matchPattern(op->getResult(0), m_Constant(&attr))) {
    return nullptr;
  }
  Type expressed_type = op.getResult().getType();
  Type quantized_type = quant_type.castFromExpressedType(expressed_type);
  ShapedType shaped_quantized_type = mlir::cast<ShapedType>(quantized_type);
  DenseElementsAttr tensor_proto_attr =
      mlir::dyn_cast<DenseElementsAttr>(Quantize(attr, shaped_quantized_type));
  if (!tensor_proto_attr) {
    return nullptr;
  }
  Type storage_type =
      mlir::cast<QuantizedType>(shaped_quantized_type.getElementType())
          .getStorageType();
  ShapedType new_type = shaped_quantized_type.clone(storage_type);
  rewriter.setInsertionPointAfter(op);
  auto const_op =
      rewriter.create<TF::ConstOp>(op.getLoc(), new_type, tensor_proto_attr);
  auto new_identity_op = rewriter.create<TF::IdentityOp>(
      op->getLoc(), const_op.getType(), const_op);
  return new_identity_op.getResult();
}
Operation* LogicsForUniformDequanization(PatternRewriter& rewriter,
                                         Operation* func_op, Value input_val,
                                         ShapedType original_input_tensor_type,
                                         QuantizedType quant_type) {
  auto loc = input_val.getLoc();
  rewriter.setInsertionPointToStart(
      &(cast<func::FuncOp>(func_op)).getBody().front());
  UnrankedTensorType create_unknown_input_shape =
      CreateUnknownShapeFromElementType(original_input_tensor_type);
  auto new_cast_op =
      rewriter.create<TF::CastOp>(loc, create_unknown_input_shape, input_val);
  auto qtype = mlir::dyn_cast<UniformQuantizedType>(quant_type);
  TensorType scale_type = RankedTensorType::get({}, rewriter.getF32Type());
  Value scale_op = rewriter.create<TF::ConstOp>(
      loc, scale_type,
      DenseFPElementsAttr::get(scale_type,
                               {static_cast<float>(qtype.getScale())}));
  if (original_input_tensor_type.getElementType().isBF16()) {
    scale_op = rewriter.create<TF::CastOp>(
        loc, UnrankedTensorType::get(rewriter.getBF16Type()), scale_op);
  }
  auto mul_op = rewriter.create<TF::MulOp>(loc, new_cast_op.getType(), scale_op,
                                           new_cast_op);
  return mul_op;
}
std::optional<TF::PartitionedCallOp> AddUniformDequantizeOps(
    PatternRewriter& rewriter, QuantizedType quant_type,
    Value val_to_dequantize, ShapedType result_type) {
  auto func_name = absl::StrJoin(
      {kDequantizeFunctionName, kUniformQuantizationFunctionName}, "_");
  std::optional<TF::PartitionedCallOp> dequant_op = RegisterOperationsInFuncOp(
      func_name, rewriter, quant_type, val_to_dequantize, result_type,
      LogicsForUniformDequanization);
  return dequant_op;
}
}  
std::optional<TF::PartitionedCallOp> ApplyUniformQuantization(
    PatternRewriter& rewriter, TF::ConstOp op,
    tensorflow::quantization::QuantizationComponentSpec& weight_spec) {
  QuantizedType quant_type =
      CalculateUniformQuantParams(rewriter, op, weight_spec);
  if (!quant_type) return nullptr;
  std::optional<Value> quantized_val =
      AddUniformQuantizeOps(rewriter, op, quant_type);
  if (!quantized_val.has_value()) return std::nullopt;
  std::optional<TF::PartitionedCallOp> dequantized_val =
      AddUniformDequantizeOps(rewriter, quant_type, quantized_val.value(),
                              mlir::cast<ShapedType>(op.getType()));
  return dequantized_val;
}
}  
}  