#include "tensorflow/compiler/mlir/lite/utils/perception_ops_utils.h"
#include <string>
#include "flatbuffers/base.h"  
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/IR/Attributes.h"  
#include "mlir/IR/Builders.h"  
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/OpDefinition.h"  
#include "mlir/IR/Types.h"  
#include "mlir/IR/Value.h"  
#include "mlir/Support/LLVM.h"  
#include "mlir/Support/LogicalResult.h"  
#include "tensorflow/compiler/mlir/lite/core/c/builtin_op_data.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
namespace mlir {
namespace TFL {
namespace {
constexpr char kTFImplements[] = "tf._implements";
constexpr char kMaxUnpooling[] = "MaxUnpooling2D";
constexpr char kImageWarping[] = "DenseImageWarp";
inline ConstBytesAttr CustomOption(OpBuilder* builder,
                                   const std::string& content) {
  return ConstBytesAttr::get(builder->getContext(),
                             StringRef(content.data(), content.size()));
}
inline LogicalResult HasIntegerArrayWithSize(func::FuncOp* func,
                                             const DictionaryAttr& attrs,
                                             const std::string& attr_name,
                                             int N) {
  ArrayAttr array_attr =
      mlir::dyn_cast_or_null<ArrayAttr>(attrs.get(attr_name));
  if (array_attr == nullptr || array_attr.size() != N) {
    return func->emitWarning()
           << "'" << attr_name << "' attribute for " << kMaxUnpooling
           << " must be set and has size of " << N;
  }
  for (Attribute integer_attr : array_attr.getValue()) {
    IntegerAttr value = mlir::dyn_cast<IntegerAttr>(integer_attr);
    if (!value) {
      return func->emitWarning()
             << "'" << attr_name << "' attribute for " << kMaxUnpooling
             << " does not contain integer values";
    }
  }
  return success();
}
inline LogicalResult GetIntegerArraySafe(
    func::FuncOp* func, const DictionaryAttr& attrs,
    const std::string& attr_name, llvm::SmallVectorImpl<int32_t>* results,
    int N) {
  ArrayAttr array_attr =
      mlir::dyn_cast_or_null<ArrayAttr>(attrs.get(attr_name));
  if (array_attr == nullptr || array_attr.size() != N) {
    return func->emitError()
           << "'" << attr_name << "' attribute for " << kMaxUnpooling
           << " must be set and has size of " << N;
  }
  results->reserve(N);
  for (Attribute integer_attr : array_attr.getValue()) {
    IntegerAttr value = mlir::dyn_cast<IntegerAttr>(integer_attr);
    if (!value) {
      return func->emitError()
             << "'" << attr_name << "' attribute for " << kMaxUnpooling
             << " does not contain integer values";
    }
    results->push_back(value.getInt());
  }
  return success();
}
}  
LogicalResult ConvertMaxUnpoolingFunc::RewriteFunc() {
  func_.eraseBody();
  func_.addEntryBlock();
  func_->setAttr(kTFImplements,
                 StringAttr::get(func_.getContext(), kMaxUnpooling));
  OpBuilder builder(func_.getBody());
  std::string custom_option_buffer;
  if (failed(CreateCustomOptions(custom_option_buffer))) {
    return failure();
  }
  auto op = builder.create<CustomOp>(
      func_.getLoc(), func_.getFunctionType().getResults(),
      func_.getArguments(), kMaxUnpooling,
      CustomOption(&builder, custom_option_buffer));
  builder.create<func::ReturnOp>(func_.getLoc(), op.getResults());
  return success();
}
LogicalResult ConvertMaxUnpoolingFunc::VerifySignature() {
  if (func_.getNumArguments() != 2) {
    return func_.emitWarning()
           << "Invalid number of arguments to " << kMaxUnpooling << ": "
           << func_.getNumArguments();
  }
  if (func_.getFunctionType().getNumResults() != 1) {
    return func_.emitWarning()
           << "Invalid number of results from " << kMaxUnpooling << ": "
           << func_.getFunctionType().getNumResults();
  }
  auto attrs = attr_.getAttrs();
  if (failed(HasIntegerArrayWithSize(&func_, attrs, "pool_size", 2))) {
    return failure();
  }
  if (failed(HasIntegerArrayWithSize(&func_, attrs, "strides", 2))) {
    return failure();
  }
  auto padding = mlir::dyn_cast_or_null<StringAttr>(attrs.get("padding"));
  if (!padding) {
    return func_.emitWarning() << "'padding' attribute for " << kMaxUnpooling
                               << " is not set or not a string";
  }
  if (padding.getValue() != "VALID" && padding.getValue() != "SAME") {
    return func_.emitWarning()
           << "Padding for " << kMaxUnpooling << " must be 'SAME' or 'VALID'";
  }
  return success();
}
LogicalResult ConvertMaxUnpoolingFunc::CreateCustomOptions(
    std::string& custom_option_buffer) {
  auto attrs = attr_.getAttrs();
  TfLitePoolParams pool_params;
  llvm::SmallVector<int32_t, 2> pool_size;
  if (failed(GetIntegerArraySafe(&func_, attrs, "pool_size", &pool_size, 2))) {
    return failure();
  }
  pool_params.filter_height = pool_size[0];
  pool_params.filter_width = pool_size[1];
  llvm::SmallVector<int32_t, 2> strides;
  if (failed(GetIntegerArraySafe(&func_, attrs, "strides", &strides, 2))) {
    return failure();
  }
  pool_params.stride_height = strides[0];
  pool_params.stride_width = strides[1];
  auto padding = mlir::dyn_cast_or_null<StringAttr>(attrs.get("padding"));
  if (!padding) {
    return func_.emitError() << "'padding' attribute for " << kMaxUnpooling
                             << " is not set or not a string";
  }
  if (padding.getValue() == "VALID") {
    pool_params.padding = kTfLitePaddingValid;
  } else if (padding.getValue() == "SAME") {
    pool_params.padding = kTfLitePaddingSame;
  } else {
    return func_.emitError()
           << "Padding for " << kMaxUnpooling << " must be 'SAME' or 'VALID'";
  }
  pool_params.activation = kTfLiteActNone;
  pool_params.computed.padding = TfLitePaddingValues{0, 0, 0, 0};
#if FLATBUFFERS_LITTLEENDIAN == 0
  int32_t* p = reinterpret_cast<int32_t*>(&pool_params);
  for (size_t i = 0; i < sizeof(TfLitePoolParams) / 4; i++, p++)
    *p = flatbuffers::EndianSwap(*p);
#endif
  custom_option_buffer.assign(reinterpret_cast<char*>(&pool_params),
                              sizeof(TfLitePoolParams));
  return success();
}
LogicalResult ConvertDenseImageWarpFunc::RewriteFunc() {
  func_.eraseBody();
  func_.addEntryBlock();
  func_->setAttr(kTFImplements,
                 StringAttr::get(func_.getContext(), kImageWarping));
  OpBuilder builder(func_.getBody());
  auto op = builder.create<CustomOp>(func_.getLoc(),
                                     func_.getFunctionType().getResults(),
                                     func_.getArguments(), kImageWarping,
                                     CustomOption(&builder, ""));
  builder.create<func::ReturnOp>(func_.getLoc(), op.getResults());
  return success();
}
LogicalResult ConvertDenseImageWarpFunc::VerifySignature() {
  if (func_.getNumArguments() != 2) {
    return func_.emitWarning()
           << "Invalid number of arguments to " << kImageWarping << ": "
           << func_.getNumArguments();
  }
  if (func_.getFunctionType().getNumResults() != 1) {
    return func_.emitWarning()
           << "Invalid number of results from " << kImageWarping << ": "
           << func_.getFunctionType().getNumResults();
  }
  auto image_type = mlir::dyn_cast_or_null<RankedTensorType>(
      func_.getFunctionType().getInput(0));
  if (!image_type || !image_type.getElementType().isF32() ||
      image_type.getRank() != 4) {
    return func_.emitWarning() << "Image should be a 4D float tensor";
  }
  auto flow_type = mlir::dyn_cast_or_null<RankedTensorType>(
      func_.getFunctionType().getInput(1));
  if (!flow_type || !flow_type.getElementType().isF32() ||
      flow_type.getRank() != 4) {
    return func_.emitWarning() << "Flow should be a 4D float tensor";
  }
  auto output_type = mlir::dyn_cast_or_null<RankedTensorType>(
      func_.getFunctionType().getResult(0));
  if (!output_type || !output_type.getElementType().isF32() ||
      output_type.getRank() != 4) {
    return func_.emitWarning() << "Output should be a 4D float tensor";
  }
  return success();
}
}  
}  