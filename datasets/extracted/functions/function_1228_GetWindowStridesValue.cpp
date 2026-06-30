#include "tensorflow/compiler/mlir/quantization/tensorflow/utils/tf_to_uniform_attribute_utils.h"
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/IR/Attributes.h"  
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/BuiltinTypeInterfaces.h"  
#include "mlir/IR/OperationSupport.h"  
#include "mlir/IR/PatternMatch.h"  
#include "mlir/IR/TypeUtilities.h"  
#include "mlir/Support/LLVM.h"  
#include "mlir/Support/LogicalResult.h"  
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/uniform_op_quant_spec.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/core/util/quantization/uniform_quant_ops_attr.pb.h"
namespace mlir::quant {
using QuantMethod = tensorflow::quantization::QuantizationMethod::PresetMethod;
enum class OpType {
  kDynamicRangeOp,  
  kUnaryOp,         
  kBinaryOp,        
  kQuantizationOp,  
};
constexpr std::array<absl::string_view, 3> kQuantizationAxisAttrs = {
    "input_quantization_axis", "quantization_axis", "rhs_quantization_axis"};
constexpr std::array<absl::string_view, 2> kSuffixes = {"_min_val", "_max_val"};
Attribute GetWindowStridesValue(
    PatternRewriter& rewriter, llvm::StringMap<Attribute>& identifier_to_attr) {
  ArrayAttr stride = mlir::dyn_cast<ArrayAttr>(identifier_to_attr["strides"]);
  const int stride_h = mlir::cast<IntegerAttr>(stride[1]).getInt();
  const int stride_w = mlir::cast<IntegerAttr>(stride[2]).getInt();
  return rewriter.getI64ArrayAttr({stride_h, stride_w});
}
Attribute GetLhsDilationValue(PatternRewriter& rewriter,
                              llvm::StringMap<Attribute>& identifier_to_attr) {
  return rewriter.getI64ArrayAttr({1, 1});
}
Attribute GetRhsDilationValue(PatternRewriter& rewriter,
                              llvm::StringMap<Attribute>& identifier_to_attr) {
  ArrayAttr dilations =
      mlir::dyn_cast<ArrayAttr>(identifier_to_attr["dilations"]);
  const int dilation_h = mlir::cast<IntegerAttr>(dilations[1]).getInt();
  const int dilation_w = mlir::cast<IntegerAttr>(dilations[2]).getInt();
  return rewriter.getI64ArrayAttr({dilation_h, dilation_w});
}
Attribute GetPaddingValue(PatternRewriter& rewriter,
                          llvm::StringMap<Attribute>& identifier_to_attr) {
  llvm::StringRef padding =
      mlir::dyn_cast<StringAttr>(identifier_to_attr["padding"]).getValue();
  return rewriter.getStringAttr(padding);
}
Attribute GetExplicitPaddingValue(
    PatternRewriter& rewriter, llvm::StringMap<Attribute>& identifier_to_attr) {
  ArrayAttr explicit_padding =
      mlir::dyn_cast<ArrayAttr>(identifier_to_attr["explicit_paddings"]);
  return explicit_padding;
}
Attribute GetDimensionNumbersValue(
    PatternRewriter& rewriter, llvm::StringMap<Attribute>& identifier_to_attr) {
  tensorflow::UniformQuantizedConvolutionDimensionNumbersAttr dimension_numbers;
  if (!tensorflow::protobuf::TextFormat::ParseFromString(
          R"pb(
            input_batch_dimension: 0
            input_feature_dimension: 3
            input_spatial_dimensions: 1
            input_spatial_dimensions: 2
            kernel_output_feature_dimension: 3
            kernel_input_feature_dimension: 2
            kernel_spatial_dimensions: 0
            kernel_spatial_dimensions: 1
            output_batch_dimension: 0
            output_feature_dimension: 3
            output_spatial_dimensions: 1
            output_spatial_dimensions: 2
          )pb",
          &dimension_numbers)) {
    return rewriter.getStringAttr("");
  }
  return rewriter.getStringAttr(dimension_numbers.SerializeAsString());
}
Attribute GetBatchGroupCountValue(
    PatternRewriter& rewriter, llvm::StringMap<Attribute>& identifier_to_attr) {
  return rewriter.getI64IntegerAttr(1);
}
Attribute GetQuantizationAxis(PatternRewriter& rewriter, Operation* op,
                              const int operand_index) {
  auto* defining_op = op->getOperand(operand_index).getDefiningOp();
  for (auto attr : kQuantizationAxisAttrs) {
    if (defining_op->hasAttr(attr)) {
      return defining_op->getAttr(attr);
    }
  }
  return rewriter.getI64IntegerAttr(-1);
}
LogicalResult CheckIfAttrIs8Bit(const std::string& attr, Operation* op,
                                bool& is_8_bit) {
  Type element_type;
  if (attr == "lhs_quantization" || attr == "input_quantization" ||
      attr == "quantization") {
    if (op->getNumOperands() < 1) {
      return failure();
    }
    element_type = getElementTypeOrSelf(op->getOperand(0).getType());
  }
  if (attr == "rhs_quantization") {
    if (op->getNumOperands() < 2) {
      return failure();
    }
    element_type = getElementTypeOrSelf(op->getOperand(1).getType());
  }
  if (attr == "output_quantization") {
    if (op->getNumResults() < 1) {
      return failure();
    }
    element_type = getElementTypeOrSelf(op->getOpResult(0).getType());
  }
  if (element_type) {
    is_8_bit = mlir::isa<TF::Qint8Type>(element_type);
    return success();
  }
  return failure();
}
LogicalResult FillQuantizationAttributes(
    PatternRewriter& rewriter, Operation* op, NamedAttrList& attrs,
    llvm::StringMap<Attribute>& identifier_to_attr, OpType op_type) {
  absl::flat_hash_map<std::string, int> min_max_scheme_for_8bit = {
      {"min", -128}, {"max", 127}};
  absl::flat_hash_map<std::string, int> min_max_schema_for_32bit = {
      {"min", -2147483648}, {"max", 2147483647}};
  std::vector<std::string> quantization_attributes;
  switch (op_type) {
    case OpType::kDynamicRangeOp:
      quantization_attributes = {"rhs_quantization"};
      break;
    case OpType::kUnaryOp:
      quantization_attributes = {"quantization"};
      break;
    case OpType::kBinaryOp:
      quantization_attributes = {"lhs_quantization", "rhs_quantization",
                                 "output_quantization"};
      break;
    case OpType::kQuantizationOp:
      quantization_attributes = {"input_quantization", "output_quantization"};
      break;
    default:
      quantization_attributes = {};
      break;
  }
  for (const auto& attr : quantization_attributes) {
    bool attr_is_8_bit;
    if (failed(CheckIfAttrIs8Bit(attr, op, attr_is_8_bit))) {
      return failure();
    }
    for (int i = 0; i < kSuffixes.size(); i++) {
      int64_t quant_val;
      if (attr_is_8_bit) {
        quant_val = i == 0 ? min_max_scheme_for_8bit["min"]
                           : min_max_scheme_for_8bit["max"];
      } else {
        quant_val = i == 0 ? min_max_schema_for_32bit["min"]
                           : min_max_schema_for_32bit["max"];
      }
      std::string attr_minmax = absl::StrCat(attr, kSuffixes[i]);
      attrs.push_back(rewriter.getNamedAttr(
          attr_minmax, rewriter.getI64IntegerAttr(quant_val)));
    }
  }
  return success();
}
LogicalResult FillAttributesForUniformQuantizedDotOp(
    PatternRewriter& rewriter, Operation* op,
    llvm::StringMap<Attribute>& identifier_to_attr,
    QuantMethod quantization_method, bool enable_per_channel_quantization) {
  NamedAttrList attrs;
  if (quantization_method ==
      tensorflow::quantization::QuantizationMethod::METHOD_DYNAMIC_RANGE_INT8) {
    if (failed(FillQuantizationAttributes(rewriter, op, attrs,
                                          identifier_to_attr,
                                          OpType::kDynamicRangeOp))) {
      return failure();
    }
  } else {
    if (failed(FillQuantizationAttributes(
            rewriter, op, attrs, identifier_to_attr, OpType::kBinaryOp))) {
      return failure();
    }
    attrs.push_back(rewriter.getNamedAttr("lhs_quantization_axis",
                                          rewriter.getI64IntegerAttr(-1)));
  }
  std::unique_ptr<OpQuantSpec> spec = GetUniformOpQuantSpec(op);
  absl::flat_hash_set<int> operands = spec->quantizable_operands;
  int quant_dim = -1;
  if (enable_per_channel_quantization && operands.size() == 1) {
    quant_dim = spec->coeff_op_quant_dim[*(operands.begin())];
  }
  attrs.push_back(rewriter.getNamedAttr("rhs_quantization_axis",
                                        rewriter.getI64IntegerAttr(quant_dim)));
  attrs.push_back(rewriter.getNamedAttr("output_quantization_axis",
                                        rewriter.getI64IntegerAttr(quant_dim)));
  op->setAttrs(rewriter.getDictionaryAttr(attrs));
  return success();
}
LogicalResult FillAttributesForUniformQuantizedConvolutionOp(
    PatternRewriter& rewriter, Operation* op,
    llvm::StringMap<Attribute>& identifier_to_attr,
    QuantMethod quantization_method, bool enable_per_channel_quantization) {
  NamedAttrList attrs;
  absl::flat_hash_map<std::string, Attribute (*)(PatternRewriter&,
                                                 llvm::StringMap<Attribute>&)>
      attribute_getter_map;
  attribute_getter_map = {{"window_strides", GetWindowStridesValue},
                          {"lhs_dilation", GetLhsDilationValue},
                          {"rhs_dilation", GetRhsDilationValue},
                          {"padding", GetPaddingValue},
                          {"explicit_padding", GetExplicitPaddingValue},
                          {"dimension_numbers", GetDimensionNumbersValue},
                          {"batch_group_count", GetBatchGroupCountValue}};
  for (auto& attr : op->getAttrs()) {
    llvm::StringRef attr_name = attr.getName().getValue();
    if (attribute_getter_map.find(attr_name.str()) !=
        attribute_getter_map.end()) {
      auto attr_val =
          (attribute_getter_map[attr_name.str()])(rewriter, identifier_to_attr);
      attrs.push_back(rewriter.getNamedAttr(attr_name, attr_val));
    }
  }
  auto feature_group_cnt_attr = llvm::StringRef("feature_group_count");
  int feature_group_cnt = 1;
  ShapedType input_shape =
      mlir::dyn_cast<ShapedType>(op->getOperand(0).getType());
  if (!input_shape) {
    return op->emitError(
        "Only input with known shape is supported for Uniform Quantized "
        "opset.");
  }
  if (op->getParentOfType<func::FuncOp>().getName().contains("depthwise_")) {
    feature_group_cnt = input_shape.getDimSize(3);
  }
  attrs.push_back(rewriter.getNamedAttr(
      feature_group_cnt_attr, rewriter.getI64IntegerAttr(feature_group_cnt)));
  if (quantization_method ==
      tensorflow::quantization::QuantizationMethod::METHOD_DYNAMIC_RANGE_INT8) {
    if (failed(FillQuantizationAttributes(rewriter, op, attrs,
                                          identifier_to_attr,
                                          OpType::kDynamicRangeOp))) {
      return failure();
    }
  } else {
    if (failed(FillQuantizationAttributes(
            rewriter, op, attrs, identifier_to_attr, OpType::kBinaryOp))) {
      return failure();
    }
  }
  if (quantization_method !=
      tensorflow::quantization::QuantizationMethod::METHOD_DYNAMIC_RANGE_INT8) {
    attrs.push_back(rewriter.getNamedAttr("lhs_quantization_axis",
                                          rewriter.getI64IntegerAttr(-1)));
  }
  std::unique_ptr<OpQuantSpec> spec = GetUniformOpQuantSpec(op);
  absl::flat_hash_set<int> operands = spec->quantizable_operands;
  int quant_dim = -1;
  if (enable_per_channel_quantization && operands.size() == 1) {
    quant_dim = spec->coeff_op_quant_dim[*(operands.begin())];
  }
  attrs.push_back(rewriter.getNamedAttr("rhs_quantization_axis",
                                        rewriter.getI64IntegerAttr(quant_dim)));
  attrs.push_back(rewriter.getNamedAttr("output_quantization_axis",
                                        rewriter.getI64IntegerAttr(quant_dim)));
  op->setAttrs(rewriter.getDictionaryAttr(attrs));
  return success();
}
LogicalResult FillAttributesForUniformQuantizedAddOp(
    PatternRewriter& rewriter, Operation* op,
    llvm::StringMap<Attribute>& identifier_to_attr,
    const QuantMethod quantization_method,
    const bool enable_per_channel_quantization) {
  NamedAttrList attrs;
  if (failed(FillQuantizationAttributes(rewriter, op, attrs, identifier_to_attr,
                                        OpType::kBinaryOp))) {
    return failure();
  }
  Attribute activation_quantization_axis = rewriter.getI64IntegerAttr(-1);
  if (enable_per_channel_quantization) {
    activation_quantization_axis =
        GetQuantizationAxis(rewriter, op, 0);
    if (activation_quantization_axis == rewriter.getI64IntegerAttr(-1)) {
      activation_quantization_axis =
          GetQuantizationAxis(rewriter, op, 1);
    }
  }
  attrs.push_back(rewriter.getNamedAttr("lhs_quantization_axis",
                                        activation_quantization_axis));
  attrs.push_back(rewriter.getNamedAttr("rhs_quantization_axis",
                                        activation_quantization_axis));
  attrs.push_back(rewriter.getNamedAttr("output_quantization_axis",
                                        activation_quantization_axis));
  op->setAttrs(rewriter.getDictionaryAttr(attrs));
  return success();
}
LogicalResult FillAttributesForUniformQuantizedClipByValueOp(
    PatternRewriter& rewriter, Operation* op,
    llvm::StringMap<Attribute>& identifier_to_attr,
    QuantMethod quantization_method, bool enable_per_channel_quantization) {
  NamedAttrList attrs;
  if (failed(FillQuantizationAttributes(rewriter, op, attrs, identifier_to_attr,
                                        OpType::kUnaryOp))) {
    return failure();
  }
  Attribute activation_quantization_axis = rewriter.getI64IntegerAttr(-1);
  if (enable_per_channel_quantization) {
    activation_quantization_axis =
        GetQuantizationAxis(rewriter, op, 0);
  }
  attrs.push_back(
      rewriter.getNamedAttr("quantization_axis", activation_quantization_axis));
  op->setAttrs(rewriter.getDictionaryAttr(attrs));
  return success();
}
LogicalResult FillAttributesForUniformRequantizeOp(
    PatternRewriter& rewriter, Operation* op,
    llvm::StringMap<Attribute>& identifier_to_attr,
    QuantMethod quantization_method, bool enable_per_channel_quantization) {
  NamedAttrList attrs;
  if (failed(FillQuantizationAttributes(rewriter, op, attrs, identifier_to_attr,
                                        OpType::kQuantizationOp))) {
    return failure();
  }
  Attribute activation_quantization_axis = rewriter.getI64IntegerAttr(-1);
  Attribute output_quantization_axis = rewriter.getI64IntegerAttr(-1);
  if (enable_per_channel_quantization) {
    activation_quantization_axis =
        GetQuantizationAxis(rewriter, op, 0);
    auto output_scale_type =
        mlir::dyn_cast<ShapedType>(op->getOperand(3).getType());
    if (!output_scale_type) {
      return failure();
    }
    if (output_scale_type.hasRank() && 0 < output_scale_type.getRank()) {
      output_quantization_axis = activation_quantization_axis;
    }
  }
  attrs.push_back(rewriter.getNamedAttr("input_quantization_axis",
                                        activation_quantization_axis));
  attrs.push_back(rewriter.getNamedAttr("output_quantization_axis",
                                        output_quantization_axis));
  op->setAttrs(rewriter.getDictionaryAttr(attrs));
  return success();
}
LogicalResult FillAttributesForUniformQuantizeOp(
    PatternRewriter& rewriter, Operation* op,
    llvm::StringMap<Attribute>& identifier_to_attr,
    QuantMethod quantization_method, bool enable_per_channel_quantization) {
  NamedAttrList attrs;
  if (failed(FillQuantizationAttributes(rewriter, op, attrs, identifier_to_attr,
                                        OpType::kUnaryOp))) {
    return failure();
  }
  Attribute quantization_axis = rewriter.getI64IntegerAttr(-1);
  if (enable_per_channel_quantization) {
    quantization_axis = rewriter.getI64IntegerAttr(3);
  }
  attrs.push_back(
      rewriter.getNamedAttr("quantization_axis", quantization_axis));
  op->setAttrs(rewriter.getDictionaryAttr(attrs));
  return success();
}
}  