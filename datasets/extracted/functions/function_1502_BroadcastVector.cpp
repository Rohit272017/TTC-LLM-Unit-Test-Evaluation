#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  
#include "mlir/IR/Attributes.h"  
#include "mlir/IR/Builders.h"  
#include "mlir/IR/BuiltinAttributeInterfaces.h"  
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/BuiltinTypeInterfaces.h"  
#include "mlir/IR/BuiltinTypes.h"  
#include "mlir/IR/Diagnostics.h"  
#include "mlir/IR/MLIRContext.h"  
#include "mlir/IR/OpDefinition.h"  
#include "mlir/Support/LLVM.h"  
#include "mlir/Support/LogicalResult.h"  
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantizeUtils.h"
#include "tensorflow/compiler/mlir/lite/quantization/lite/toco_legacy/portable_tensor_utils.h"
#include "tensorflow/compiler/mlir/quantization/common/ir/FakeQuantSupport.h"
#include "tensorflow/compiler/mlir/quantization/common/ir/UniformSupport.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_traits.h"
#include "tensorflow/compiler/mlir/tools/optimize/quantization_utils.h"
namespace mlir {
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_interface.cc.inc"
namespace quant {
namespace {
constexpr double kSmallestHalfRange = kNearZeroTolerance / 2;
using QType = quant::QuantizedType;
template <typename T>
bool BroadcastVector(int target_size, SmallVectorImpl<T>& data) {
  const int size = data.size();
  if (size != target_size) {
    if (target_size % size != 0) return true;
    data.reserve(target_size);
    for (int i = 1; i < target_size / size; ++i) {
      data.insert(data.end(), data.begin(), data.begin() + size);
    }
  }
  return false;
}
void ExpandVerySmallRange(const ArrayRef<double> mins,
                          const ArrayRef<double> maxs,
                          SmallVectorImpl<double>& effective_mins,
                          SmallVectorImpl<double>& effective_maxs) {
  for (const auto [min, max] : llvm::zip(mins, maxs)) {
    if (max - min > kNearZeroTolerance) {
      effective_mins.push_back(min);
      effective_maxs.push_back(max);
    } else {
      effective_mins.push_back(std::min(min, -kSmallestHalfRange));
      effective_maxs.push_back(std::max(max, kSmallestHalfRange));
    }
  }
}
QuantizedType ResetMinMaxFromNumBits(const QuantizedType type,
                                     const int num_bits,
                                     const bool narrow_range,
                                     const bool is_signed) {
  if (num_bits >= 8) {
    return type;
  }
  int64_t qmin = QType::getDefaultMinimumForInteger(is_signed, num_bits);
  int64_t qmax = QType::getDefaultMaximumForInteger(is_signed, num_bits);
  if (narrow_range) {
    qmin += 1;
  }
  const int64_t storage_type_min = type.getStorageTypeMin();
  const int64_t storage_type_max = type.getStorageTypeMax();
  const double rate =
      static_cast<double>(storage_type_max - storage_type_min) / (qmax - qmin);
  const auto& recalculate_scale = [&](double scale) -> double {
    return scale * rate;
  };
  const auto& recalculate_zero_point = [&](int64_t zero_point) -> int64_t {
    return qmax - std::round((storage_type_max - zero_point) / rate);
  };
  if (auto q_type = dyn_cast<UniformQuantizedType>(type)) {
    const double scale = recalculate_scale(q_type.getScale());
    const double zero_point = recalculate_zero_point(q_type.getZeroPoint());
    return UniformQuantizedType::get(q_type.getFlags(), q_type.getStorageType(),
                                     q_type.getExpressedType(), scale,
                                     zero_point, qmin, qmax);
  } else if (auto q_type = dyn_cast<quant::UniformQuantizedPerAxisType>(type)) {
    const int size = q_type.getScales().size();
    SmallVector<double, 4> scales(size);
    SmallVector<int64_t, 4> zero_points(size);
    for (int i = 0; i < size; ++i) {
      scales[i] = recalculate_scale(q_type.getScales()[i]);
      zero_points[i] = recalculate_zero_point(q_type.getZeroPoints()[i]);
    }
    return quant::UniformQuantizedPerAxisType::get(
        q_type.getFlags(), q_type.getStorageType(), q_type.getExpressedType(),
        scales, zero_points, q_type.getQuantizedDimension(), qmin, qmax);
  } else {
    llvm_unreachable("Unsupported QuantizedType in ResetMinMaxFromNumBits");
  }
  return type;
}
quant::UniformQuantizedPerAxisType ResetAxisAndBroadcast(
    const ArrayRef<int64_t> shape,
    const quant::UniformQuantizedPerAxisType qtype, const Type target,
    const int quant_dim) {
  const auto shaped = dyn_cast<RankedTensorType>(target);
  if (!shaped) return {};
  const ArrayRef<int64_t> new_shape = shaped.getShape();
  SmallVector<double, 4> scales(qtype.getScales().begin(),
                                qtype.getScales().end());
  SmallVector<int64_t, 4> zero_points(qtype.getZeroPoints().begin(),
                                      qtype.getZeroPoints().end());
  if (new_shape.size() == shape.size()) {  
    if (BroadcastVector<double>(shaped.getDimSize(quant_dim), scales) ||
        BroadcastVector<int64_t>(shaped.getDimSize(quant_dim), zero_points)) {
      return {};
    }
  } else if ((new_shape.size() == shape.size() + 1) && new_shape.front() == 1) {
    if (!(std::equal(shape.begin(), shape.end(), new_shape.begin() + 1) &&
          quant_dim == new_shape.size() - 1)) {
      return {};
    }
  } else {
    return {};
  }
  return quant::UniformQuantizedPerAxisType::get(
      qtype.getFlags(), qtype.getStorageType(), qtype.getExpressedType(),
      scales, zero_points, quant_dim, qtype.getStorageTypeMin(),
      qtype.getStorageTypeMax());
}
}  
bool IsOpQuantizable(Operation* op) {
  if (isa<func::ConstantOp, arith::ConstantOp, quantfork::StatisticsOp>(op)) {
    return true;
  } else if (op->hasTrait<OpTrait::IsTerminator>() ||
             isa<quantfork::QuantizeCastOp, quantfork::DequantizeCastOp>(op)) {
    return false;
  }
  const bool attr_enforced_quantizable =
      op->hasAttrOfType<StringAttr>(kQuantTraitAttrName) &&
      op->getAttrOfType<StringAttr>(kQuantTraitAttrName).getValue().str() ==
          QuantTraitValues[QuantizationTrait::FullyQuantizable];
  const bool trait_enforced_quantizable =
      op->hasTrait<OpTrait::quant::QuantizableResult>();
  return attr_enforced_quantizable || trait_enforced_quantizable;
}
Type GetQuantizedType(Builder builder, const Type input_type,
                      const ArrayRef<double> min, const ArrayRef<double> max,
                      const int quant_dim, const int storage_type_width,
                      const bool narrow_range, const bool is_signed,
                      const bool legacy_float_scale,
                      const bool use_fake_quant_num_bits) {
  auto converter =
      quantfork::ExpressedToQuantizedConverter::forInputType(input_type);
  SmallVector<double, 4> effective_mins, effective_maxs;
  ExpandVerySmallRange(min, max, effective_mins, effective_maxs);
  quant::QuantizedType quantized_element_type;
  if (min.size() == 1 && max.size() == 1 && quant_dim == -1) {
    quantized_element_type = quantfork::fakeQuantAttrsToType(
        builder.getUnknownLoc(), storage_type_width, effective_mins[0],
        effective_maxs[0], narrow_range, converter.expressed_type, is_signed);
    if (legacy_float_scale) {
      quantized_element_type =
          DownCastScale(quantized_element_type, effective_mins[0],
                        effective_maxs[0], builder.getUnknownLoc());
    }
  } else if (min.size() == max.size()) {
    auto shape = dyn_cast<ShapedType>(input_type);
    if (!shape || shape.getRank() <= quant_dim ||
        static_cast<int64_t>(min.size()) != shape.getDimSize(quant_dim)) {
      return {};
    }
    quantized_element_type = quantfork::fakeQuantAttrsToType(
        builder.getUnknownLoc(), storage_type_width, quant_dim, effective_mins,
        effective_maxs, narrow_range, converter.expressed_type, is_signed);
    if (legacy_float_scale) {
      quantized_element_type =
          DownCastScale(quantized_element_type, effective_mins, effective_maxs,
                        builder.getUnknownLoc());
    }
  }
  if (!quantized_element_type) return {};
  if (use_fake_quant_num_bits && storage_type_width > 1 &&
      storage_type_width < 8 &&
      quantized_element_type.getStorageTypeMax() >
          QType::getDefaultMinimumForInteger(is_signed, storage_type_width)) {
    const auto resetEleType = ResetMinMaxFromNumBits(
        quantized_element_type, storage_type_width, narrow_range, is_signed);
    return converter.convert(resetEleType);
  }
  return converter.convert(quantized_element_type);
}
TypeAttr RescaleQuantizedType(const Type input, const Attribute factor) {
  const auto factor_values = dyn_cast_or_null<DenseFPElementsAttr>(factor);
  if (!factor_values) return {};
  const auto element_type =
      quant::QuantizedType::getQuantizedElementType(input);
  if (!element_type) return {};
  if (auto qtype = dyn_cast<quant::UniformQuantizedPerAxisType>(element_type)) {
    const ArrayRef<double> scales = qtype.getScales();
    if (static_cast<int64_t>(scales.size()) != factor_values.getNumElements())
      return {};
    SmallVector<double, 4> new_scales;
    new_scales.reserve(scales.size());
    auto scales_iter = scales.begin();
    for (const auto& f : factor_values) {
      new_scales.push_back(*scales_iter *
                           std::fabs(FloatAttr::getValueAsDouble(f)));
      ++scales_iter;
    }
    auto new_ele_type = quant::UniformQuantizedPerAxisType::get(
        qtype.getFlags(), qtype.getStorageType(), qtype.getExpressedType(),
        new_scales, qtype.getZeroPoints(), qtype.getQuantizedDimension(),
        qtype.getStorageTypeMin(), qtype.getStorageTypeMax());
    if (const auto new_type = new_ele_type.castFromExpressedType(
            quant::QuantizedType::castToExpressedType(input))) {
      return TypeAttr::get(new_type);
    }
  }
  return {};
}
TypeAttr GetQuantizedTypeAttr(const Builder builder, const Type input_type,
                              const Attribute min, const Attribute max,
                              const int quant_dim, const IntegerAttr num_bits,
                              const BoolAttr narrow_range, const bool is_signed,
                              const bool legacy_float_scale,
                              const bool use_fake_quant_num_bits) {
  SmallVector<double, 4> min_value, max_value;
  const auto mins = dyn_cast<DenseFPElementsAttr>(min);
  const auto maxs = dyn_cast<DenseFPElementsAttr>(max);
  if (mins && maxs) {
    min_value.reserve(mins.getNumElements());
    max_value.reserve(maxs.getNumElements());
    for (auto it = mins.begin(); it != mins.end(); ++it) {
      min_value.push_back(FloatAttr::getValueAsDouble(*it));
    }
    for (auto it = maxs.begin(); it != maxs.end(); ++it) {
      max_value.push_back(FloatAttr::getValueAsDouble(*it));
    }
  } else {
    const auto fmin = dyn_cast<FloatAttr>(min);
    const auto fmax = dyn_cast<FloatAttr>(max);
    if (fmin && fmax) {
      min_value.push_back(fmin.getValueAsDouble());
      max_value.push_back(fmax.getValueAsDouble());
    } else {
      return {};
    }
  }
  const Type final_type =
      GetQuantizedType(builder, input_type, min_value, max_value, quant_dim,
                       num_bits.getInt(), narrow_range.getValue(), is_signed,
                       legacy_float_scale, use_fake_quant_num_bits);
  if (!final_type) return {};
  return TypeAttr::get(final_type);
}
TypeAttr CastQuantizedTypeAttrFromExpressedType(const Builder builder,
                                                const TypeAttr source,
                                                const Type target,
                                                const int axis) {
  const auto source_type = dyn_cast_or_null<ShapedType>(source.getValue());
  if (!source_type) return {};
  const auto src_ele_type = source_type.getElementType();
  auto qtype = dyn_cast<quant::QuantizedType>(src_ele_type);
  if (const auto per_axis =
          dyn_cast_or_null<quant::UniformQuantizedPerAxisType>(qtype)) {
    if (axis == -1) return {};
    qtype =
        ResetAxisAndBroadcast(source_type.getShape(), per_axis, target, axis);
  }
  if (!qtype) return {};
  const Type final_type = qtype.castFromExpressedType(target);
  if (!final_type) return {};
  return TypeAttr::get(final_type);
}
void ExtractMinMaxFromAttr(const DenseFPElementsAttr values, const int dim_size,
                           const int slice_size, bool symmetric,
                           SmallVectorImpl<double>& mins,
                           SmallVectorImpl<double>& maxs) {
  if (values.isSplat()) {
    const double single_value =
        FloatAttr::getValueAsDouble(values.getSplatValue<llvm::APFloat>());
    if (single_value < 0.0) {
      mins[0] = single_value;
      maxs[0] = symmetric ? -single_value : 0.0;
    } else if (single_value > 0.0) {
      mins[0] = symmetric ? -single_value : 0.0;
      maxs[0] = single_value;
    } else {
      mins[0] = maxs[0] = single_value;
    }
    for (int i = 1; i < dim_size; ++i) {
      mins[i] = mins[0];
      maxs[i] = maxs[0];
    }
  } else {
    int64_t flatten_index = 0;
    auto begin = values.begin();
    auto end = values.end();
    for (auto it = begin; it != end; ++it, ++flatten_index) {
      const double ele_value = FloatAttr::getValueAsDouble(*it);
      const int slice_index = flatten_index / slice_size;
      const int channel_index = slice_index % dim_size;
      mins[channel_index] = std::min(mins[channel_index], ele_value);
      maxs[channel_index] = std::max(maxs[channel_index], ele_value);
    }
    for (int i = 0; i < dim_size; ++i) {
      maxs[i] = std::max(maxs[i], 0.0);
      mins[i] = std::min(mins[i], 0.0);
    }
    if (symmetric) {
      for (int i = 0; i < dim_size; ++i) {
        maxs[i] = std::max(std::abs(mins[i]), std::abs(maxs[i]));
        mins[i] = -maxs[i];
      }
    }
  }
}
Type GetUniformQuantizedTypeForWeight(
    const ElementsAttr attr, const bool symmetric, const unsigned num_bits,
    const bool is_signed, const bool narrow_range,
    const bool legacy_float_scale, const bool use_fake_quant_num_bits) {
  const Builder builder(attr.getContext());
  if (symmetric && (!is_signed || !narrow_range)) return {};
  SmallVector<double, 4> mins(1, std::numeric_limits<double>::max());
  SmallVector<double, 4> maxs(1, std::numeric_limits<double>::min());
  const auto fp = dyn_cast<DenseFPElementsAttr>(attr);
  if (!fp) return {};
  ExtractMinMaxFromAttr(fp, 1, 1, symmetric, mins,
                        maxs);
  const auto type =
      GetQuantizedType(builder, attr.getType(), mins[0], maxs[0],
                       -1, num_bits, narrow_range, is_signed,
                       legacy_float_scale, use_fake_quant_num_bits);
  if (const auto ele_type = dyn_cast_or_null<TensorType>(type))
    return ele_type.getElementType();
  return {};
}
Type GetUniformQuantizedPerAxisTypeForWeight(
    const ElementsAttr attr, const int quant_dim, const bool symmetric,
    const unsigned num_bits, const bool is_signed, const bool narrow_range,
    const bool legacy_float_scale, const bool use_fake_quant_num_bits) {
  const Builder builder(attr.getContext());
  const auto shape = cast<ShapedType>(attr.getType()).getShape();
  if (static_cast<int>(shape.size()) <= quant_dim) return {};
  if (symmetric && (!is_signed || !narrow_range)) return {};
  const int dim_size = shape[quant_dim];
  const int slice_size =
      std::accumulate(std::next(shape.begin(), quant_dim + 1), shape.end(), 1,
                      std::multiplies<int64_t>());
  SmallVector<double, 4> mins(dim_size, std::numeric_limits<double>::max());
  SmallVector<double, 4> maxs(dim_size, std::numeric_limits<double>::min());
  const auto fp = dyn_cast<DenseFPElementsAttr>(attr);
  if (!fp) return {};
  ExtractMinMaxFromAttr(fp, dim_size, slice_size, symmetric, mins, maxs);
  const auto type = GetQuantizedType(
      builder, attr.getType(), mins, maxs, quant_dim, num_bits, narrow_range,
      is_signed, legacy_float_scale, use_fake_quant_num_bits);
  if (auto ele_type = dyn_cast_or_null<TensorType>(type))
    return ele_type.getElementType();
  return {};
}
quant::QuantizedType GetUniformQuantizedTypeForBias(
    const std::vector<quant::QuantizedType>& op_types,
    const int adjusted_quant_dim, const bool legacy_float_scale) {
  if (op_types.empty()) return {};
  size_t axis_size = 1;
  int32_t quant_dim = -1;
  Type expressed_type;
  for (const auto op_type : op_types) {
    if (!op_type) return {};
    if (expressed_type && expressed_type != op_type.getExpressedType()) {
      return {};
    }
    expressed_type = op_type.getExpressedType();
    if (const auto type =
            dyn_cast<quant::UniformQuantizedPerAxisType>(op_type)) {
      if (axis_size != 1 && axis_size != type.getScales().size()) return {};
      if (quant_dim != -1 && quant_dim != type.getQuantizedDimension())
        return {};
      axis_size = type.getScales().size();
      quant_dim = type.getQuantizedDimension();
    } else if (!isa<quant::UniformQuantizedType>(op_type)) {
      return {};
    }
  }
  SmallVector<double, 4> scales(axis_size, 1.0);
  for (const auto op_type : op_types) {
    if (const auto type =
            dyn_cast<quant::UniformQuantizedPerAxisType>(op_type)) {
      for (const auto& index_scale : llvm::enumerate(type.getScales())) {
        scales[index_scale.index()] *= index_scale.value();
      }
    } else if (const auto type =
                   dyn_cast<quant::UniformQuantizedType>(op_type)) {
      for (int index = 0; index < axis_size; ++index) {
        scales[index] *= type.getScale();
      }
    }
  }
  if (legacy_float_scale) {
    for (int i = 0; i < scales.size(); ++i) {
      scales[i] = static_cast<float>(scales[i]);
    }
  }
  Builder builder(expressed_type.getContext());
  const IntegerType storage_type = builder.getIntegerType(32);
  const int64_t storage_type_min =
      quant::QuantizedType::getDefaultMinimumForInteger(true, 32);
  const int64_t storage_type_max =
      quant::QuantizedType::getDefaultMaximumForInteger(true, 32);
  if (axis_size == 1) {
    return quant::UniformQuantizedType::getChecked(
        builder.getUnknownLoc(),
        true, storage_type, expressed_type, scales[0],
        0, storage_type_min, storage_type_max);
  } else {
    SmallVector<int64_t, 4> zero_points(axis_size, 0);
    return quant::UniformQuantizedPerAxisType::getChecked(
        builder.getUnknownLoc(),
        true, storage_type, expressed_type, scales, zero_points,
        std::max(adjusted_quant_dim, 0),
        storage_type_min, storage_type_max);
  }
}
ElementsAttr QuantizeLegacy(const Attribute real_value,
                            const Type tensor_type) {
  if (!isa<DenseFPElementsAttr>(real_value) ||
      !quant::QuantizedType::getQuantizedElementType(tensor_type)) {
    return {};
  }
  const auto real_values_attr = cast<DenseFPElementsAttr>(real_value);
  auto q_type = quant::QuantizedType::getQuantizedElementType(tensor_type);
  std::vector<float> real_values;
  SmallVector<APInt, 8> quantized_attr;
  real_values.reserve(real_values_attr.getNumElements());
  quantized_attr.reserve(real_values_attr.getNumElements());
  std::transform(real_values_attr.begin(), real_values_attr.end(),
                 std::back_inserter(real_values), [&](APFloat value) -> float {
                   return value.convertToFloat();
                 });
  const ShapedType new_dense_type = dyn_cast_or_null<ShapedType>(
      q_type.castExpressedToStorageType(real_values_attr.getType()));
  const int width = dyn_cast<IntegerType>(q_type.getStorageType()).getWidth();
  if (width == 8 && q_type.getStorageTypeMax() == 127 &&
      q_type.getStorageTypeMin() == -127) {
    std::vector<int8_t> quantized_values(real_values_attr.getNumElements());
    if (auto uniform_type = dyn_cast<UniformQuantizedType>(q_type)) {
      float min, max, scale;
      mlir::lite::toco_legacy::PortableSymmetricQuantizeFloats(
          real_values.data(), real_values.size(), quantized_values.data(), &min,
          &max, &scale);
      if (std::abs(scale - uniform_type.getScale()) > 1e-3) {
        return Quantize(real_value, tensor_type);
      }
    } else if (auto uniform_type =
                   dyn_cast<quant::UniformQuantizedPerAxisType>(q_type)) {
      std::vector<float> scales_inv;
      std::vector<int32_t> dimension;
      dimension.insert(dimension.end(), new_dense_type.getShape().begin(),
                       new_dense_type.getShape().end());
      std::transform(uniform_type.getScales().begin(),
                     uniform_type.getScales().end(),
                     std::back_inserter(scales_inv),
                     [](float scale) { return 1.0 / scale; });
      tflite_migration::optimize::utils::SymmetricPerChannelQuantizeValues(
          real_values.data(), scales_inv, dimension,
          uniform_type.getQuantizedDimension(), &quantized_values);
    } else {
      return {};
    }
    std::transform(quantized_values.begin(), quantized_values.end(),
                   std::back_inserter(quantized_attr),
                   [&](int8_t value) -> APInt {
                     return APInt(8, value, true);
                   });
    return DenseElementsAttr::get(new_dense_type, quantized_attr);
  } else if (width == 8) {
    return Quantize(real_value, tensor_type);
  } else if (width == 16) {
    if (const auto uniform_type = dyn_cast<UniformQuantizedType>(q_type)) {
      const auto quantized_values =
          tflite_migration::optimize::utils::SymmetricQuantizeFloatsToInt16(
              real_values.data(), real_values.size(), uniform_type.getScale());
      std::transform(quantized_values.begin(), quantized_values.end(),
                     std::back_inserter(quantized_attr),
                     [&](int16_t value) -> APInt {
                       return APInt(16, value, true);
                     });
      return DenseElementsAttr::get(new_dense_type, quantized_attr);
    }
  } else if (width == 32) {
    std::vector<float> scales;
    if (const auto uniform_type = dyn_cast<UniformQuantizedType>(q_type)) {
      scales.push_back(uniform_type.getScale());
    } else if (const auto uniform_type =
                   dyn_cast<quant::UniformQuantizedPerAxisType>(q_type)) {
      scales.insert(scales.end(), uniform_type.getScales().begin(),
                    uniform_type.getScales().end());
    } else {
      return {};
    }
    const auto quantized_bias =
        tflite_migration::optimize::utils::SymmetricBiasQuantize<std::int32_t>(
            real_values.data(), real_values.size(), scales);
    std::transform(quantized_bias.begin(), quantized_bias.end(),
                   std::back_inserter(quantized_attr),
                   [&](int32_t value) -> APInt {
                     return APInt(32, value, true);
                   });
    return DenseElementsAttr::get(new_dense_type, quantized_attr);
  }
  return {};
}
ElementsAttr Quantize(const Attribute real_value, const Type tensor_type) {
  if (const auto q_type =
          quant::QuantizedType::getQuantizedElementType(tensor_type)) {
    Type converted_type;
    return dyn_cast_or_null<ElementsAttr>(
        quantfork::quantizeAttr(real_value, q_type, converted_type));
  }
  return {};
}
quant::QuantizedType DownCastScale(QuantizedType type, double min, double max,
                                   Location loc) {
  const SmallVector<double, 1> mins = {min};
  const SmallVector<double, 1> maxs = {max};
  return DownCastScale(type, mins, maxs, loc);
}
quant::QuantizedType DownCastScale(QuantizedType type,
                                   const SmallVectorImpl<double>& mins,
                                   const SmallVectorImpl<double>& maxs,
                                   Location loc) {
  if (!type) return type;
  SmallVector<double, 4> scales(mins.size());
  SmallVector<int64_t, 4> zero_points(mins.size());
  if (auto q_type = dyn_cast<UniformQuantizedType>(type)) {
    zero_points.push_back(q_type.getZeroPoint());
  } else if (auto q_type = dyn_cast<quant::UniformQuantizedPerAxisType>(type)) {
    zero_points = {q_type.getZeroPoints().begin(),
                   q_type.getZeroPoints().end()};
  }
  for (int i = 0; i < mins.size(); ++i) {
    scales[i] = (static_cast<float>(maxs[i]) - static_cast<float>(mins[i])) /
                (type.getStorageTypeMax() - type.getStorageTypeMin());
    if (type.getStorageTypeMax() != -type.getStorageTypeMin()) {
      const float zero_point_from_min =
          type.getStorageTypeMin() - mins[i] / scales[i];
      if (zero_point_from_min < type.getStorageTypeMin()) {
        zero_points[i] = static_cast<int64_t>(type.getStorageTypeMin());
      } else if (zero_point_from_min > type.getStorageTypeMax()) {
        zero_points[i] = static_cast<int64_t>(type.getStorageTypeMax());
      } else {
        zero_points[i] = static_cast<int64_t>(std::round(zero_point_from_min));
      }
    }
  }
  if (auto q_type = dyn_cast<UniformQuantizedType>(type)) {
    return UniformQuantizedType::get(q_type.getFlags(), q_type.getStorageType(),
                                     q_type.getExpressedType(), scales[0],
                                     zero_points[0], q_type.getStorageTypeMin(),
                                     q_type.getStorageTypeMax());
  } else if (auto q_type = dyn_cast<quant::UniformQuantizedPerAxisType>(type)) {
    return quant::UniformQuantizedPerAxisType::get(
        q_type.getFlags(), q_type.getStorageType(), q_type.getExpressedType(),
        scales, zero_points, q_type.getQuantizedDimension(),
        q_type.getStorageTypeMin(), q_type.getStorageTypeMax());
  }
  return type;
}
static bool PreferResultScale(Operation* op) {
  int float_operands = 0;
  for (auto operand : op->getOperands()) {
    if (auto operand_type = dyn_cast<ShapedType>(operand.getType())) {
      if (isa<FloatType>(operand_type.getElementType())) {
        if (++float_operands > 1) return true;
      }
    }
  }
  return false;
}
std::unique_ptr<OpQuantScaleSpec> GetDefaultQuantScaleSpec(Operation* op) {
  auto spec = std::make_unique<OpQuantScaleSpec>();
  if (isa<SameScalesOpInterface>(op)) {
    spec->has_same_scale_requirement = true;
    spec->required_same_scale_func = [op](const bool sign,
                                          const int bit_width) {
      return cast<SameScalesOpInterface>(op)
          .RequiredSameOperandsAndResultsScale(sign, bit_width);
    };
    spec->required_same_quantized_axes_func = [op]() {
      return cast<SameScalesOpInterface>(op).RequiredSameQuantizedAxes();
    };
  }
  if (isa<FixedOutputRangeInterface>(op)) {
    spec->has_fixed_output_range = true;
    spec->fixed_output_range_func = [op](bool sign, int bit_width) {
      return cast<FixedOutputRangeInterface>(op).GetFixedOutputRange(sign,
                                                                     bit_width);
    };
  }
  return spec;
}
static bool IsStatsRedundant(
    Operation* op, const OpQuantSpecGetter op_quant_spec_getter,
    const OpQuantScaleSpecGetter op_quant_scale_spec_getter) {
  return isa<FixedOutputRangeInterface>(op) ||
         op_quant_scale_spec_getter(op)->has_fixed_output_range;
}
static bool IsSameScaleOp(
    Operation* op, const OpQuantScaleSpecGetter op_quant_scale_spec_getter) {
  return dyn_cast<SameScalesOpInterface>(op) ||
         op_quant_scale_spec_getter(op)->has_same_scale_requirement;
}
bool RemoveRedundantStatsOps(
    func::FuncOp func, const OpQuantSpecGetter op_quant_spec_getter,
    const OpQuantScaleSpecGetter op_quant_scale_spec_getter) {
  SmallVector<quantfork::StatisticsOp, 16> all_stats_ops;
  llvm::DenseSet<Operation*> redundant_stats_ops;
  func.walk([&](quantfork::QuantizeCastOp q) {
    auto input_op = q.getArg().getDefiningOp();
    if (auto stats = dyn_cast_or_null<quantfork::StatisticsOp>(input_op)) {
      q.setOperand(stats.getArg());
      if (stats.use_empty()) stats.erase();
    }
  });
  func.walk([&](quantfork::StatisticsOp stats_op) {
    all_stats_ops.push_back(stats_op);
  });
  while (!all_stats_ops.empty()) {
    quantfork::StatisticsOp stats_op = all_stats_ops.back();
    all_stats_ops.pop_back();
    if (auto def = stats_op.getArg().getDefiningOp()) {
      if (IsStatsRedundant(def, op_quant_spec_getter,
                           op_quant_scale_spec_getter)) {
        redundant_stats_ops.insert(stats_op);
      }
    }
    for (Operation* user : stats_op.getResult().getUsers()) {
      if (!IsSameScaleOp(user, op_quant_scale_spec_getter) ||
          PreferResultScale(user)) {
        continue;
      }
      for (Value res : user->getResults()) {
        if (!res.hasOneUse()) {
          continue;
        }
        if (auto next_stats =
                dyn_cast<quantfork::StatisticsOp>(*res.getUsers().begin())) {
          redundant_stats_ops.insert(next_stats);
          all_stats_ops.push_back(next_stats);
        }
      }
    }
  }
  func.walk([&](quantfork::StatisticsOp stats_op) {
    if (redundant_stats_ops.find(stats_op) == redundant_stats_ops.end()) {
      all_stats_ops.push_back(stats_op);
    }
  });
  while (!all_stats_ops.empty()) {
    quantfork::StatisticsOp stats_op = all_stats_ops.back();
    all_stats_ops.pop_back();
    if (Operation* def = stats_op.getArg().getDefiningOp()) {
      if (!IsSameScaleOp(def, op_quant_scale_spec_getter)) {
        continue;
      }
      for (Value input : def->getOperands()) {
        if (auto next_stats = dyn_cast_or_null<quantfork::StatisticsOp>(
                input.getDefiningOp())) {
          redundant_stats_ops.insert(next_stats);
          all_stats_ops.push_back(next_stats);
        }
      }
    }
  }
  for (Operation* it : redundant_stats_ops) {
    if (!isa<quantfork::StatisticsOp>(it)) return true;
    auto stats_op = cast<quantfork::StatisticsOp>(it);
    stats_op.getResult().replaceAllUsesWith(stats_op.getArg());
    stats_op.erase();
  }
  return false;
}
LogicalResult VerifySameScales(Operation* op) {
  auto same_scale_op = cast<SameScalesOpInterface>(op);
  SmallVector<QuantizedType, 4> collected_quant_params;
  for (Value input : op->getOperands()) {
    QuantizedType quant_params =
        QuantizedType::getQuantizedElementType(input.getType());
    if (quant_params) {
      collected_quant_params.push_back(quant_params);
    }
  }
  for (Value output : op->getResults()) {
    const QuantizedType quant_params =
        QuantizedType::getQuantizedElementType(output.getType());
    if (quant_params) {
      collected_quant_params.push_back(quant_params);
    }
  }
  if (collected_quant_params.size() <= 1) return success();
  const auto& expected_params = collected_quant_params[0];
  for (int i = 1; i < collected_quant_params.size(); ++i) {
    const auto& compared_params = collected_quant_params[i];
    if (!same_scale_op.RequiredSameQuantizedAxes()) {
      const auto expected_per_axis_qtype =
          dyn_cast<quant::UniformQuantizedPerAxisType>(expected_params);
      const auto compared_per_axis_qtype =
          dyn_cast<quant::UniformQuantizedPerAxisType>(compared_params);
      if (expected_per_axis_qtype && compared_per_axis_qtype &&
          llvm::equal(expected_per_axis_qtype.getScales(),
                      compared_per_axis_qtype.getScales()) &&
          llvm::equal(expected_per_axis_qtype.getZeroPoints(),
                      compared_per_axis_qtype.getZeroPoints()) &&
          expected_params.getStorageType() ==
              compared_params.getStorageType() &&
          expected_params.getExpressedType() ==
              compared_params.getExpressedType()) {
        continue;
      }
    }
    if (expected_params == compared_params) continue;
    if (expected_params.isSigned() == compared_params.isSigned() &&
        expected_params.getStorageTypeIntegralWidth() ==
            compared_params.getStorageTypeIntegralWidth() &&
        !same_scale_op.RequiredSameOperandsAndResultsScale(
            expected_params.isSigned(),
            expected_params.getStorageTypeIntegralWidth()))
      continue;
    std::string err_msg =
        "quantization parameters violate the same scale constraint: ";
    llvm::raw_string_ostream os(err_msg);
    expected_params.print(os);
    os << " vs. ";
    compared_params.print(os);
    os.flush();
    return op->emitOpError(err_msg);
  }
  return success();
}
quant::UniformQuantizedType GetFixedOutputRange(
    const bool is_signed, const int bit_width, const Type tensor_type,
    const double scale, int64_t zero_point, int64_t storage_min,
    int64_t storage_max) {
  const auto result_type = cast<ShapedType>(tensor_type);
  if (!isa<FloatType>(result_type.getElementType())) return {};
  Builder builder(result_type.getContext());
  if (bit_width != 8 && bit_width != 16) return {};
  const IntegerType storage_type = builder.getIntegerType(bit_width);
  if (!is_signed && bit_width == 8) {
    zero_point += 128;
    storage_min += 128;
    storage_max += 128;
  }
  return quant::UniformQuantizedType::getChecked(
      builder.getUnknownLoc(), is_signed, storage_type,
      result_type.getElementType(), scale, zero_point, storage_min,
      storage_max);
}
quant::UniformQuantizedType GetFixedOutputRange(const bool is_signed,
                                                const int bit_width,
                                                const Type tensor_type,
                                                const double scale,
                                                const int64_t zero_point) {
  return GetFixedOutputRange(is_signed, bit_width, tensor_type, scale,
                             zero_point,
                             -(1 << (bit_width - 1)),
                             (1 << (bit_width - 1)) - 1);
}
Type ConvertSignedQuantizedToUnsigned(const Type signed_tensor_type,
                                      const Location loc) {
  const auto qtype = QType::getQuantizedElementType(signed_tensor_type);
  if (!qtype || !qtype.isSigned()) return {};
  const int num_bits = qtype.getStorageTypeIntegralWidth();
  const int64_t offset =
      QType::getDefaultMinimumForInteger(true, num_bits) -
      QType::getDefaultMinimumForInteger(false, num_bits);
  const auto flags = !quant::QuantizationFlags::Signed;
  QType new_qtype;
  if (auto uqtype = dyn_cast<quant::UniformQuantizedType>(qtype)) {
    new_qtype = quant::UniformQuantizedType::getChecked(
        loc, flags, qtype.getStorageType(), qtype.getExpressedType(),
        uqtype.getScale(), uqtype.getZeroPoint() - offset,
        uqtype.getStorageTypeMin() - offset,
        uqtype.getStorageTypeMax() - offset);
  } else if (auto aqtype =
                 dyn_cast<quant::UniformQuantizedPerAxisType>(qtype)) {
    const auto zero_points = aqtype.getZeroPoints();
    SmallVector<int64_t, 4> new_zero_points(zero_points.begin(),
                                            zero_points.end());
    for (int i = 0; i < new_zero_points.size(); ++i) {
      new_zero_points[i] -= offset;
    }
    new_qtype = quant::UniformQuantizedPerAxisType::getChecked(
        loc, flags, qtype.getStorageType(), qtype.getExpressedType(),
        aqtype.getScales(), new_zero_points, aqtype.getQuantizedDimension(),
        aqtype.getStorageTypeMin() - offset,
        aqtype.getStorageTypeMax() - offset);
  }
  return new_qtype.castFromExpressedType(
      QType::castToExpressedType(signed_tensor_type));
}
LogicalResult RemoveDebugAttrPattern::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  return success(
                 op->removeAttr(kDebugModeOpQuantAttrName) ||
                 op->removeAttr(kDebugModeOpFloatAttrName));
}
}  
}  