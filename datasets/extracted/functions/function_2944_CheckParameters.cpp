#include <cmath>
#include <cstddef>
#include <type_traits>
#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/legacy/include/shlo.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/storage.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/util.h"
namespace stablehlo {
namespace {
template <typename Value>
absl::Status CheckParameters(const Value& lhs, const Value& rhs,
                             Value& result) {
  if (!(lhs.baseline_type() == rhs.baseline_type() and
        lhs.baseline_type() == result.baseline_type())) {
    return absl::InvalidArgumentError(
        "Constraint violation: baseline_type(on_true) = "
        "baseline_type(on_false) = baseline_type(result)");
  }
  if constexpr (std::is_same_v<Value, QuantizedTensor>) {
    if (!(lhs.is_per_tensor_quantized() and rhs.is_per_tensor_quantized() and
          result.is_per_tensor_quantized())) {
      return absl::InvalidArgumentError("Expected per=tensor quantization");
    }
  }
  if (lhs.layout().has_strides() || rhs.layout().has_strides() ||
      result.layout().has_strides()) {
    return absl::InvalidArgumentError("Stides not supported yet");
  }
  return absl::OkStatus();
}
template <ElementType storage_type, ElementType expressed_type, typename Value,
          typename Op>
absl::Status ElementwiseBinaryOp(const Value& lhs, const Value& rhs,
                                 Value& result, Op&& op) {
  if (auto check = CheckParameters(lhs, rhs, result); !check.ok()) {
    return check;
  }
  using S = Storage<storage_type>;
  auto lhs_buffer = lhs.buffer();
  auto rhs_buffer = rhs.buffer();
  auto result_buffer = result.buffer();
  size_t n = lhs.num_elements();
  if constexpr (std::is_same_v<Value, Tensor>) {
    if (storage_type != result.element_type()) {
      return absl::InvalidArgumentError("Unexpected tensor element type");
    }
    for (size_t i = 0; i < n; ++i) {
      auto x = S::Get(lhs_buffer, i);
      auto y = S::Get(rhs_buffer, i);
      auto z = op(x, y);
      S::Set(result_buffer, i, z);
    }
  } else {
    static_assert(std::is_same_v<Value, QuantizedTensor>);
    if (storage_type != result.storage_type()) {
      return absl::InvalidArgumentError("Unexpected storage type");
    } else if (expressed_type != result.expressed_type()) {
      return absl::InvalidArgumentError("Unexpected expressed type");
    }
    const QuantizedParameter& lhs_quant_param =
        lhs.type().element_type().parameters(0);
    const QuantizedParameter& rhs_quant_param =
        rhs.type().element_type().parameters(0);
    const QuantizedParameter& result_quant_param =
        result.type().element_type().parameters(0);
    using ET = typename Storage<expressed_type>::Type;
    ET result_scale_inv = ET(1.0) / static_cast<ET>(result_quant_param.scale);
    for (size_t i = 0; i < n; ++i) {
      auto lhs_storage = S::Get(lhs_buffer, i);
      auto rhs_storage = S::Get(rhs_buffer, i);
      auto result_storage =
          DequantizeOpQuantizePartial<storage_type, expressed_type>(
              lhs_storage, rhs_storage, lhs_quant_param, rhs_quant_param,
              result_scale_inv, result_quant_param.zero_point, op);
      S::Set(result_buffer, i, result_storage);
    }
    if (auto status = CompleteQuantization<storage_type>(result);
        !status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
}
#define DEFINE_ELEMENTWISE_BINARY_OP(name, element_type, expression)        \
  absl::Status name(const Tensor& lhs, const Tensor& rhs, Tensor& result) { \
    return ElementwiseBinaryOp<element_type, element_type>(                 \
        lhs, rhs, result, [](auto x, auto y) { return expression; });       \
  }
#define DEFINE_ELEMENTWISE_BINARY_QUANTIZED_OP(name, storage_type,          \
                                               expressed_type, expression)  \
  absl::Status name(const QuantizedTensor& lhs, const QuantizedTensor& rhs, \
                    QuantizedTensor& result) {                              \
    return ElementwiseBinaryOp<storage_type, expressed_type>(               \
        lhs, rhs, result, [](auto x, auto y) { return expression; });       \
  }
#define DEFINE_ELEMENTWISE_BINARY_OP_BOOL(name, expression) \
  DEFINE_ELEMENTWISE_BINARY_OP(name##_i1, ElementType::kI1, expression);
#define DEFINE_ELEMENTWISE_BINARY_OP_INT(name, expression)                   \
  DEFINE_ELEMENTWISE_BINARY_OP(name##_si8, ElementType::kSI8, expression);   \
  DEFINE_ELEMENTWISE_BINARY_OP(name##_si16, ElementType::kSI16, expression); \
  DEFINE_ELEMENTWISE_BINARY_OP(name##_si32, ElementType::kSI32, expression);
#define DEFINE_ELEMENTWISE_BINARY_OP_FLOAT(name, expression)                   \
  DEFINE_ELEMENTWISE_BINARY_OP(name##_bf16, ElementType::kBF16, expression);   \
  DEFINE_ELEMENTWISE_BINARY_OP(name##_f16, ElementType::kF16, expression);     \
  DEFINE_ELEMENTWISE_BINARY_OP(name##_f32, ElementType::kF32, expression);     \
  DEFINE_ELEMENTWISE_BINARY_QUANTIZED_OP(name##_q_si8_bf16, ElementType::kSI8, \
                                         ElementType::kBF16, expression);      \
  DEFINE_ELEMENTWISE_BINARY_QUANTIZED_OP(name##_q_si8_f16, ElementType::kSI8,  \
                                         ElementType::kF16, expression);       \
  DEFINE_ELEMENTWISE_BINARY_QUANTIZED_OP(name##_q_si8_f32, ElementType::kSI8,  \
                                         ElementType::kF32, expression);       \
  DEFINE_ELEMENTWISE_BINARY_QUANTIZED_OP(                                      \
      name##_q_si16_bf16, ElementType::kSI16, ElementType::kBF16, expression); \
  DEFINE_ELEMENTWISE_BINARY_QUANTIZED_OP(                                      \
      name##_q_si16_f16, ElementType::kSI16, ElementType::kF16, expression);   \
  DEFINE_ELEMENTWISE_BINARY_QUANTIZED_OP(                                      \
      name##_q_si16_f32, ElementType::kSI16, ElementType::kF32, expression);   \
  DEFINE_ELEMENTWISE_BINARY_QUANTIZED_OP(                                      \
      name##_q_si32_bf16, ElementType::kSI32, ElementType::kBF16, expression); \
  DEFINE_ELEMENTWISE_BINARY_QUANTIZED_OP(                                      \
      name##_q_si32_f16, ElementType::kSI32, ElementType::kF16, expression);   \
  DEFINE_ELEMENTWISE_BINARY_QUANTIZED_OP(                                      \
      name##_q_si32_f32, ElementType::kSI32, ElementType::kF32, expression);
#define CALL_BINARY_OP_BOOL_HELPER(name, lhs, rhs, result) \
  case ElementType::kI1:                                   \
    return name##_i1(lhs, rhs, result);
#define CALL_BINARY_OP_INT_HELPER(name, lhs, rhs, result) \
  case ElementType::kSI8:                                 \
    return name##_si8(lhs, rhs, result);                  \
  case ElementType::kSI16:                                \
    return name##_si16(lhs, rhs, result);                 \
  case ElementType::kSI32:                                \
    return name##_si32(lhs, rhs, result);
#define CALL_BINARY_OP_FLOAT_HELPER(name, lhs, rhs, result) \
  case ElementType::kBF16:                                  \
    return name##_bf16(lhs, rhs, result);                   \
  case ElementType::kF16:                                   \
    return name##_f16(lhs, rhs, result);                    \
  case ElementType::kF32:                                   \
    return name##_f32(lhs, rhs, result);
#define CALL_BINARY_OP_BOOL_INT(name, lhs, rhs, result)                      \
  {                                                                          \
    auto element_type = lhs.element_type();                                  \
    switch (element_type) {                                                  \
      CALL_BINARY_OP_BOOL_HELPER(name, lhs, rhs, result);                    \
      CALL_BINARY_OP_INT_HELPER(name, lhs, rhs, result);                     \
      default:                                                               \
        return absl::InvalidArgumentError("Unexpected tensor element type"); \
    }                                                                        \
  }
#define CALL_BINARY_OP_INT(name, lhs, rhs, result)                           \
  {                                                                          \
    auto element_type = lhs.element_type();                                  \
    switch (element_type) {                                                  \
      CALL_BINARY_OP_INT_HELPER(name, lhs, rhs, result);                     \
      default:                                                               \
        return absl::InvalidArgumentError("Unexpected tensor element type"); \
    }                                                                        \
  }
#define CALL_BINARY_OP_INT_FLOAT(name, lhs, rhs, result)                     \
  {                                                                          \
    auto element_type = lhs.element_type();                                  \
    switch (element_type) {                                                  \
      CALL_BINARY_OP_INT_HELPER(name, lhs, rhs, result);                     \
      CALL_BINARY_OP_FLOAT_HELPER(name, lhs, rhs, result);                   \
      default:                                                               \
        return absl::InvalidArgumentError("Unexpected tensor element type"); \
    }                                                                        \
  }
#define CALL_BINARY_OP_FLOAT(name, lhs, rhs, result)                         \
  {                                                                          \
    auto element_type = lhs.element_type();                                  \
    switch (element_type) {                                                  \
      CALL_BINARY_OP_FLOAT_HELPER(name, lhs, rhs, result);                   \
      default:                                                               \
        return absl::InvalidArgumentError("Unexpected tensor element type"); \
    }                                                                        \
  }
#define CALL_BINARY_OP_BOOL_INT_FLOAT(name, lhs, rhs, result)                \
  {                                                                          \
    auto element_type = lhs.element_type();                                  \
    switch (element_type) {                                                  \
      CALL_BINARY_OP_BOOL_HELPER(name, lhs, rhs, result);                    \
      CALL_BINARY_OP_INT_HELPER(name, lhs, rhs, result);                     \
      CALL_BINARY_OP_FLOAT_HELPER(name, lhs, rhs, result);                   \
      default:                                                               \
        return absl::InvalidArgumentError("Unexpected tensor element type"); \
    }                                                                        \
  }
#define CALL_BINARY_QUANTIZED_OP(name, lhs, rhs, result)                    \
  {                                                                         \
    auto storage_type = lhs.storage_type();                                 \
    auto expressed_type = lhs.expressed_type();                             \
    switch (storage_type) {                                                 \
      case ElementType::kSI8:                                               \
        switch (expressed_type) {                                           \
          case ElementType::kBF16:                                          \
            return name##_q_si8_bf16(lhs, rhs, result);                     \
          case ElementType::kF16:                                           \
            return name##_q_si8_f16(lhs, rhs, result);                      \
          case ElementType::kF32:                                           \
            return name##_q_si8_f32(lhs, rhs, result);                      \
          default:                                                          \
            return absl::InvalidArgumentError("Unexpected expressed type"); \
        }                                                                   \
      case ElementType::kSI16:                                              \
        switch (expressed_type) {                                           \
          case ElementType::kBF16:                                          \
            return name##_q_si16_bf16(lhs, rhs, result);                    \
          case ElementType::kF16:                                           \
            return name##_q_si16_f16(lhs, rhs, result);                     \
          case ElementType::kF32:                                           \
            return name##_q_si16_f32(lhs, rhs, result);                     \
          default:                                                          \
            return absl::InvalidArgumentError("Unexpected expressed type"); \
        }                                                                   \
      case ElementType::kSI32:                                              \
        switch (expressed_type) {                                           \
          case ElementType::kBF16:                                          \
            return name##_q_si32_bf16(lhs, rhs, result);                    \
          case ElementType::kF16:                                           \
            return name##_q_si32_f16(lhs, rhs, result);                     \
          case ElementType::kF32:                                           \
            return name##_q_si32_f32(lhs, rhs, result);                     \
          default:                                                          \
            return absl::InvalidArgumentError("Unexpected expressed type"); \
        }                                                                   \
      default:                                                              \
        return absl::InvalidArgumentError("Unexpected storage type");       \
    }                                                                       \
  }
}  
namespace {
DEFINE_ELEMENTWISE_BINARY_OP_BOOL(Add, x or y);
DEFINE_ELEMENTWISE_BINARY_OP_INT(Add, x + y);
DEFINE_ELEMENTWISE_BINARY_OP_FLOAT(Add, x + y);
}  
absl::Status Add(const Tensor& lhs, const Tensor& rhs, Tensor& result) {
  CALL_BINARY_OP_BOOL_INT_FLOAT(Add, lhs, rhs, result);
}
absl::Status Add(const QuantizedTensor& lhs, const QuantizedTensor& rhs,
                 QuantizedTensor& result) {
  CALL_BINARY_QUANTIZED_OP(Add, lhs, rhs, result);
}
namespace {
DEFINE_ELEMENTWISE_BINARY_OP_BOOL(And, x&& y);
DEFINE_ELEMENTWISE_BINARY_OP_INT(And, x& y);
}  
absl::Status And(const Tensor& lhs, const Tensor& rhs, Tensor& result) {
  CALL_BINARY_OP_BOOL_INT(And, lhs, rhs, result);
}
namespace {
DEFINE_ELEMENTWISE_BINARY_OP_FLOAT(Atan2, std::atan2(static_cast<float>(x),
                                                     static_cast<float>(y)));
}  
absl::Status Atan2(const Tensor& lhs, const Tensor& rhs, Tensor& result) {
  CALL_BINARY_OP_FLOAT(Atan2, lhs, rhs, result);
}
absl::Status Atan2(const QuantizedTensor& lhs, const QuantizedTensor& rhs,
                   QuantizedTensor& result) {
  CALL_BINARY_QUANTIZED_OP(Atan2, lhs, rhs, result);
}
namespace {
DEFINE_ELEMENTWISE_BINARY_OP_INT(Divide, x / y);
DEFINE_ELEMENTWISE_BINARY_OP_FLOAT(Divide, x / y);
}  
absl::Status Divide(const Tensor& lhs, const Tensor& rhs, Tensor& result) {
  CALL_BINARY_OP_INT_FLOAT(Divide, lhs, rhs, result);
}
absl::Status Divide(const QuantizedTensor& lhs, const QuantizedTensor& rhs,
                    QuantizedTensor& result) {
  CALL_BINARY_QUANTIZED_OP(Divide, lhs, rhs, result);
}
namespace {
DEFINE_ELEMENTWISE_BINARY_OP_BOOL(Maximum, x or y);
DEFINE_ELEMENTWISE_BINARY_OP_INT(Maximum, (x > y) ? x : y);
DEFINE_ELEMENTWISE_BINARY_OP_FLOAT(Maximum, (x > y) ? x : y);
}  
absl::Status Maximum(const Tensor& lhs, const Tensor& rhs, Tensor& result) {
  CALL_BINARY_OP_BOOL_INT_FLOAT(Maximum, lhs, rhs, result);
}
absl::Status Maximum(const QuantizedTensor& lhs, const QuantizedTensor& rhs,
                     QuantizedTensor& result) {
  CALL_BINARY_QUANTIZED_OP(Maximum, lhs, rhs, result);
}
namespace {
DEFINE_ELEMENTWISE_BINARY_OP_BOOL(Minimum, x and y);
DEFINE_ELEMENTWISE_BINARY_OP_INT(Minimum, (x > y) ? y : x);
DEFINE_ELEMENTWISE_BINARY_OP_FLOAT(Minimum, (x > y) ? y : x);
}  
absl::Status Minimum(const Tensor& lhs, const Tensor& rhs, Tensor& result) {
  CALL_BINARY_OP_BOOL_INT_FLOAT(Minimum, lhs, rhs, result);
}
absl::Status Minimum(const QuantizedTensor& lhs, const QuantizedTensor& rhs,
                     QuantizedTensor& result) {
  CALL_BINARY_QUANTIZED_OP(Minimum, lhs, rhs, result);
}
namespace {
DEFINE_ELEMENTWISE_BINARY_OP_BOOL(Multiply, x and y);
DEFINE_ELEMENTWISE_BINARY_OP_INT(Multiply, x* y);
DEFINE_ELEMENTWISE_BINARY_OP_FLOAT(Multiply, x* y);
}  
absl::Status Multiply(const Tensor& lhs, const Tensor& rhs, Tensor& result) {
  CALL_BINARY_OP_BOOL_INT_FLOAT(Multiply, lhs, rhs, result);
}
absl::Status Multiply(const QuantizedTensor& lhs, const QuantizedTensor& rhs,
                      QuantizedTensor& result) {
  CALL_BINARY_QUANTIZED_OP(Multiply, lhs, rhs, result);
}
namespace {
DEFINE_ELEMENTWISE_BINARY_OP_BOOL(Or, x or y);
DEFINE_ELEMENTWISE_BINARY_OP_INT(Or, x | y);
}  
absl::Status Or(const Tensor& lhs, const Tensor& rhs, Tensor& result) {
  CALL_BINARY_OP_BOOL_INT(Or, lhs, rhs, result);
}
namespace {
DEFINE_ELEMENTWISE_BINARY_OP_INT(Power, std::pow(static_cast<float>(x),
                                                 static_cast<int>(y)));
DEFINE_ELEMENTWISE_BINARY_OP_FLOAT(Power, std::powf(static_cast<float>(x),
                                                    static_cast<float>(y)));
}  
absl::Status Power(const Tensor& lhs, const Tensor& rhs, Tensor& result) {
  CALL_BINARY_OP_INT_FLOAT(Power, lhs, rhs, result);
}
absl::Status Power(const QuantizedTensor& lhs, const QuantizedTensor& rhs,
                   QuantizedTensor& result) {
  CALL_BINARY_QUANTIZED_OP(Power, lhs, rhs, result);
}
namespace {
DEFINE_ELEMENTWISE_BINARY_OP_INT(Remainder, x % y);
DEFINE_ELEMENTWISE_BINARY_OP_FLOAT(Remainder, std::fmod(static_cast<float>(x),
                                                        static_cast<float>(y)));
}  
absl::Status Remainder(const Tensor& lhs, const Tensor& rhs, Tensor& result) {
  CALL_BINARY_OP_INT_FLOAT(Remainder, lhs, rhs, result);
}
absl::Status Remainder(const QuantizedTensor& lhs, const QuantizedTensor& rhs,
                       QuantizedTensor& result) {
  CALL_BINARY_QUANTIZED_OP(Remainder, lhs, rhs, result);
}
namespace {
DEFINE_ELEMENTWISE_BINARY_OP_INT(ShiftLeft, x << y);
}  
absl::Status ShiftLeft(const Tensor& lhs, const Tensor& rhs, Tensor& result) {
  CALL_BINARY_OP_INT(ShiftLeft, lhs, rhs, result);
}
namespace {
DEFINE_ELEMENTWISE_BINARY_OP_INT(ShiftRightArithmetic, x >> y);
}  
absl::Status ShiftRightArithmetic(const Tensor& lhs, const Tensor& rhs,
                                  Tensor& result) {
  CALL_BINARY_OP_INT(ShiftRightArithmetic, lhs, rhs, result);
}
namespace {
template <typename Int>
inline Int ShiftRightLogical(Int x, Int y) {
  using UInt = typename std::make_unsigned<Int>::type;
  return static_cast<UInt>(x) >> y;
}
DEFINE_ELEMENTWISE_BINARY_OP_INT(ShiftRightLogical, ShiftRightLogical(x, y));
}  
absl::Status ShiftRightLogical(const Tensor& lhs, const Tensor& rhs,
                               Tensor& result) {
  CALL_BINARY_OP_INT(ShiftRightLogical, lhs, rhs, result);
}
namespace {
DEFINE_ELEMENTWISE_BINARY_OP_INT(Subtract, x - y);
DEFINE_ELEMENTWISE_BINARY_OP_FLOAT(Subtract, x - y);
}  
absl::Status Subtract(const Tensor& lhs, const Tensor& rhs, Tensor& result) {
  CALL_BINARY_OP_INT_FLOAT(Subtract, lhs, rhs, result);
}
absl::Status Subtract(const QuantizedTensor& lhs, const QuantizedTensor& rhs,
                      QuantizedTensor& result) {
  CALL_BINARY_QUANTIZED_OP(Subtract, lhs, rhs, result);
}
namespace {
DEFINE_ELEMENTWISE_BINARY_OP_BOOL(Xor, x xor y);
DEFINE_ELEMENTWISE_BINARY_OP_INT(Xor, x ^ y);
}  
absl::Status Xor(const Tensor& lhs, const Tensor& rhs, Tensor& result) {
  CALL_BINARY_OP_BOOL_INT(Xor, lhs, rhs, result);
}
}  