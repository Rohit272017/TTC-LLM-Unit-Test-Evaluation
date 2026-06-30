#include "tensorflow/compiler/mlir/quantization/stablehlo/utils/math_utils.h"
#include <cmath>
#include <cstdint>
#include "mlir/Support/LogicalResult.h"  
namespace mlir::quant::stablehlo {
LogicalResult QuantizeMultiplier(double double_multiplier,
                                 int32_t& quantized_fraction, int32_t& shift) {
  if (!std::isfinite(double_multiplier) || double_multiplier <= 0) {
    return failure();
  }
  const double fraction = std::frexp(double_multiplier, &shift);
  quantized_fraction = static_cast<int32_t>(std::round(fraction * (1L << 15)));
  if (quantized_fraction == (1L << 15)) {
    quantized_fraction /= 2;
    ++shift;
  }
  if (shift < -15) {
    shift = 0;
    quantized_fraction = 0;
  }
  if (shift > 14) {
    shift = 14;
    quantized_fraction = (1LL << 15) - 1;
  }
  return success();
}
}  