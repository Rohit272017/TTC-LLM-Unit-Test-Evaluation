#include "tensorflow/compiler/mlir/lite/quantization/numerical_utils.h"
#include <assert.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include "absl/types/optional.h"
namespace mlir {
namespace quant {
QuantizedMultiplier QuantizeMultiplier(double double_multiplier) {
  if (double_multiplier < 1e-6) {
    return {0, 0};
  }
  int32_t shift;
  const double q = frexp(double_multiplier, &shift);
  int64_t quantized_multiplier = round(q * (1LL << 31));
  assert(quantized_multiplier <= (1LL << 31));
  if (quantized_multiplier == (1LL << 31)) {
    quantized_multiplier /= 2;
    ++shift;
  }
  assert(quantized_multiplier <= std::numeric_limits<int32_t>::max());
  if (shift > 31 || shift < -31) {
    return {0, 0};
  }
  return {static_cast<int32_t>(quantized_multiplier), shift};
}
QuantizedRange CalculateQuantizedRange(double scale, int32_t zero_point,
                                       std::optional<double> rmin,
                                       std::optional<double> rmax, int32_t qmin,
                                       int32_t qmax) {
  auto quantize = [scale, zero_point](float f) {
    return zero_point + static_cast<int32_t>(std::round(f / scale));
  };
  if (rmin.has_value() && rmax.has_value()) {
    return {std::max(qmin, quantize(rmin.value())),
            std::min(qmax, quantize(rmax.value()))};
  } else if (rmin.has_value()) {
    return {std::max(qmin, quantize(rmin.value())), qmax};
  } else if (rmax.has_value()) {
    return {qmin, std::min(qmax, quantize(rmax.value()))};
  } else {
    return {qmin, qmax};
  }
}
}  
}  