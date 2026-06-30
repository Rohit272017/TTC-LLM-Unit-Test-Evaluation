#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_CALIBRATION_CALIBRATION_PARAMETERS_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_CALIBRATION_CALIBRATION_PARAMETERS_H_
#include <algorithm>
#include <cmath>
#include <cstdint>
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
namespace stablehlo::quantization {
inline float CalculateBinWidth(const float min_value, const float max_value,
                               const int32_t num_bins) {
  const float raw_bin_width = (max_value - min_value) / num_bins;
  return std::pow(2, std::ceil(std::log2(raw_bin_width)));
}
inline float CalculateLowerBound(const float min_value, const float bin_width) {
  return std::floor(min_value / bin_width) * bin_width;
}
inline int32_t CalculateBinIndex(const float value, const float lower_bound,
                                 const float bin_width) {
  return std::floor((value - lower_bound) / bin_width);
}
inline int32_t CalculateBinIndexSafe(const float value, const float lower_bound,
                                     const float bin_width,
                                     const int32_t num_bins) {
  const int32_t bin_index = CalculateBinIndex(value, lower_bound, bin_width);
  return std::clamp(bin_index, 0, num_bins - 1);
}
inline bool IsHistogramCalibration(
    const CalibrationOptions::CalibrationMethod method) {
  return method ==
             CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_PERCENTILE ||
         method ==
             CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE ||
         method == CalibrationOptions::
                       CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY ||
         method ==
             CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_MSE_SYMMETRIC;
}
inline int32_t GetNumBins(const CalibrationOptions& calib_opts) {
  return IsHistogramCalibration(calib_opts.calibration_method())
             ? calib_opts.calibration_parameters().num_bins()
             : 0;
}
}  
#endif  