#ifndef TENSORFLOW_LITE_TOOLS_OPTIMIZE_REDUCED_PRECISION_SUPPORT_H_
#define TENSORFLOW_LITE_TOOLS_OPTIMIZE_REDUCED_PRECISION_SUPPORT_H_
#include <string>
#include "tensorflow/compiler/mlir/lite/tools/optimize/reduced_precision_metadata.h"
namespace tflite {
namespace optimize {
inline bool ReadInferenceType(const std::string& metadata, size_t* idx,
                              ReducedPrecisionSupport* mask) {
  if (metadata.substr(*idx, 4) == kTfLiteFloat16String) {
    *idx += 4;
    *mask = *mask | ReducedPrecisionSupport::Float16Inference;
    return true;
  } else if (metadata.substr(*idx, 4) == kTfLiteBfloat16String) {
    *idx += 4;
    *mask = *mask | ReducedPrecisionSupport::Bfloat16Inference;
    return true;
  }
  return false;
}
inline bool ReadAccumulationType(const std::string& metadata, size_t* idx,
                                 ReducedPrecisionSupport* mask) {
  if (metadata.substr(*idx, 4) == kTfLiteFloat16String) {
    *idx += 4;
    *mask = *mask | ReducedPrecisionSupport::Float16Accumulation;
    return true;
  } else if (metadata.substr(*idx, 4) == kTfLiteFloat32String) {
    *idx += 4;
    *mask = *mask | ReducedPrecisionSupport::Float32Accumulation;
    return true;
  }
  return false;
}
inline bool SetMaskFromReducedPrecisionMetadata(const std::string& metadata,
                                                ReducedPrecisionSupport* mask) {
  bool check = true;
  size_t idx = 0;
  ReducedPrecisionSupport rsp = ReducedPrecisionSupport::None;
  do {
    check = ReadInferenceType(metadata, &idx, &rsp);
  } while (check);
  if (idx == 0) {
    return false;
  }
  if (metadata.substr(idx, 3) != kTfLiteAccumulationString) {
    return false;
  }
  idx += std::string(kTfLiteAccumulationString).size();
  if (!ReadAccumulationType(metadata, &idx, &rsp)) {
    return false;
  }
  if (idx != metadata.length()) {
    return false;
  }
  *mask = rsp;
  return true;
}
}  
}  
#endif  