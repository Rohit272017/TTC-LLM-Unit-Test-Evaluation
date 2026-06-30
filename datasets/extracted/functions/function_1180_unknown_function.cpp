#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_FLEXBUFFERS_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_FLEXBUFFERS_UTIL_H_
#include "flatbuffers/base.h"  
#include "flatbuffers/flexbuffers.h"  
namespace tflite::xnnpack {
struct FloatPointer {
  const float* ptr = nullptr;
};
}  
namespace flexbuffers {
template <>
tflite::xnnpack::FloatPointer inline flexbuffers::Reference::As<
    tflite::xnnpack::FloatPointer>() const {
#if !FLATBUFFERS_LITTLEENDIAN
  return nullptr;
#else
  return {IsFloat() ? reinterpret_cast<const float*>(data_) : nullptr};
#endif
}
}  
#endif  