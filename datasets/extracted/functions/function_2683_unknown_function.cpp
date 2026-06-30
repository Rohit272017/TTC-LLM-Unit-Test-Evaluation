#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_F16_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_F16_H_
#include "tensorflow/lite/experimental/shlo/legacy/src/has_keyword.h"
#if defined(__STDCPP_FLOAT16_T__)
#include <stdfloat>
namespace stablehlo {
using F16 = float16_t;
}  
#elif __has_keyword(_Float16)
namespace stablehlo {
using F16 = _Float16;
}  
#else
#error Type F16 is not available
#endif
#endif  