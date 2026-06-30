#ifndef TENSORFLOW_LITE_TYPE_TO_TFLITETYPE_H_
#define TENSORFLOW_LITE_TYPE_TO_TFLITETYPE_H_
#include <complex>
#include <string>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/portable_type_to_tflitetype.h"
namespace tflite {
MATCH_TYPE_AND_TFLITE_TYPE(std::string, kTfLiteString);
MATCH_TYPE_AND_TFLITE_TYPE(std::complex<float>, kTfLiteComplex64);
MATCH_TYPE_AND_TFLITE_TYPE(std::complex<double>, kTfLiteComplex128);
}  
#endif  