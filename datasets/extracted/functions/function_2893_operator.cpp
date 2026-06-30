#include "tensorflow/lite/array.h"
namespace tflite {
namespace array_internal {
void TfLiteArrayDeleter::operator()(TfLiteIntArray* a) {
  if (a) {
    TfLiteIntArrayFree(a);
  }
}
void TfLiteArrayDeleter::operator()(TfLiteFloatArray* a) {
  if (a) {
    TfLiteFloatArrayFree(a);
  }
}
}  
}  