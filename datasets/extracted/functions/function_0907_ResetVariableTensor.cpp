#include "tensorflow/lite/core/api/tensor_utils.h"
#include <string.h>
#include "tensorflow/lite/core/c/common.h"
namespace tflite {
TfLiteStatus ResetVariableTensor(TfLiteTensor* tensor) {
  if (!tensor->is_variable) {
    return kTfLiteOk;
  }
  int value = 0;
  if (tensor->type == kTfLiteInt8) {
    value = tensor->params.zero_point;
  }
#if __ANDROID__ || defined(__x86_64__) || defined(__i386__) || \
    defined(__i386) || defined(__x86__) || defined(__X86__) || \
    defined(_X86_) || defined(_M_IX86) || defined(_M_X64)
  memset(tensor->data.raw, value, tensor->bytes);
#else
  char* raw_ptr = tensor->data.raw;
  for (size_t i = 0; i < tensor->bytes; ++i) {
    *raw_ptr = value;
    raw_ptr++;
  }
#endif
  return kTfLiteOk;
}
}  