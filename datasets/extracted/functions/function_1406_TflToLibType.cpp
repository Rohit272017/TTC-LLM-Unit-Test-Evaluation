#include "tensorflow/lite/experimental/ml_adjacent/tflite/tfl_tensor_ref.h"
#include <cstddef>
#include <vector>
#include "tensorflow/lite/array.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/experimental/ml_adjacent/lib.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"
namespace ml_adj {
namespace data {
using ::tflite::BuildTfLiteArray;
using ::tflite::TfLiteArrayUniquePtr;
using ::tflite::TfLiteTypeGetSize;
namespace {
etype_t TflToLibType(const TfLiteType tfl_type) {
  switch (tfl_type) {
    case kTfLiteFloat32:
      return etype_t::f32;
    case kTfLiteInt32:
      return etype_t::i32;
    case kTfLiteFloat64:
      return etype_t::f64;
    default:
      return etype_t::i32;
  }
}
}  
TflTensorRef::TflTensorRef(const TfLiteTensor* tfl_tensor)
    : DataRef(TflToLibType(tfl_tensor->type)), tfl_tensor_(tfl_tensor) {
  dims_.assign(tfl_tensor->dims->data,
               tfl_tensor->dims->data + tfl_tensor->dims->size);
}
const void* TflTensorRef::Data() const { return tfl_tensor_->data.data; }
ind_t TflTensorRef::NumElements() const {
  return tfl_tensor_->bytes / TfLiteTypeGetSize(tfl_tensor_->type);
}
size_t TflTensorRef::Bytes() const { return tfl_tensor_->bytes; }
MutableTflTensorRef::MutableTflTensorRef(TfLiteTensor* tfl_tensor,
                                         TfLiteContext* tfl_ctx)
    : MutableDataRef(TflToLibType(tfl_tensor->type)),
      tfl_tensor_(tfl_tensor),
      tfl_ctx_(tfl_ctx) {
  dims_.assign(tfl_tensor->dims->data,
               tfl_tensor->dims->data + tfl_tensor->dims->size);
}
void MutableTflTensorRef::Resize(dims_t&& dims) {
  TfLiteArrayUniquePtr<int> arr =
      BuildTfLiteArray(std::vector<int>(dims.begin(), dims.end()));
  TFLITE_CHECK_EQ(tfl_ctx_->ResizeTensor(tfl_ctx_, tfl_tensor_, arr.release()),
                  kTfLiteOk);
  dims_ = dims;
}
const void* MutableTflTensorRef::Data() const { return tfl_tensor_->data.data; }
ind_t MutableTflTensorRef::NumElements() const {
  return tfl_tensor_->bytes / TfLiteTypeGetSize(tfl_tensor_->type);
}
size_t MutableTflTensorRef::Bytes() const { return tfl_tensor_->bytes; }
void* MutableTflTensorRef::Data() { return tfl_tensor_->data.data; }
}  
}  