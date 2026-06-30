#include <stdint.h>
#include <algorithm>
#include <functional>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
namespace tflite {
namespace ops {
namespace builtin {
namespace unsorted_segment {
enum SegmentType {
  kSegmentMax,
  kSegmentMin,
  kSegmentProd,
  kSegmentSum,
};
static const int kInputDataTensor = 0;
static const int kInputSegmentIdsTensor = 1;
static const int kInputNumSegmentsTensor = 2;
static const int kOutputTensor = 0;
TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                const TfLiteTensor* data,
                                const TfLiteTensor* segment_ids,
                                const TfLiteTensor* num_segments,
                                TfLiteTensor* output) {
  const int segment_ids_rank = NumDimensions(segment_ids);
  const int data_rank = NumDimensions(data);
  TF_LITE_ENSURE(context, segment_ids_rank <= data_rank);
  for (int i = 0; i < segment_ids_rank; ++i) {
    TF_LITE_ENSURE_EQ(context, segment_ids->dims->data[i], data->dims->data[i]);
  }
  TF_LITE_ENSURE(context, (num_segments->dims->size == 1 &&
                           num_segments->dims->data[0] == 1) ||
                              num_segments->dims->size == 0);
  int32_t num_segments_ = GetTensorData<int32_t>(num_segments)[0];
  const int num_segment_ids = NumElements(segment_ids);
  int max_index = -1;
  for (int i = 0; i < num_segment_ids; i++) {
    max_index = std::max(GetTensorData<int32_t>(segment_ids)[i], max_index);
  }
  TF_LITE_ENSURE(context, max_index < num_segments_);
  const int output_rank = data_rank - segment_ids_rank + 1;
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(output_rank);
  output_shape->data[0] = num_segments_;
  for (int i = segment_ids_rank; i < data_rank; ++i) {
    output_shape->data[i - segment_ids_rank + 1] = data->dims->data[i];
  }
  return context->ResizeTensor(context, output, output_shape);
}
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* data;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputDataTensor, &data));
  const TfLiteTensor* segment_ids;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputSegmentIdsTensor,
                                          &segment_ids));
  const TfLiteTensor* num_segments;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, kInputNumSegmentsTensor, &num_segments));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TF_LITE_ENSURE(context,
                 data->type == kTfLiteInt32 || data->type == kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, segment_ids->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, num_segments->type, kTfLiteInt32);
  if (IsDynamicTensor(data) || !IsConstantOrPersistentTensor(segment_ids) ||
      !IsConstantOrPersistentTensor(num_segments)) {
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }
  return ResizeOutputTensor(context, data, segment_ids, num_segments, output);
}
template <typename T>
struct SegmenMax {
  inline T operator()(const T& a, const T& b) const { return std::max(a, b); }
  static constexpr T kInitialValue = std::numeric_limits<T>::lowest();
};
template <typename T>
struct SegmenMin {
  inline T operator()(const T& a, const T& b) const { return std::min(a, b); }
  static constexpr T kInitialValue = std::numeric_limits<T>::max();
};
template <typename T>
struct SegmenProd {
  inline T operator()(const T& a, const T& b) const { return a * b; }
  static constexpr T kInitialValue = T(1);
};
template <typename T>
struct SegmenSum {
  inline T operator()(const T& a, const T& b) const { return a + b; }
  static constexpr T kInitialValue = T(0);
};
template <typename T>
TfLiteStatus EvalType(TfLiteContext* context, const RuntimeShape& input_shape,
                      const T* input_data,
                      const RuntimeShape& segment_ids_shape,
                      const int32_t* segment_ids_data,
                      const RuntimeShape& output_shape, T* output_data,
                      SegmentType segment_type) {
  switch (segment_type) {
    case kSegmentProd:
      reference_ops::UnsortedSegmentRef<T, SegmenProd>(
          input_shape, input_data, segment_ids_shape, segment_ids_data,
          output_shape, output_data);
      break;
    case kSegmentMax:
      reference_ops::UnsortedSegmentRef<T, SegmenMax>(
          input_shape, input_data, segment_ids_shape, segment_ids_data,
          output_shape, output_data);
      break;
    case kSegmentSum:
      reference_ops::UnsortedSegmentRef<T, SegmenSum>(
          input_shape, input_data, segment_ids_shape, segment_ids_data,
          output_shape, output_data);
      break;
    case kSegmentMin:
      reference_ops::UnsortedSegmentRef<T, SegmenMin>(
          input_shape, input_data, segment_ids_shape, segment_ids_data,
          output_shape, output_data);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Not recognized segment type: %d",
                         segment_type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}
TfLiteStatus EvalGeneric(TfLiteContext* context, TfLiteNode* node,
                         SegmentType segment_type) {
  const TfLiteTensor* data;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputDataTensor, &data));
  const TfLiteTensor* segment_ids;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputSegmentIdsTensor,
                                          &segment_ids));
  const TfLiteTensor* num_segments;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, kInputNumSegmentsTensor, &num_segments));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, data, segment_ids,
                                                  num_segments, output));
  }
  TF_LITE_ENSURE_EQ(context, GetTensorShape(data).Dims(0),
                    GetTensorShape(segment_ids).Dims(0));
#define TF_LITE_UNSORTED_SEGMENT(dtype)                                        \
  EvalType<dtype>(context, GetTensorShape(data), GetTensorData<dtype>(data),   \
                  GetTensorShape(segment_ids),                                 \
                  GetTensorData<int32_t>(segment_ids), GetTensorShape(output), \
                  GetTensorData<dtype>(output), segment_type);
  switch (data->type) {
    case kTfLiteInt32:
      TF_LITE_UNSORTED_SEGMENT(int32_t);
      break;
    case kTfLiteFloat32:
      TF_LITE_UNSORTED_SEGMENT(float);
      break;
    default:
      TF_LITE_KERNEL_LOG(
          context, "Currently UnsortedSegment doesn't support data type: %s",
          TfLiteTypeGetName(data->type));
      return kTfLiteError;
  }
#undef TF_LITE_UNSORTED_SEGMENT
  return kTfLiteOk;
}
TfLiteStatus EvalProd(TfLiteContext* context, TfLiteNode* node) {
  return EvalGeneric(context, node, kSegmentProd);
}
TfLiteStatus EvalMax(TfLiteContext* context, TfLiteNode* node) {
  return EvalGeneric(context, node, kSegmentMax);
}
TfLiteStatus EvalSum(TfLiteContext* context, TfLiteNode* node) {
  return EvalGeneric(context, node, kSegmentSum);
}
TfLiteStatus EvalMin(TfLiteContext* context, TfLiteNode* node) {
  return EvalGeneric(context, node, kSegmentMin);
}
}  
TfLiteRegistration* Register_UNSORTED_SEGMENT_PROD() {
  static TfLiteRegistration r = {nullptr, nullptr, unsorted_segment::Prepare,
                                 unsorted_segment::EvalProd};
  return &r;
}
TfLiteRegistration* Register_UNSORTED_SEGMENT_MAX() {
  static TfLiteRegistration r = {nullptr, nullptr, unsorted_segment::Prepare,
                                 unsorted_segment::EvalMax};
  return &r;
}
TfLiteRegistration* Register_UNSORTED_SEGMENT_SUM() {
  static TfLiteRegistration r = {nullptr, nullptr, unsorted_segment::Prepare,
                                 unsorted_segment::EvalSum};
  return &r;
}
TfLiteRegistration* Register_UNSORTED_SEGMENT_MIN() {
  static TfLiteRegistration r = {nullptr, nullptr, unsorted_segment::Prepare,
                                 unsorted_segment::EvalMin};
  return &r;
}
}  
}  
}  