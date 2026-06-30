#include <stdint.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
namespace tflite {
namespace ops {
namespace builtin {
namespace segment_sum {
static const int kInputDataTensor = 0;
static const int kInputSegmentIdsTensor = 1;
static const int kOutputTensor = 0;
TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                const TfLiteTensor* data,
                                const TfLiteTensor* segment_ids,
                                TfLiteTensor* output) {
  const int segment_id_size = segment_ids->dims->data[0];
  TF_LITE_ENSURE_EQ(context, segment_id_size, data->dims->data[0]);
  int previous_segment_id = -1;
  for (int i = 0; i < segment_id_size; i++) {
    const int current_segment_id = GetTensorData<int32_t>(segment_ids)[i];
    if (i == 0) {
      TF_LITE_ENSURE_EQ(context, current_segment_id, 0);
    } else {
      int delta = current_segment_id - previous_segment_id;
      TF_LITE_ENSURE(context, delta == 0 || delta == 1);
    }
    previous_segment_id = current_segment_id;
  }
  const int max_index = previous_segment_id;
  const int data_rank = NumDimensions(data);
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(NumDimensions(data));
  output_shape->data[0] = max_index + 1;
  for (int i = 1; i < data_rank; ++i) {
    output_shape->data[i] = data->dims->data[i];
  }
  return context->ResizeTensor(context, output, output_shape);
}
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* data;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputDataTensor, &data));
  const TfLiteTensor* segment_ids;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputSegmentIdsTensor,
                                          &segment_ids));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TF_LITE_ENSURE(context,
                 data->type == kTfLiteInt32 || data->type == kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, segment_ids->type, kTfLiteInt32);
  if (!IsConstantOrPersistentTensor(data) ||
      !IsConstantOrPersistentTensor(segment_ids)) {
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }
  return ResizeOutputTensor(context, data, segment_ids, output);
}
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* data;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputDataTensor, &data));
  const TfLiteTensor* segment_ids;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputSegmentIdsTensor,
                                          &segment_ids));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context,
                      ResizeOutputTensor(context, data, segment_ids, output));
  }
#define TF_LITE_SEGMENT_SUM(dtype)                                      \
  reference_ops::SegmentSum<dtype>(                                     \
      GetTensorShape(data), GetTensorData<dtype>(data),                 \
      GetTensorShape(segment_ids), GetTensorData<int32_t>(segment_ids), \
      GetTensorShape(output), GetTensorData<dtype>(output));
  switch (data->type) {
    case kTfLiteInt32:
      TF_LITE_SEGMENT_SUM(int32_t);
      break;
    case kTfLiteFloat32:
      TF_LITE_SEGMENT_SUM(float);
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Currently SegmentSum doesn't support type: %s",
                         TfLiteTypeGetName(data->type));
      return kTfLiteError;
  }
#undef TF_LITE_SEGMENT_SUM
  return kTfLiteOk;
}
}  
TfLiteRegistration* Register_SEGMENT_SUM() {
  static TfLiteRegistration r = {nullptr, nullptr, segment_sum::Prepare,
                                 segment_sum::Eval};
  return &r;
}
}  
}  
}  