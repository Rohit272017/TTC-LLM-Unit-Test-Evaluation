#include "tensorflow/lite/kernels/internal/reference/non_max_suppression.h"
#include <initializer_list>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
namespace tflite {
namespace ops {
namespace builtin {
namespace non_max_suppression {
constexpr int kInputTensorBoxes = 0;
constexpr int kInputTensorScores = 1;
constexpr int kInputTensorMaxOutputSize = 2;
constexpr int kInputTensorIouThreshold = 3;
constexpr int kInputTensorScoreThreshold = 4;
constexpr int kInputTensorSigma = 5;
constexpr int kNMSOutputTensorSelectedIndices = 0;
constexpr int kNMSOutputTensorNumSelectedIndices = 1;
constexpr int kSoftNMSOutputTensorSelectedIndices = 0;
constexpr int kSoftNMSOutputTensorSelectedScores = 1;
constexpr int kSoftNMSOutputTensorNumSelectedIndices = 2;
TfLiteStatus SetTensorSizes(TfLiteContext* context, TfLiteTensor* tensor,
                            std::initializer_list<int> values) {
  TfLiteIntArray* size = TfLiteIntArrayCreate(values.size());
  int index = 0;
  for (const auto& v : values) {
    size->data[index++] = v;
  }
  return context->ResizeTensor(context, tensor, size);
}
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const int num_inputs = NumInputs(node);
  const bool is_soft_nms = num_inputs == 6;
  if (num_inputs != 5 && num_inputs != 6) {
    TF_LITE_KERNEL_LOG(context, "Found NMS op with invalid num inputs: %d",
                       NumInputs(node));
    return kTfLiteError;
  }
  const TfLiteTensor* input_boxes;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kInputTensorBoxes, &input_boxes));
  TF_LITE_ENSURE_EQ(context, input_boxes->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_boxes), 2);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(input_boxes, 1), 4);
  const int num_boxes = SizeOfDimension(input_boxes, 0);
  const TfLiteTensor* input_scores;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kInputTensorScores, &input_scores));
  TF_LITE_ENSURE_EQ(context, input_scores->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_scores), 1);
  TF_LITE_ENSURE_EQ(context, num_boxes, SizeOfDimension(input_scores, 0));
  const TfLiteTensor* input_max_output_size;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorMaxOutputSize,
                                 &input_max_output_size));
  TF_LITE_ENSURE_EQ(context, input_max_output_size->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_max_output_size), 0);
  const bool is_max_output_size_const =
      IsConstantOrPersistentTensor(input_max_output_size);
  int max_output_size_value = 0;
  if (is_max_output_size_const) {
    max_output_size_value = *GetTensorData<int>(input_max_output_size);
    TF_LITE_ENSURE(context, (max_output_size_value >= 0));
  }
  const TfLiteTensor* input_iou_threshold;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorIouThreshold,
                                 &input_iou_threshold));
  TF_LITE_ENSURE_EQ(context, input_iou_threshold->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_iou_threshold), 0);
  const TfLiteTensor* input_score_threshold;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorScoreThreshold,
                                 &input_score_threshold));
  TF_LITE_ENSURE_EQ(context, input_iou_threshold->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_score_threshold), 0);
  if (is_soft_nms) {
    const TfLiteTensor* input_sigma;
    TF_LITE_ENSURE_OK(
        context, GetInputSafe(context, node, kInputTensorSigma, &input_sigma));
    TF_LITE_ENSURE_EQ(context, input_sigma->type, kTfLiteFloat32);
    TF_LITE_ENSURE_EQ(context, NumDimensions(input_sigma), 0);
    TF_LITE_ENSURE_EQ(context, NumOutputs(node), 3);
    TfLiteTensor* output_selected_indices;
    TF_LITE_ENSURE_OK(
        context,
        GetOutputSafe(context, node, kSoftNMSOutputTensorSelectedIndices,
                      &output_selected_indices));
    output_selected_indices->type = kTfLiteInt32;
    TfLiteTensor* output_selected_scores;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node,
                                             kSoftNMSOutputTensorSelectedScores,
                                             &output_selected_scores));
    output_selected_scores->type = kTfLiteFloat32;
    TfLiteTensor* output_num_selected_indices;
    TF_LITE_ENSURE_OK(
        context,
        GetOutputSafe(context, node, kSoftNMSOutputTensorNumSelectedIndices,
                      &output_num_selected_indices));
    output_num_selected_indices->type = kTfLiteInt32;
    SetTensorSizes(context, output_num_selected_indices, {});
    if (is_max_output_size_const) {
      SetTensorSizes(context, output_selected_indices, {max_output_size_value});
      SetTensorSizes(context, output_selected_scores, {max_output_size_value});
    } else {
      SetTensorToDynamic(output_selected_indices);
      SetTensorToDynamic(output_selected_scores);
    }
  } else {
    TF_LITE_ENSURE_EQ(context, NumOutputs(node), 2);
    TfLiteTensor* output_selected_indices;
    TF_LITE_ENSURE_OK(
        context, GetOutputSafe(context, node, kNMSOutputTensorSelectedIndices,
                               &output_selected_indices));
    output_selected_indices->type = kTfLiteInt32;
    TfLiteTensor* output_num_selected_indices;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node,
                                             kNMSOutputTensorNumSelectedIndices,
                                             &output_num_selected_indices));
    output_num_selected_indices->type = kTfLiteInt32;
    SetTensorSizes(context, output_num_selected_indices, {});
    if (is_max_output_size_const) {
      SetTensorSizes(context, output_selected_indices, {max_output_size_value});
    } else {
      SetTensorToDynamic(output_selected_indices);
    }
  }
  return kTfLiteOk;
}
void ResetUnusedElementsToZeroes(const int max_output_size,
                                 const int num_selected_indices,
                                 int* selected_indices,
                                 float* selected_scores) {
  for (int i = num_selected_indices; i < max_output_size; ++i) {
    selected_indices[i] = 0;
    if (selected_scores) {
      selected_scores[i] = 0.0;
    }
  }
}
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const bool is_soft_nms = NumInputs(node) == 6;
  const TfLiteTensor* input_boxes;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kInputTensorBoxes, &input_boxes));
  const int num_boxes = SizeOfDimension(input_boxes, 0);
  const TfLiteTensor* input_scores;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kInputTensorScores, &input_scores));
  const TfLiteTensor* input_max_output_size;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorMaxOutputSize,
                                 &input_max_output_size));
  const int max_output_size_value = *GetTensorData<int>(input_max_output_size);
  TF_LITE_ENSURE(context, (max_output_size_value >= 0));
  const bool is_max_output_size_const =
      IsConstantOrPersistentTensor(input_max_output_size);
  const TfLiteTensor* input_iou_threshold;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorIouThreshold,
                                 &input_iou_threshold));
  const float iou_threshold = *GetTensorData<float>(input_iou_threshold);
  const TfLiteTensor* input_score_threshold;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorScoreThreshold,
                                 &input_score_threshold));
  const float score_threshold = *GetTensorData<float>(input_score_threshold);
  TfLiteTensor* output_selected_indices = nullptr;
  TfLiteTensor* output_selected_scores = nullptr;
  TfLiteTensor* output_num_selected_indices = nullptr;
  if (is_soft_nms) {
    const TfLiteTensor* input_sigma;
    TF_LITE_ENSURE_OK(
        context, GetInputSafe(context, node, kInputTensorSigma, &input_sigma));
    const float soft_nms_sigma = *GetTensorData<float>(input_sigma);
    if (soft_nms_sigma < 0) {
      TF_LITE_KERNEL_LOG(context, "Invalid sigma value for soft NMS: %f",
                         soft_nms_sigma);
      return kTfLiteError;
    }
    TF_LITE_ENSURE_OK(
        context,
        GetOutputSafe(context, node, kSoftNMSOutputTensorSelectedIndices,
                      &output_selected_indices));
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node,
                                             kSoftNMSOutputTensorSelectedScores,
                                             &output_selected_scores));
    TF_LITE_ENSURE_OK(
        context,
        GetOutputSafe(context, node, kSoftNMSOutputTensorNumSelectedIndices,
                      &output_num_selected_indices));
    if (!is_max_output_size_const) {
      SetTensorSizes(context, output_selected_indices, {max_output_size_value});
      SetTensorSizes(context, output_selected_scores, {max_output_size_value});
    }
    reference_ops::NonMaxSuppression(
        input_boxes->data.f, num_boxes, input_scores->data.f,
        max_output_size_value, iou_threshold, score_threshold, soft_nms_sigma,
        output_selected_indices->data.i32, output_selected_scores->data.f,
        output_num_selected_indices->data.i32);
    ResetUnusedElementsToZeroes(
        max_output_size_value, *output_num_selected_indices->data.i32,
        output_selected_indices->data.i32, output_selected_scores->data.f);
  } else {
    TF_LITE_ENSURE_OK(
        context, GetOutputSafe(context, node, kNMSOutputTensorSelectedIndices,
                               &output_selected_indices));
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node,
                                             kNMSOutputTensorNumSelectedIndices,
                                             &output_num_selected_indices));
    if (!is_max_output_size_const) {
      SetTensorSizes(context, output_selected_indices, {max_output_size_value});
    }
    reference_ops::NonMaxSuppression(
        input_boxes->data.f, num_boxes, input_scores->data.f,
        max_output_size_value, iou_threshold, score_threshold,  0.0,
        output_selected_indices->data.i32,  nullptr,
        output_num_selected_indices->data.i32);
    ResetUnusedElementsToZeroes(max_output_size_value,
                                *output_num_selected_indices->data.i32,
                                output_selected_indices->data.i32, nullptr);
  }
  return kTfLiteOk;
}
}  
TfLiteRegistration* Register_NON_MAX_SUPPRESSION_V4() {
  static TfLiteRegistration r = {nullptr, nullptr, non_max_suppression::Prepare,
                                 non_max_suppression::Eval};
  return &r;
}
TfLiteRegistration* Register_NON_MAX_SUPPRESSION_V5() {
  static TfLiteRegistration r = {nullptr, nullptr, non_max_suppression::Prepare,
                                 non_max_suppression::Eval};
  return &r;
}
}  
}  
}  