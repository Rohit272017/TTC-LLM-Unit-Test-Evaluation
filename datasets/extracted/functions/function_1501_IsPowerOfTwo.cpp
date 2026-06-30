#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <complex>
#include "third_party/fft2d/fft2d.h"
#include "ruy/profiler/instrumentation.h"  
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
namespace tflite {
namespace ops {
namespace builtin {
namespace rfft2d {
using std::complex;
constexpr int kInputTensor = 0;
constexpr int kFftLengthTensor = 1;
constexpr int kOutputTensor = 0;
constexpr int kFftIntegerWorkingAreaTensor = 0;
constexpr int kFftDoubleWorkingAreaTensor = 1;
constexpr int kTensorNotAllocated = -1;
struct OpData {
  int fft_integer_working_area_id = kTensorNotAllocated;
  int fft_double_working_area_id = kTensorNotAllocated;
};
bool IsPowerOfTwo(uint32_t v) { return v && !(v & (v - 1)); }
static TfLiteStatus InitTemporaryTensors(TfLiteContext* context,
                                         TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  if (data->fft_integer_working_area_id != kTensorNotAllocated &&
      data->fft_double_working_area_id != kTensorNotAllocated) {
    return kTfLiteOk;
  }
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(2);
  int first_new_index;
  TF_LITE_ENSURE_STATUS(context->AddTensors(context, 2, &first_new_index));
  node->temporaries->data[kFftIntegerWorkingAreaTensor] = first_new_index;
  data->fft_integer_working_area_id = first_new_index;
  node->temporaries->data[kFftDoubleWorkingAreaTensor] = first_new_index + 1;
  data->fft_double_working_area_id = first_new_index + 1;
  TfLiteTensor* fft_integer_working_area;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, kFftIntegerWorkingAreaTensor,
                                &fft_integer_working_area));
  fft_integer_working_area->type = kTfLiteInt32;
  fft_integer_working_area->allocation_type = kTfLiteArenaRw;
  TfLiteTensor* fft_double_working_area;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, kFftDoubleWorkingAreaTensor,
                                     &fft_double_working_area));
  fft_double_working_area->type = kTfLiteInt64;
  fft_double_working_area->allocation_type = kTfLiteArenaRw;
  return kTfLiteOk;
}
TfLiteStatus ResizeOutputandTemporaryTensors(TfLiteContext* context,
                                             TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const int num_dims = NumDimensions(input);
  TF_LITE_ENSURE(context, num_dims >= 2);
  const TfLiteTensor* fft_length;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kFftLengthTensor, &fft_length));
  const int32_t* fft_length_data = GetTensorData<int32_t>(fft_length);
  TF_LITE_ENSURE(context, IsPowerOfTwo(fft_length_data[0]));
  TF_LITE_ENSURE(context, IsPowerOfTwo(fft_length_data[1]));
  int fft_height, fft_width;
  fft_height = fft_length_data[0];
  fft_width = fft_length_data[1];
  int fft_working_length = std::max(fft_height, fft_width / 2);
  int half_fft_working_length = fft_working_length / 2;
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TfLiteIntArray* output_shape = TfLiteIntArrayCopy(input->dims);
  output_shape->data[num_dims - 2] = fft_length_data[0];
  output_shape->data[num_dims - 1] = fft_length_data[1] / 2 + 1;
  TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, output, output_shape));
  TfLiteTensor* fft_integer_working_area;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, kFftIntegerWorkingAreaTensor,
                                &fft_integer_working_area));
  TfLiteIntArray* fft_integer_working_area_shape = TfLiteIntArrayCreate(1);
  fft_integer_working_area_shape->data[0] =
      2 + static_cast<int>(sqrt(fft_working_length));
  TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, fft_integer_working_area,
                                              fft_integer_working_area_shape));
  TfLiteTensor* fft_double_working_area;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, kFftDoubleWorkingAreaTensor,
                                     &fft_double_working_area));
  TfLiteIntArray* fft_double_working_area_shape = TfLiteIntArrayCreate(1);
  fft_double_working_area_shape->data[0] =
      half_fft_working_length + fft_width / 4;
  TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, fft_double_working_area,
                                              fft_double_working_area_shape));
  return kTfLiteOk;
}
void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new OpData;
  return data;
}
void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TF_LITE_ENSURE(context, NumDimensions(input) >= 2);
  if (input->type != kTfLiteFloat32) {
    TF_LITE_KERNEL_LOG(context,
                       "Type '%s' for input is not supported by rfft2d.",
                       TfLiteTypeGetName(input->type));
    return kTfLiteError;
  }
  const TfLiteTensor* fft_length;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kFftLengthTensor, &fft_length));
  const RuntimeShape fft_length_shape = GetTensorShape(fft_length);
  TF_LITE_ENSURE_EQ(context, NumDimensions(fft_length), 1);
  TF_LITE_ENSURE_EQ(context, fft_length_shape.Dims(0), 2);
  if (fft_length->type != kTfLiteInt32) {
    TF_LITE_KERNEL_LOG(context,
                       "Type '%s' for fft_length is not supported by rfft2d.",
                       TfLiteTypeGetName(fft_length->type));
    return kTfLiteError;
  }
  TF_LITE_ENSURE_STATUS(InitTemporaryTensors(context, node));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  output->type = kTfLiteComplex64;
  if (!IsConstantOrPersistentTensor(fft_length)) {
    TfLiteTensor* fft_integer_working_area;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, kFftIntegerWorkingAreaTensor,
                                  &fft_integer_working_area));
    TfLiteTensor* fft_double_working_area;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, kFftDoubleWorkingAreaTensor,
                                  &fft_double_working_area));
    SetTensorToDynamic(fft_integer_working_area);
    SetTensorToDynamic(fft_double_working_area);
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }
  TF_LITE_ENSURE_STATUS(ResizeOutputandTemporaryTensors(context, node));
  return kTfLiteOk;
}
void Rfft2dReorder(int fft_height, int fft_width, double** fft_input_output) {
  int fft_height_half;
  ruy::profiler::ScopeLabel label("Rfft2dReorder");
  double real, img;
  fft_height_half = fft_height >> 1;
  for (int i = fft_height_half + 1; i < fft_height; ++i) {
    real = fft_input_output[i][0];
    img = fft_input_output[i][1];
    fft_input_output[i][fft_width] = img;
    fft_input_output[i][fft_width + 1] = real;
    fft_input_output[fft_height - i][fft_width] = img;
    fft_input_output[fft_height - i][fft_width + 1] = -real;
    fft_input_output[i][0] = fft_input_output[fft_height - i][0];
    fft_input_output[i][1] = -fft_input_output[fft_height - i][1];
  }
  double temp = fft_input_output[0][1];
  fft_input_output[0][fft_width + 1] = 0;
  fft_input_output[0][1] = 0;
  fft_input_output[fft_height_half][fft_width] =
      fft_input_output[fft_height_half][1];
  fft_input_output[fft_height_half][fft_width + 1] = 0;
  fft_input_output[fft_height_half][1] = 0;
  fft_input_output[0][fft_width] = temp;
  for (int i = 0; i < fft_height; ++i) {
    for (int j = 1; j < fft_width + 2; j += 2) {
      fft_input_output[i][j] = -fft_input_output[i][j];
    }
  }
}
void Rfft2dImpl(int fft_height, int fft_width, double** fft_input_output,
                int* fft_integer_working_area_data,
                double* fft_double_working_area_data) {
  ruy::profiler::ScopeLabel label("Rfft2dImpl");
  double* fft_dynamic_working_area = nullptr;
  const int kForwardFft = 1;
  rdft2d(fft_height, fft_width, kForwardFft, fft_input_output,
         fft_dynamic_working_area, fft_integer_working_area_data,
         fft_double_working_area_data);
  Rfft2dReorder(fft_height, fft_width, fft_input_output);
}
void PrepareInputBuffer(const float* input_data, int input_height,
                        int input_width, int fft_height, int fft_width,
                        double** fft_input_output) {
  int valid_input_height = std::min(input_height, fft_height);
  int valid_input_width = std::min(input_width, fft_width);
  for (int i = 0; i < valid_input_height; ++i) {
    int in_pos = i * input_width;
    for (int j = 0; j < valid_input_width; ++j) {
      fft_input_output[i][j] = input_data[in_pos++];
    }
    for (int j = valid_input_width; j < fft_width + 2; ++j) {
      fft_input_output[i][j] = 0;
    }
  }
  for (int i = valid_input_height; i < fft_height; ++i) {
    for (int j = 0; j < fft_width + 2; ++j) {
      fft_input_output[i][j] = 0;
    }
  }
}
void PrepareOutputBuffer(complex<float>* output_data, int fft_height,
                         int fft_width, double** fft_input_output) {
  int cnt = 0;
  for (int i = 0; i < fft_height; ++i) {
    for (int j = 0; j < fft_width / 2 + 1; ++j) {
      output_data[cnt++] = complex<float>(fft_input_output[i][j * 2],
                                          fft_input_output[i][j * 2 + 1]);
    }
  }
}
TfLiteStatus Rfft2dHelper(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const float* input_data = GetTensorData<float>(input);
  const TfLiteTensor* fft_length;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kFftLengthTensor, &fft_length));
  const int32_t* fft_length_data = GetTensorData<int32_t>(fft_length);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  complex<float>* output_data = GetTensorData<complex<float>>(output);
  int fft_height, fft_width;
  fft_height = fft_length_data[0];
  fft_width = fft_length_data[1];
  const RuntimeShape input_shape = GetTensorShape(input);
  const int input_dims_count = input_shape.DimensionsCount();
  const auto* input_dims_data = input_shape.DimsData();
  int num_slices = 1;
  for (int i = 0; i < input_dims_count - 2; ++i) {
    num_slices *= input_dims_data[i];
  }
  int input_height = input_dims_data[input_dims_count - 2];
  int input_width = input_dims_data[input_dims_count - 1];
  int input_slice_size = input_height * input_width;
  int output_slice_size = fft_height * (fft_width / 2 + 1);
  double** fft_input_output = new double*[fft_height];
  for (int i = 0; i < fft_height; ++i) {
    fft_input_output[i] = new double[fft_width + 2];
  }
  TfLiteTensor* fft_integer_working_area;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, kFftIntegerWorkingAreaTensor,
                                &fft_integer_working_area));
  int* fft_integer_working_area_data =
      GetTensorData<int>(fft_integer_working_area);
  TfLiteTensor* fft_double_working_area;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, kFftDoubleWorkingAreaTensor,
                                     &fft_double_working_area));
  double* fft_double_working_area_data = reinterpret_cast<double*>(
      GetTensorData<int64_t>(fft_double_working_area));
  for (int i = 0; i < num_slices; ++i) {
    PrepareInputBuffer(input_data, input_height, input_width, fft_height,
                       fft_width, fft_input_output);
    memset(fft_integer_working_area_data, 0, fft_integer_working_area->bytes);
    memset(fft_double_working_area_data, 0, fft_double_working_area->bytes);
    Rfft2dImpl(fft_height, fft_width, fft_input_output,
               fft_integer_working_area_data, fft_double_working_area_data);
    PrepareOutputBuffer(output_data, fft_height, fft_width, fft_input_output);
    input_data += input_slice_size;
    output_data += output_slice_size;
  }
  for (int i = 0; i < fft_height; ++i) {
    delete[] fft_input_output[i];
  }
  delete[] fft_input_output;
  return kTfLiteOk;
}
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* fft_length;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kFftLengthTensor, &fft_length));
  const int32_t* fft_length_data = GetTensorData<int32_t>(fft_length);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  if (output->type != kTfLiteComplex64) {
    TF_LITE_KERNEL_LOG(context,
                       "Type '%s' for output is not supported by rfft2d.",
                       TfLiteTypeGetName(output->type));
    return kTfLiteError;
  }
  if (!IsConstantTensor(fft_length)) {
    TF_LITE_ENSURE_STATUS(ResizeOutputandTemporaryTensors(context, node));
  } else {
    int num_dims_output = NumDimensions(output);
    const RuntimeShape output_shape = GetTensorShape(output);
    TF_LITE_ENSURE_EQ(context, num_dims_output, NumDimensions(input));
    TF_LITE_ENSURE(context, num_dims_output >= 2);
    TF_LITE_ENSURE_EQ(context, output_shape.Dims(num_dims_output - 2),
                      fft_length_data[0]);
    TF_LITE_ENSURE_EQ(context, output_shape.Dims(num_dims_output - 1),
                      fft_length_data[1] / 2 + 1);
  }
  return Rfft2dHelper(context, node);
}
}  
TfLiteRegistration* Register_RFFT2D() {
  static TfLiteRegistration r = {rfft2d::Init, rfft2d::Free, rfft2d::Prepare,
                                 rfft2d::Eval};
  return &r;
}
}  
}  
}  