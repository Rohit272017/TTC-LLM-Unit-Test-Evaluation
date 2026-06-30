#include <stdint.h>
#include <cstring>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
namespace tflite {
namespace ops {
namespace builtin {
namespace embedding_lookup {
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* lookup;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &lookup));
  TF_LITE_ENSURE_EQ(context, NumDimensions(lookup), 1);
  TF_LITE_ENSURE_EQ(context, lookup->type, kTfLiteInt32);
  const TfLiteTensor* value;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &value));
  TF_LITE_ENSURE(context, NumDimensions(value) >= 2);
  if (value->quantization.type == kTfLiteAffineQuantization) {
    const auto qparams = static_cast<const TfLiteAffineQuantization*>(
        value->quantization.params);
    TF_LITE_ENSURE(context, qparams->scale != nullptr);
    TF_LITE_ENSURE(context, qparams->zero_point != nullptr);
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
    if ((value->type == kTfLiteUInt8 || value->type == kTfLiteInt8 ||
         value->type == kTfLiteInt4) &&
        (output->type == kTfLiteFloat32)) {
      TF_LITE_ENSURE(context, qparams->zero_point->data[0] == 0);
    }
    if (qparams->scale->size > 1 || qparams->zero_point->size > 1) {
      TF_LITE_ENSURE(context, value->type == kTfLiteUInt8 ||
                                  value->type == kTfLiteInt8 ||
                                  value->type == kTfLiteInt4);
      TF_LITE_ENSURE(context, output->type == kTfLiteFloat32);
      TF_LITE_ENSURE(context, qparams->quantized_dimension == 0);
      const int row_size = SizeOfDimension(value, 0);
      TF_LITE_ENSURE(context, qparams->scale->size == row_size);
      TF_LITE_ENSURE(context, qparams->zero_point->size == row_size);
    }
  }
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(NumDimensions(value));
  output_size->data[0] = SizeOfDimension(lookup, 0);
  output_size->data[1] = SizeOfDimension(value, 1);
  for (int i = 2; i < NumDimensions(value); i++) {
    output_size->data[i] = SizeOfDimension(value, i);
  }
  return context->ResizeTensor(context, output, output_size);
}
TfLiteStatus EvalSimple(TfLiteContext* context, TfLiteNode* node,
                        const TfLiteTensor* lookup, const TfLiteTensor* value,
                        TfLiteTensor* output) {
  const int row_size = SizeOfDimension(value, 0);
  if (row_size == 0) {
    return kTfLiteOk;
  }
  const int64_t row_bytes = value->bytes / row_size;
  char* output_raw = GetTensorData<char>(output);
  const char* value_raw = GetTensorData<char>(value);
  const int32_t* lookup_data = GetTensorData<int32_t>(lookup);
  for (int i = 0; i < SizeOfDimension(lookup, 0); i++) {
    int64_t idx = lookup_data[i];
    if (idx >= row_size || idx < 0) {
      TF_LITE_KERNEL_LOG(context,
                         "Embedding Lookup: index out of bounds. "
                         "Got %d, and bounds are [0, %d]",
                         idx, row_size - 1);
      return kTfLiteError;
    } else {
      std::memcpy(output_raw + i * row_bytes, value_raw + idx * row_bytes,
                  row_bytes);
    }
  }
  return kTfLiteOk;
}
TfLiteStatus EvalHybrid(TfLiteContext* context, TfLiteNode* node,
                        const TfLiteTensor* lookup, const TfLiteTensor* value,
                        TfLiteTensor* output) {
  const int row_size = SizeOfDimension(value, 0);
  int col_size = 1;
  for (int i = 1; i < NumDimensions(value); i++) {
    col_size *= SizeOfDimension(value, i);
  }
  float* output_ptr = GetTensorData<float>(output);
  const int8_t* value_ptr = GetTensorData<int8_t>(value);
  const int32_t* lookup_data = GetTensorData<int32_t>(lookup);
  for (int i = 0; i < SizeOfDimension(lookup, 0); i++) {
    int idx = lookup_data[i];
    if (idx >= row_size || idx < 0) {
      TF_LITE_KERNEL_LOG(context,
                         "Embedding Lookup: index out of bounds. "
                         "Got %d, and bounds are [0, %d]",
                         idx, row_size - 1);
      return kTfLiteError;
    } else {
      double scaling_factor = value->params.scale;
      if (value->quantization.type == kTfLiteAffineQuantization) {
        const auto qparams = static_cast<const TfLiteAffineQuantization*>(
            value->quantization.params);
        if (qparams->scale->size > 1) {
          scaling_factor = qparams->scale->data[idx];
        }
      }
      if (value->type == kTfLiteInt4) {
        for (int j = 0; j < col_size; j++) {
          int i8_idx = j + idx * col_size;
          int i4_idx = i8_idx / 2;
          bool even = i8_idx % 2 == 0;
          int8_t i4_val = value_ptr[i4_idx];
          int8_t i8_val =
              even ? static_cast<int8_t>(i4_val << 4) >> 4 : i4_val >> 4;
          output_ptr[j + i * col_size] = i8_val * scaling_factor;
        }
      } else {
        for (int j = 0; j < col_size; j++) {
          output_ptr[j + i * col_size] =
              value_ptr[j + idx * col_size] * scaling_factor;
        }
      }
    }
  }
  return kTfLiteOk;
}
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* lookup;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &lookup));
  const TfLiteTensor* value;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &value));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  switch (value->type) {
    case kTfLiteFloat32:
      return EvalSimple(context, node, lookup, value, output);
    case kTfLiteInt4:
      return EvalHybrid(context, node, lookup, value, output);
    case kTfLiteUInt8:
    case kTfLiteInt8:
      if (output->type == kTfLiteFloat32) {
        return EvalHybrid(context, node, lookup, value, output);
      } else {
        return EvalSimple(context, node, lookup, value, output);
      }
    default:
      TF_LITE_KERNEL_LOG(context, "Type not currently supported.");
      return kTfLiteError;
  }
}
}  
TfLiteRegistration* Register_EMBEDDING_LOOKUP() {
  static TfLiteRegistration r = {nullptr, nullptr, embedding_lookup::Prepare,
                                 embedding_lookup::Eval};
  return &r;
}
}  
}  
}  