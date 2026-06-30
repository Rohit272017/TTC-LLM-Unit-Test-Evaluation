#include <cstdint>
#include <cstring>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
namespace tflite {
namespace ops {
namespace custom {
namespace llm {
static const int kKeyTensor = 0;
static const int kValueTensor = 1;
static const int kPositionTensor = 2;
static const int kKeySliceTensor = 3;
static const int kValueSliceTensor = 4;
static const int kRequiredNumDimensions = 4;
TfLiteStatus ExternalKVCachePrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 5);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 2);
  const TfLiteTensor* k_cache;
  const TfLiteTensor* v_cache;
  const TfLiteTensor* position;
  const TfLiteTensor* k_slice;
  const TfLiteTensor* v_slice;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kKeyTensor, &k_cache));
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kValueTensor, &v_cache));
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kPositionTensor, &position));
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kKeySliceTensor, &k_slice));
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kValueSliceTensor, &v_slice));
  TfLiteTensor* updated_k_cache;
  TfLiteTensor* updated_v_cache;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kKeyTensor, &updated_k_cache));
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, kValueTensor, &updated_v_cache));
  TF_LITE_ENSURE_EQ(context, k_cache->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, v_cache->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, position->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, k_slice->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, v_slice->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, updated_k_cache->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, updated_v_cache->type, kTfLiteFloat32);
  TF_LITE_ENSURE(context, HaveSameShapes(k_cache, v_cache));
  TF_LITE_ENSURE(context, HaveSameShapes(k_slice, v_slice));
  TF_LITE_ENSURE(context, HaveSameShapes(updated_k_cache, updated_v_cache));
  TF_LITE_ENSURE(context, HaveSameShapes(k_cache, updated_k_cache));
  TF_LITE_ENSURE(context, NumDimensions(k_slice) == kRequiredNumDimensions);
  TF_LITE_ENSURE(context, NumDimensions(k_cache) == kRequiredNumDimensions);
  TF_LITE_ENSURE(context, NumDimensions(position) == 1);
  TF_LITE_ENSURE(context, GetTensorShape(position).Dims(0) ==
                              GetTensorShape(k_slice).Dims(1));
  TF_LITE_ENSURE(context, GetTensorShape(k_slice).Dims(0) == 1);
  return kTfLiteOk;
}
TfLiteStatus ExternalKVCacheEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* k_cache;
  const TfLiteTensor* v_cache;
  const TfLiteTensor* position;
  const TfLiteTensor* k_slice;
  const TfLiteTensor* v_slice;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kKeyTensor, &k_cache));
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kValueTensor, &v_cache));
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kPositionTensor, &position));
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kKeySliceTensor, &k_slice));
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kValueSliceTensor, &v_slice));
  TfLiteTensor* updated_k_cache;
  TfLiteTensor* updated_v_cache;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kKeyTensor, &updated_k_cache));
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, kValueTensor, &updated_v_cache));
  if (k_cache->data.raw != updated_k_cache->data.raw) {
    memcpy(updated_k_cache->data.data, k_cache->data.data, k_cache->bytes);
  }
  if (v_cache->data.raw != updated_v_cache->data.raw) {
    memcpy(updated_v_cache->data.data, v_cache->data.data, v_cache->bytes);
  }
  const int32_t elements_in_one_entry =
      GetTensorShape(k_cache).Dims(2) * GetTensorShape(k_cache).Dims(3);
  const int32_t cache_size = GetTensorShape(k_cache).Dims(1);
  int32_t last_update_position = -1;
  for (int i = 0; i < position->bytes / sizeof(int32_t); ++i) {
    const int32_t update_position = position->data.i32[i];
    if (update_position < last_update_position) {
      break;
    }
    last_update_position = update_position;
    TF_LITE_ENSURE(context, update_position < cache_size);
    const int32_t cache_offset = update_position * elements_in_one_entry;
    const int32_t update_offset = i * elements_in_one_entry;
    TF_LITE_ENSURE(context,
                   (cache_offset + elements_in_one_entry) * sizeof(float) <=
                       k_cache->bytes);
    memcpy(updated_k_cache->data.f + cache_offset,
           k_slice->data.f + update_offset,
           elements_in_one_entry * sizeof(float));
    memcpy(updated_v_cache->data.f + cache_offset,
           v_slice->data.f + update_offset,
           elements_in_one_entry * sizeof(float));
  }
  return kTfLiteOk;
}
}  
TfLiteRegistration* Register_EXTERNAL_KV_CACHE() {
  static TfLiteRegistration r = {nullptr, nullptr,
                                 llm::ExternalKVCachePrepare,
                                 llm::ExternalKVCacheEval};
  return &r;
}
}  
}  
}  