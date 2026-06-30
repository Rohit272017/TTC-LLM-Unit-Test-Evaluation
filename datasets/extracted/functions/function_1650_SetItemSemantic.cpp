#include <utility>
#include "tensorflow/lite/array.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/variants/tensor_array.h"
#include "tensorflow/lite/util.h"
namespace tflite {
namespace variants {
namespace ops {
namespace {
constexpr int kListInputIdx = 0;
constexpr int kIndexInputIdx = 1;
constexpr int kListOutputIdx = 0;
class SetItemSemantic {
 public:
  SetItemSemantic(TfLiteContext* ctx, TfLiteNode* node)
      : ctx_(ctx), node_(node) {}
  static constexpr int kItemInputIdx = 2;
  TfLiteStatus CheckIndexInput() const {
    const TfLiteTensor* index_input;
    TF_LITE_ENSURE_OK(ctx_,
                      GetInputSafe(ctx_, node_, kIndexInputIdx, &index_input));
    TF_LITE_ENSURE_TYPES_EQ(ctx_, index_input->type, kTfLiteInt32);
    return kTfLiteOk;
  }
  TfLiteStatus GetIndexVal(const TensorArray& arr, int& result) const {
    const TfLiteTensor* index_input;
    TF_LITE_ENSURE_OK(ctx_,
                      GetInputSafe(ctx_, node_, kIndexInputIdx, &index_input));
    TF_LITE_ENSURE_EQ(ctx_, index_input->bytes, sizeof(int));
    const int* index_data = GetTensorData<int>(index_input);
    TF_LITE_ENSURE(ctx_, index_data != nullptr);
    const int index = *index_data;
    TF_LITE_ENSURE(ctx_, index >= 0);
    result = index;
    return kTfLiteOk;
  }
 private:
  TfLiteContext* const ctx_;
  TfLiteNode* const node_;
};
class PushBackSemantic {
 public:
  PushBackSemantic(TfLiteContext* ctx, TfLiteNode* node) {}
  static constexpr int kItemInputIdx = 1;
  TfLiteStatus CheckIndexInput() const { return kTfLiteOk; }
  TfLiteStatus GetIndexVal(const TensorArray& arr, int& result) const {
    result = arr.NumElements();
    return kTfLiteOk;
  }
};
template <class Semantic>
TfLiteStatus Prepare(TfLiteContext* ctx, TfLiteNode* node) {
  const auto semantic = Semantic(ctx, node);
  const TfLiteTensor* list_input;
  TF_LITE_ENSURE_OK(ctx, GetInputSafe(ctx, node, kListInputIdx, &list_input));
  TF_LITE_ENSURE_TYPES_EQ(ctx, list_input->type, kTfLiteVariant);
  TF_LITE_ENSURE_OK(ctx, semantic.CheckIndexInput());
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(ctx, GetOutputSafe(ctx, node, kListOutputIdx, &output));
  TF_LITE_ENSURE_TYPES_EQ(ctx, output->type, kTfLiteVariant);
  output->allocation_type = kTfLiteVariantObject;
  return kTfLiteOk;
}
template <class Semantic>
TfLiteStatus Eval(TfLiteContext* ctx, TfLiteNode* node) {
  const auto semantic = Semantic(ctx, node);
  const TfLiteTensor* list_input;
  TF_LITE_ENSURE_OK(ctx, GetInputSafe(ctx, node, kListInputIdx, &list_input));
  TF_LITE_ENSURE_EQ(ctx, list_input->allocation_type, kTfLiteVariantObject);
  TensorArray* input_arr =
      reinterpret_cast<TensorArray*>(list_input->data.data);
  int index;
  TF_LITE_ENSURE_OK(ctx, semantic.GetIndexVal(*input_arr, index));
  const TfLiteTensor* item_input;
  TF_LITE_ENSURE_OK(
      ctx, GetInputSafe(ctx, node, semantic.kItemInputIdx, &item_input));
  TF_LITE_ENSURE_TYPES_EQ(ctx, input_arr->ElementType(), item_input->type);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(ctx, GetOutputSafe(ctx, node, kListOutputIdx, &output));
  TensorArray* output_arr = static_cast<TensorArray*>(
      input_arr->CloneTo(static_cast<VariantData*>(output->data.data)));
  TensorUniquePtr item_copy = BuildTfLiteTensor(
      item_input->type, BuildTfLiteArray(*item_input->dims), kTfLiteDynamic);
  TfLiteTensorCopy(item_input, item_copy.get());
  if (index >= output_arr->NumElements()) {
    output_arr->Resize(index + 1);
  }
  TF_LITE_ENSURE(ctx, output_arr->Set(index, std::move(item_copy)));
  output->data.data = static_cast<VariantData*>(output_arr);
  return kTfLiteOk;
}
}  
TfLiteRegistration* Register_LIST_SET_ITEM() {
  static TfLiteRegistration r = {nullptr, nullptr, Prepare<SetItemSemantic>,
                                 Eval<SetItemSemantic>};
  return &r;
}
TfLiteRegistration* Register_LIST_PUSH_BACK() {
  static TfLiteRegistration r = {nullptr, nullptr, Prepare<PushBackSemantic>,
                                 Eval<PushBackSemantic>};
  return &r;
}
}  
}  
}  