#include <stdint.h>
#include <algorithm>
#include <iterator>
#include <vector>
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
namespace tflite {
namespace ops {
namespace builtin {
namespace topk_v2 {
constexpr int kInputTensor = 0;
constexpr int kInputTopK = 1;
constexpr int kOutputValues = 0;
constexpr int kOutputIndexes = 1;
namespace {
TfLiteStatus ResizeOutput(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* top_k;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTopK, &top_k));
  TF_LITE_ENSURE(context,
                 top_k->type == kTfLiteInt32 || top_k->type == kTfLiteInt16);
  TF_LITE_ENSURE_EQ(context, NumElements(top_k), 1);
  int32 k;
  if (top_k->type == kTfLiteInt16) {
    k = *GetTensorData<int16_t>(top_k);
  } else {
    k = *GetTensorData<int32_t>(top_k);
  }
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const int num_dimensions = NumDimensions(input);
  TF_LITE_ENSURE_MSG(context, input->dims->size >= 1,
                     "TopK k input must have 1 or more dimensions.");
  TF_LITE_ENSURE_MSG(context, k <= input->dims->data[num_dimensions - 1],
                     "TopK k is higher than the internal dimension.");
  TfLiteIntArray* output_indexes_shape = TfLiteIntArrayCreate(num_dimensions);
  TfLiteIntArray* output_values_shape = TfLiteIntArrayCreate(num_dimensions);
  for (int i = 0; i < num_dimensions - 1; ++i) {
    output_indexes_shape->data[i] = input->dims->data[i];
    output_values_shape->data[i] = input->dims->data[i];
  }
  output_indexes_shape->data[num_dimensions - 1] = k;
  output_values_shape->data[num_dimensions - 1] = k;
  TfLiteTensor* output_indexes;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, kOutputIndexes, &output_indexes));
  TfLiteTensor* output_values;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, kOutputValues, &output_values));
  output_values->type = input->type;
  auto resize_tensor = [context](TfLiteTensor* tensor, TfLiteIntArray* new_size,
                                 TfLiteIntArray* delete_on_error) {
    TfLiteStatus status = context->ResizeTensor(context, tensor, new_size);
    if (status != kTfLiteOk) {
      if (delete_on_error != nullptr) {
        TfLiteIntArrayFree(delete_on_error);
      }
    }
    return status;
  };
  TF_LITE_ENSURE_OK(context, resize_tensor(output_indexes, output_indexes_shape,
                                           output_values_shape));
  TF_LITE_ENSURE_OK(context,
                    resize_tensor(output_values, output_values_shape, nullptr));
  return kTfLiteOk;
}
template <typename T, typename Tidx>
class TopContainer {
 public:
  TopContainer() = delete;
  TopContainer(int32 k, int32 row_size) : k_(k) {
    container_.reserve(std::min(k, row_size) + 1);
  }
  void start_collecting(const T* values) {
    values_ = values;
    container_.clear();
    is_heap_ = false;
  }
  void push(Tidx a) {
    auto comparator = [this](Tidx a, Tidx b) { return compare_fun(a, b); };
    if (!is_heap_) {
      container_.push_back(a);
      if (container_.size() == k_ + 1) {
        std::make_heap(container_.begin(), container_.end(), comparator);
        std::pop_heap(container_.begin(), container_.end(), comparator);
        container_.pop_back();
        is_heap_ = true;
      }
    } else if (comparator(a, container_.front())) {
      std::pop_heap(container_.begin(), container_.end(), comparator);
      container_.back() = a;
      std::push_heap(container_.begin(), container_.end(), comparator);
    }
  }
  const std::vector<Tidx>& sorted_result() {
    auto comparator = [this](Tidx a, Tidx b) { return compare_fun(a, b); };
    if (!is_heap_) {
      std::sort(container_.begin(), container_.end(), comparator);
    } else {
      std::sort_heap(container_.begin(), container_.end(), comparator);
    }
    return container_;
  }
 private:
  const int32 k_;
  std::vector<Tidx> container_;
  bool is_heap_ = false;
  const T* values_ = nullptr;
  bool compare_fun(Tidx a, Tidx b) const {
    if (values_[b] < values_[a]) {
      return true;
    } else if (values_[b] > values_[a]) {
      return false;
    } else {
      return a < b;
    }
  }
};
template <typename T, typename Tidx = int32>
void TopK(int32 row_size, int32 num_rows, const T* data, int32 k,
          Tidx* output_indexes, T* output_values) {
  TopContainer<T, Tidx> topc(k, row_size);
  for (int row = 0; row < num_rows; ++row) {
    const T* values_row = data + row * row_size;
    topc.start_collecting(values_row);
    for (int32 c = 0; c < row_size; ++c) {
      topc.push(c);
    }
    Tidx* indexes_row = output_indexes + row * k;
    T* output_row = output_values + row * k;
    const auto& top_k = topc.sorted_result();
    std::copy(top_k.begin(), top_k.end(), indexes_row);
    std::transform(top_k.begin(), top_k.end(), output_row,
                   [values_row](const int32 loc) { return values_row[loc]; });
  }
}
}  
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 2);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output_values;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, kOutputValues, &output_values));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output_values->type);
  const TfLiteTensor* top_k;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTopK, &top_k));
  TF_LITE_ENSURE(context,
                 top_k->type != kTfLiteInt32 || top_k->type != kTfLiteInt16);
  if (IsConstantOrPersistentTensor(top_k) && !HasUnspecifiedDimension(input)) {
    TF_LITE_ENSURE_OK(context, ResizeOutput(context, node));
  } else {
    TfLiteTensor* output_indexes;
    TF_LITE_ENSURE_OK(
        context, GetOutputSafe(context, node, kOutputIndexes, &output_indexes));
    TfLiteTensor* output_values;
    TF_LITE_ENSURE_OK(
        context, GetOutputSafe(context, node, kOutputValues, &output_values));
    SetTensorToDynamic(output_indexes);
    SetTensorToDynamic(output_values);
  }
  return kTfLiteOk;
}
template <typename idx_type>
TfLiteStatus TopKImpl(TfLiteContext* context, TfLiteNode* node, int32_t k,
                      idx_type* output_indexes) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output_values;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, kOutputValues, &output_values));
  const int32 row_size = input->dims->data[input->dims->size - 1];
  int32 num_rows = 1;
  for (int i = 0; i < input->dims->size - 1; ++i) {
    num_rows *= input->dims->data[i];
  }
  switch (output_values->type) {
    case kTfLiteFloat32:
      TopK(row_size, num_rows, GetTensorData<float>(input), k, output_indexes,
           GetTensorData<float>(output_values));
      break;
    case kTfLiteUInt8:
      TopK(row_size, num_rows, GetTensorData<uint8_t>(input), k, output_indexes,
           output_values->data.uint8);
      break;
    case kTfLiteInt8:
      TopK(row_size, num_rows, GetTensorData<int8_t>(input), k, output_indexes,
           output_values->data.int8);
      break;
    case kTfLiteInt16:
      TopK(row_size, num_rows, GetTensorData<int16_t>(input), k, output_indexes,
           output_values->data.i16);
      break;
    case kTfLiteInt32:
      TopK(row_size, num_rows, GetTensorData<int32_t>(input), k, output_indexes,
           output_values->data.i32);
      break;
    case kTfLiteInt64:
      TopK(row_size, num_rows, GetTensorData<int64_t>(input), k, output_indexes,
           output_values->data.i64);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s is currently not supported by TopK.",
                         TfLiteTypeGetName(output_values->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* output_values;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, kOutputValues, &output_values));
  TfLiteTensor* output_indexes;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, kOutputIndexes, &output_indexes));
  if (IsDynamicTensor(output_values)) {
    TF_LITE_ENSURE_OK(context, ResizeOutput(context, node));
  }
  const TfLiteTensor* top_k;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTopK, &top_k));
  int32 k;
  switch (top_k->type) {
    case kTfLiteInt32:
      k = top_k->data.i32[0];
      break;
    case kTfLiteInt16:
      k = top_k->data.i16[0];
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Type %s is currently not supported k Type by TopK.",
                         TfLiteTypeGetName(output_values->type));
      return kTfLiteError;
  }
  switch (output_indexes->type) {
    case kTfLiteInt32: {
      return TopKImpl(context, node, k, GetTensorData<int32_t>(output_indexes));
    }
    case kTfLiteInt16: {
      return TopKImpl(context, node, k, GetTensorData<int16_t>(output_indexes));
    }
    default:
      TF_LITE_KERNEL_LOG(
          context, "Output index type %s is currently not supported by TopK.",
          TfLiteTypeGetName(output_values->type));
      return kTfLiteError;
  }
}
}  
TfLiteRegistration* Register_TOPK_V2() {
  static TfLiteRegistration r = {nullptr, nullptr, topk_v2::Prepare,
                                 topk_v2::Eval};
  return &r;
}
}  
}  
}  