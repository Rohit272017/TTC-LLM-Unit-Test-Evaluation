#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>
#include "tensorflow/lite/array.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/util.h"
namespace tflite {
namespace ops {
namespace builtin {
namespace dilate {
namespace {
constexpr size_t kMaxDilateDims = 6;
using Array = std::array<int32_t, kMaxDilateDims>;
void DilateImpl(const char* input, char* output,
                const char* const padding_values, const int32_t size,
                const int32_t* const shape, const int32_t* const input_strides,
                const int32_t* const output_strides,
                const int32_t* const output_element_sizes, size_t depth = 0) {
  const int output_stride = output_strides[depth];
  const int input_stride = input_strides[depth];
  const int num_elts = shape[depth];
  const int padding_size = output_stride - output_element_sizes[depth];
  if (depth + 1 >= size) {
    for (size_t i = 0; i + 1 < num_elts; ++i) {
      std::memcpy(output, input, input_stride);
      std::memcpy(output + input_stride, padding_values, padding_size);
      input += input_stride;
      output += output_stride;
    }
    std::memcpy(output, input, input_stride);
  } else {
    for (size_t i = 0; i + 1 < num_elts; ++i) {
      DilateImpl(input, output, padding_values, size, shape, input_strides,
                 output_strides, output_element_sizes, depth + 1);
      std::memcpy(output + output_element_sizes[depth], padding_values,
                  padding_size);
      input += input_stride;
      output += output_stride;
    }
    DilateImpl(input, output, padding_values, size, shape, input_strides,
               output_strides, output_element_sizes, depth + 1);
  }
}
class DilationRunner {
 public:
  DilationRunner(const TfLiteIntArray& shape, const int32_t* const dilations,
                 const char* padding_value, const int element_size)
      : size_(shape.size), element_size_(element_size) {
    static_assert(sizeof(shape.data[0]) == sizeof(Array::value_type),
                  "Don't use memcpy here if you change the Array type.");
    std::memcpy(shape_.data(), shape.data, size_ * sizeof(shape.data[0]));
    static_assert(sizeof(dilations[0]) == sizeof(Array::value_type),
                  "Don't use memcpy here if you change the Array type.");
    std::memcpy(dilations_.data(), dilations, size_ * sizeof(dilations[0]));
    MergeTrailingDilations();
    ComputeInputStrides();
    ComputeOutputStridesAndElementSizes();
    FillPaddingValueBuffer(padding_value, element_size);
  }
  int size() const { return size_; }
  int element_size() const { return element_size_; }
  const char* padding_values() const { return padding_value_buffer_.data(); }
  const Array& shape() const { return shape_; }
  const Array& dilations() const { return dilations_; }
  const Array& input_strides() const { return input_strides_; }
  const Array& output_strides() const { return output_strides_; }
  const Array& output_element_sizes() const { return output_element_sizes_; }
  void Run(const char* const input, char* const output) const {
    DilateImpl(input, output, padding_values(), size(), shape().data(),
               input_strides().data(), output_strides().data(),
               output_element_sizes().data());
  }
 private:
  void MergeTrailingDilations() {
    for (int i = size_ - 2; i >= 0; --i) {
      if (dilations_[i + 1] == 1) {
        element_size_ *= shape_[i + 1];
        --size_;
      } else {
        break;
      }
    }
    if (size_ == 1 && dilations_[0] == 1) {
      element_size_ *= shape_[0];
      shape_[0] = 1;
    }
  }
  void ComputeInputStrides() {
    input_strides_[size_ - 1] = element_size_;
    for (int i = size_ - 2; i >= 0; --i) {
      input_strides_[i] = shape_[i + 1] * input_strides_[i + 1];
    }
  }
  void ComputeOutputStridesAndElementSizes() {
    const int last = size_ - 1;
    output_element_sizes_[last] = element_size_;
    output_strides_[last] = dilations_[last] * output_element_sizes_[last];
    for (int i = size_ - 2; i >= 0; --i) {
      output_element_sizes_[i] = ((shape_[i + 1] - 1) * output_strides_[i + 1] +
                                  output_element_sizes_[i + 1]);
      output_strides_[i] = dilations_[i] * output_element_sizes_[i];
    }
  }
  void FillPaddingValueBuffer(const char* padding_element,
                              const size_t padding_element_size) {
    int first_dilated_idx = 0;
    while (dilations_[first_dilated_idx] == 1 &&
           first_dilated_idx + 1 < size_) {
      ++first_dilated_idx;
    }
    const size_t size = output_strides_[first_dilated_idx] -
                        output_element_sizes_[first_dilated_idx];
    if (!size) {
      return;
    }
    padding_value_buffer_.resize(size);
    std::memcpy(padding_value_buffer_.data(), padding_element,
                padding_element_size);
    size_t sz = padding_element_size;
    while (sz < size) {
      const size_t bytes_to_copy = std::min(size - sz, sz);
      std::memcpy(padding_value_buffer_.data() + sz,
                  padding_value_buffer_.data(), bytes_to_copy);
      sz += bytes_to_copy;
    }
  }
  Array shape_;
  Array dilations_;
  Array output_strides_;
  Array output_element_sizes_;
  Array input_strides_;
  std::vector<char> padding_value_buffer_;
  int size_;
  int element_size_;
};
struct DilationContext {
  enum InputTensorId { kInput, kDilations, kPaddingValue, kNumInputTensors };
  enum OutputTensorId { kOutput, kNumOutputTensors };
  DilationContext(TfLiteContext* context, TfLiteNode* node)
      : context(context),
        node(node),
        input_tensor(GetInput(context, node, kInput)),
        dilations_tensor(GetInput(context, node, kDilations)),
        padding_value_tensor(GetInput(context, node, kPaddingValue)),
        output_tensor(GetOutput(context, node, kOutput)) {}
  TfLiteContext* context;
  TfLiteNode* node;
  const TfLiteTensor* input_tensor;
  const TfLiteTensor* dilations_tensor;
  const TfLiteTensor* padding_value_tensor;
  TfLiteTensor* output_tensor;
};
int DilateDim(int dim, int dilate_factor) {
  return (dim - 1) * dilate_factor + 1;
}
TfLiteStatus SetupOutputTensor(const DilationContext& ctx) {
  const TfLiteIntArray& input_shape = *(ctx.input_tensor->dims);
  const int32_t* dilations = ctx.dilations_tensor->data.i32;
  IntArrayUniquePtr output_shape = BuildTfLiteArray(input_shape.size);
  for (int i = 0; i < output_shape->size; ++i) {
    output_shape->data[i] = DilateDim(input_shape.data[i], dilations[i]);
  }
  return ctx.context->ResizeTensor(ctx.context, ctx.output_tensor,
                                   output_shape.release());
}
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node),
                    DilationContext::kNumInputTensors);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node),
                    DilationContext::kNumOutputTensors);
  const DilationContext ctx(context, node);
  TF_LITE_ENSURE(context, ctx.input_tensor->dims != nullptr);
  TF_LITE_ENSURE(context, ctx.input_tensor->dims->size > 0);
  TF_LITE_ENSURE(context, ctx.input_tensor->dims->size <= kMaxDilateDims);
  TF_LITE_ENSURE_EQ(context, ctx.input_tensor->type, ctx.output_tensor->type);
  TF_LITE_ENSURE_EQ(context, ctx.input_tensor->type,
                    ctx.padding_value_tensor->type);
  if (!IsConstantTensor(ctx.dilations_tensor)) {
    SetTensorToDynamic(ctx.output_tensor);
    return kTfLiteOk;
  }
  return SetupOutputTensor(ctx);
}
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const DilationContext ctx(context, node);
  TF_LITE_ENSURE_EQ(context, ctx.dilations_tensor->type, kTfLiteInt32);
  TF_LITE_ENSURE(context, ctx.dilations_tensor->dims != nullptr);
  TF_LITE_ENSURE_EQ(context, ctx.dilations_tensor->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, ctx.dilations_tensor->dims->data[0],
                    ctx.input_tensor->dims->size);
  for (int i = 0; i < ctx.dilations_tensor->dims->size; ++i) {
    TF_LITE_ENSURE(context, ctx.dilations_tensor->data.i32[i] >= 1);
  }
  if (!IsConstantTensor(ctx.dilations_tensor)) {
    TF_LITE_ENSURE_OK(context, SetupOutputTensor(ctx));
  }
  size_t element_size;
  TF_LITE_ENSURE_OK(
      context, GetSizeOfType(context, ctx.input_tensor->type, &element_size));
  const DilationRunner runner(
      *ctx.input_tensor->dims, ctx.dilations_tensor->data.i32,
      ctx.padding_value_tensor->data.raw_const, element_size);
  runner.Run(ctx.input_tensor->data.raw_const, ctx.output_tensor->data.raw);
  return kTfLiteOk;
}
}  
}  
TfLiteRegistration* Register_DILATE() {
  static TfLiteRegistration r = {nullptr, nullptr,
                                 dilate::Prepare,
                                 dilate::Eval};
  return &r;
}
}  
}  
}  