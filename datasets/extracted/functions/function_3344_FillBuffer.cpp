#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/util.h"
namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_pad {
namespace {
static constexpr int kMaxDims = TFLITE_STABLEHLO_PAD_PARAMS_MAX_DIMENSION_COUNT;
void FillBuffer(char* buffer, int64_t buffer_bytes, const char* data,
                int64_t data_bytes) {
  if (buffer_bytes == 0) {
    return;
  }
  assert(buffer_bytes % data_bytes == 0);
  std::memcpy(buffer, data, data_bytes);
  buffer_bytes -= data_bytes;
  while (buffer_bytes) {
    const int64_t bytes = std::min(buffer_bytes, data_bytes);
    std::memcpy(buffer + data_bytes, buffer, bytes);
    buffer_bytes -= bytes;
    data_bytes += bytes;
  }
}
void StridedCopy(const int rank, const char* input, const int64_t* input_shape,
                 const int64_t* input_strides, char* output,
                 const int64_t* output_strides, const int64_t element_size,
                 const int depth) {
  if (depth + 1 == rank) {
    for (int64_t i = 0; i < input_shape[depth]; ++i) {
      std::memcpy(output, input, element_size);
      input += input_strides[depth];
      output += output_strides[depth];
    }
  } else {
    for (int64_t i = 0; i < input_shape[depth]; ++i) {
      StridedCopy(rank, input, input_shape, input_strides, output,
                  output_strides, element_size, depth + 1);
      input += input_strides[depth];
      output += output_strides[depth];
    }
  }
}
class PadData {
 public:
  enum { kInput, kPaddingValue, kInputTensorCount };
  enum { kOutput, kOutputTensorCount };
  explicit PadData(const TfLiteStablehloPadParams& params) {
    std::memcpy(
        edge_pad_low_, params.edge_padding_low,
        TFLITE_STABLEHLO_PAD_PARAMS_MAX_DIMENSION_COUNT * sizeof(int64_t));
    std::memcpy(
        edge_pad_high_, params.edge_padding_high,
        TFLITE_STABLEHLO_PAD_PARAMS_MAX_DIMENSION_COUNT * sizeof(int64_t));
    std::memcpy(
        interior_pad_, params.interior_padding,
        TFLITE_STABLEHLO_PAD_PARAMS_MAX_DIMENSION_COUNT * sizeof(int64_t));
  }
  void Setup(const int* dims, const int rank, const int64_t element_size) {
    rank_ = rank;
    element_size_ = element_size;
    input_offset_ = 0;
    output_offset_ = 0;
    output_size_ = 0;
    for (int i = 0; i < rank; ++i) {
      output_shape_[i] = (dims[i] - 1) * (interior_pad_[i] + 1) + 1 +
                         edge_pad_low_[i] + edge_pad_high_[i];
    }
    if (std::any_of(output_shape_, output_shape_ + rank,
                    [](auto s) { return s <= 0; })) {
      std::memset(input_shape_, 0, sizeof(input_shape_));
      std::memset(output_shape_, 0, sizeof(output_shape_));
      output_size_ = 0;
      return;
    }
    output_dimension_sizes_[rank - 1] = element_size;
    for (int i = rank - 2; i >= 0; --i) {
      output_dimension_sizes_[i] =
          output_shape_[i + 1] * output_dimension_sizes_[i + 1];
    }
    output_strides_[rank - 1] = element_size * (interior_pad_[rank - 1] + 1);
    for (int i = rank - 2; i >= 0; --i) {
      output_strides_[i] = output_dimension_sizes_[i] * (interior_pad_[i] + 1);
    }
    for (int i = 0; i < rank; ++i) {
      output_offset_ +=
          std::max<int64_t>(edge_pad_low_[i], 0) * output_dimension_sizes_[i];
    }
    output_size_ = std::accumulate(output_shape_, output_shape_ + rank,
                                   element_size, std::multiplies<>());
    input_strides_[rank - 1] = element_size;
    for (int i = rank - 1; i >= 1; --i) {
      input_strides_[i - 1] = dims[i] * input_strides_[i];
    }
    auto DivNegRoundAwayOrZero = [](int64_t num, int64_t denum) -> int64_t {
      assert(denum > 0);
      return num < 0 ? (num - denum + 1) / denum : 0;
    };
    for (int i = 0; i < rank; ++i) {
      input_shape_[i] =
          dims[i] +
          DivNegRoundAwayOrZero(edge_pad_low_[i], interior_pad_[i] + 1) +
          DivNegRoundAwayOrZero(edge_pad_high_[i], interior_pad_[i] + 1);
    }
    for (int i = 0; i < rank; ++i) {
      input_offset_ -=
          DivNegRoundAwayOrZero(edge_pad_low_[i], interior_pad_[i] + 1) *
          input_strides_[i];
      if (edge_pad_low_[i] < 0) {
        int64_t tmp_offset = ((interior_pad_[i] + 1 + edge_pad_low_[i]) %
                              (interior_pad_[i] + 1));
        if (tmp_offset < 0) {
          tmp_offset += interior_pad_[i] + 1;
        }
        output_offset_ += tmp_offset * output_dimension_sizes_[i];
      }
    }
  }
  void Apply(const char* input, const char* padding_value, char* output) const {
    FillBuffer(output, output_size_, padding_value, element_size_);
    StridedCopy(rank_, input + input_offset_, input_shape_, input_strides_,
                output + output_offset_, output_strides_, element_size_,
                0);
  }
  TfLiteIntArray* BuildOuputTensorDims() const {
    TfLiteIntArray* dims = TfLiteIntArrayCreate(rank_);
    for (int64_t i = 0; i < rank_; ++i) {
      dims->data[i] = output_shape_[i];
    }
    return dims;
  }
 private:
  int64_t edge_pad_low_[kMaxDims];
  int64_t edge_pad_high_[kMaxDims];
  int64_t interior_pad_[kMaxDims];
  int64_t rank_ = 0;
  int64_t element_size_ = 0;
  int64_t input_shape_[kMaxDims];
  int64_t output_shape_[kMaxDims];
  int64_t input_strides_[kMaxDims];
  int64_t output_strides_[kMaxDims];
  int64_t output_dimension_sizes_[kMaxDims];
  int64_t input_offset_ = 0;
  int64_t output_offset_ = 0;
  int64_t output_size_ = 0;
};
void* Init(TfLiteContext* context, const char* options, size_t options_len) {
  return new PadData(
      *reinterpret_cast<const TfLiteStablehloPadParams*>(options));
}
void Free(TfLiteContext* context, void* node_data) {
  delete reinterpret_cast<PadData*>(node_data);
}
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input_tensor = GetInput(context, node, PadData::kInput);
  const TfLiteTensor* padding_value_tensor =
      GetInput(context, node, PadData::kPaddingValue);
  TF_LITE_ENSURE(context, input_tensor->type == padding_value_tensor->type);
  size_t element_size;
  TF_LITE_ENSURE(context, GetSizeOfType(context, input_tensor->type,
                                        &element_size) == kTfLiteOk);
  PadData& pad_data = *reinterpret_cast<PadData*>(node->user_data);
  pad_data.Setup(input_tensor->dims->data, input_tensor->dims->size,
                 element_size);
  TfLiteTensor* output_tensor = GetOutput(context, node, PadData::kOutput);
  TF_LITE_ENSURE(context, input_tensor->type == output_tensor->type);
  context->ResizeTensor(context, output_tensor,
                        pad_data.BuildOuputTensorDims());
  return kTfLiteOk;
}
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input_tensor = GetInput(context, node, PadData::kInput);
  const TfLiteTensor* padding_value_tensor =
      GetInput(context, node, PadData::kPaddingValue);
  TfLiteTensor* output_tensor = GetOutput(context, node, PadData::kOutput);
  PadData& pad_data = *reinterpret_cast<PadData*>(node->user_data);
  pad_data.Apply(input_tensor->data.raw_const,
                 padding_value_tensor->data.raw_const, output_tensor->data.raw);
  return kTfLiteOk;
}
}  
}  
TfLiteRegistration* Register_STABLEHLO_PAD() {
  static TfLiteRegistration r = {stablehlo_pad::Init,
                                 stablehlo_pad::Free,
                                 stablehlo_pad::Prepare,
                                 stablehlo_pad::Eval};
  return &r;
}
}  
}  
}  