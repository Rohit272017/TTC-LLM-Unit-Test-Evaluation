#include "tensorflow/lite/experimental/ml_adjacent/algo/image_utils.h"
#include "tensorflow/lite/experimental/ml_adjacent/lib.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
namespace ml_adj {
namespace rgb_to_yuv {
namespace {
using ::ml_adj::algo::Algo;
using ::ml_adj::algo::InputPack;
using ::ml_adj::algo::OutputPack;
using ::ml_adj::data::DataRef;
using ::ml_adj::data::MutableDataRef;
constexpr float kRgb2YuvKernel[] = {0.299f,       0.587f,       0.114f,       
                                    -0.14714119f, -0.28886916f, 0.43601035f,  
                                    0.61497538f,  -0.51496512f, -0.10001026f};
constexpr int kRgb2YuvKernelSize =
    sizeof(kRgb2YuvKernel) / sizeof(kRgb2YuvKernel[0]);
void ComputeRgbToYuv(const InputPack& inputs, const OutputPack& outputs) {
  TFLITE_DCHECK(inputs.size() == 1);
  TFLITE_DCHECK(outputs.size() == 1);
  const DataRef* img = inputs[0];
  const float* input_data = reinterpret_cast<const float*>(img->Data());
  const dim_t batches = img->Dims()[0];
  const dim_t height = img->Dims()[1];
  const dim_t width = img->Dims()[2];
  const dim_t channels = img->Dims()[3];
  TFLITE_DCHECK(channels == 3);
  MutableDataRef* output = outputs[0];
  output->Resize({batches, height, width, channels});
  float* output_data = reinterpret_cast<float*>(output->Data());
  ConvertColorSpace(batches, height, width, input_data, output_data,
                    &kRgb2YuvKernel[0], kRgb2YuvKernelSize);
}
}  
const Algo* Impl_RgbToYuv() {
  static const Algo rgb_to_yuv = {&ComputeRgbToYuv, nullptr};
  return &rgb_to_yuv;
}
}  
}  