#include "tensorflow/lite/experimental/ml_adjacent/algo/rgb_to_grayscale.h"
#include "tensorflow/lite/experimental/ml_adjacent/lib.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
namespace ml_adj {
namespace rgb_to_grayscale {
namespace {
using ::ml_adj::algo::Algo;
using ::ml_adj::algo::InputPack;
using ::ml_adj::algo::OutputPack;
using ::ml_adj::data::DataRef;
using ::ml_adj::data::MutableDataRef;
inline void ConvertRgbToGrayscale(dim_t batches, dim_t height, dim_t width,
                                  const float* input_data, float* output_data) {
  const dim_t output_num_pixels = batches * width * height;
  constexpr float kRgb2GrayscaleKernel[] = {0.2989f, 0.5870f, 0.1140f};
  const float* src_ptr = input_data;
  float* dst_ptr = output_data;
  for (int i = 0; i < output_num_pixels; ++i) {
    *dst_ptr = kRgb2GrayscaleKernel[0] * src_ptr[0] +
               kRgb2GrayscaleKernel[1] * src_ptr[1] +
               kRgb2GrayscaleKernel[2] * src_ptr[2];
    src_ptr += 3;  
    dst_ptr++;
  }
}
void ComputeRgbToGrayscale(const InputPack& inputs, const OutputPack& outputs) {
  TFLITE_DCHECK(inputs.size() == 1);
  TFLITE_DCHECK(outputs.size() == 1);
  const DataRef* img = inputs[0];
  const float* img_data = reinterpret_cast<const float*>(img->Data());
  const dim_t img_num_batches = img->Dims()[0];
  const dim_t img_height = img->Dims()[1];
  const dim_t img_width = img->Dims()[2];
  const dim_t channels = img->Dims()[3];
  TFLITE_DCHECK(channels == 3);
  MutableDataRef* output = outputs[0];
  output->Resize({img_num_batches, img_height, img_width, 1});
  float* output_data = reinterpret_cast<float*>(output->Data());
  ConvertRgbToGrayscale(img_num_batches, img_height, img_width, img_data,
                        output_data);
}
}  
const Algo* Impl_RgbToGrayscale() {
  static const Algo rgb_to_grayscale = {&ComputeRgbToGrayscale, nullptr};
  return &rgb_to_grayscale;
}
}  
}  