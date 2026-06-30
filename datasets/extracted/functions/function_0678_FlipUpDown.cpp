#include <cstring>
#include "tensorflow/lite/experimental/ml_adjacent/lib.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
namespace ml_adj {
namespace flip_up_down {
namespace {
using ::ml_adj::algo::Algo;
using ::ml_adj::algo::InputPack;
using ::ml_adj::algo::OutputPack;
using ::ml_adj::data::DataRef;
using ::ml_adj::data::MutableDataRef;
using ::ml_adj::data::TypeWidth;
void FlipUpDown(dim_t batches, dim_t input_height, dim_t input_width,
                const char* input_data, char* output_data, dim_t chunk_size) {
  const dim_t row_stride = input_width * chunk_size;
  const dim_t batch_stride = row_stride * input_height;
  for (int b = 0; b < batches; ++b) {
    const char* src_data_prt = input_data + b * batch_stride;
    char* dst_data_prt = output_data + b * batch_stride;
    for (int y = 0; y < input_height; ++y) {
      const char* src_ptr_row =
          src_data_prt + (input_height - y - 1) * row_stride;
      char* dst_ptr_row = dst_data_prt + y * row_stride;
      std::memcpy(dst_ptr_row, src_ptr_row, row_stride);
    }
  }
}
void ComputeFlipUpDown(const InputPack& inputs, const OutputPack& outputs) {
  TFLITE_DCHECK(inputs.size() == 1);
  TFLITE_DCHECK(outputs.size() == 1);
  const DataRef* img = inputs[0];
  const char* img_data = reinterpret_cast<const char*>(img->Data());
  const dim_t num_batches = img->Dims()[0];
  const dim_t height = img->Dims()[1];
  const dim_t width = img->Dims()[2];
  const dim_t num_channels = img->Dims()[3];
  const dim_t chunk_size = TypeWidth(img->Type()) * num_channels;
  MutableDataRef* output = outputs[0];
  output->Resize({num_batches, height, width, num_channels});
  char* output_data = reinterpret_cast<char*>(output->Data());
  FlipUpDown(num_batches, height, width, img_data, output_data, chunk_size);
}
}  
const Algo* Impl_FlipUpDown() {
  static const Algo flip_up_down = {&ComputeFlipUpDown, nullptr};
  return &flip_up_down;
}
}  
}  