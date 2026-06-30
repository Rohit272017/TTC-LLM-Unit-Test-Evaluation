#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_JPEG_DECOMPRESS_BUFFERED_STRUCT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_JPEG_DECOMPRESS_BUFFERED_STRUCT_H_
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <vector>
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg.h"
namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {
class JpegDecompressBufferedStruct {
 public:
  explicit JpegDecompressBufferedStruct(std::size_t expected_size)
      : resized_size_(std::max(sizeof(jpeg_decompress_struct), expected_size)),
        buffer_(reinterpret_cast<char*>(malloc(resized_size_))) {
    while (--expected_size >= sizeof(jpeg_decompress_struct)) {
      buffer_[expected_size] = 0;
    }
  }
  ~JpegDecompressBufferedStruct() { std::free(buffer_); }
  JpegDecompressBufferedStruct(const JpegDecompressBufferedStruct&) = delete;
  JpegDecompressBufferedStruct& operator=(const JpegDecompressBufferedStruct&) =
      delete;
  jpeg_decompress_struct* get() const {
    return reinterpret_cast<jpeg_decompress_struct*>(buffer_);
  }
  int const size() { return resized_size_; }
  const char* buffer() { return buffer_; }
 private:
  int resized_size_;
  char* const buffer_;
};
}  
}  
}  
#endif  