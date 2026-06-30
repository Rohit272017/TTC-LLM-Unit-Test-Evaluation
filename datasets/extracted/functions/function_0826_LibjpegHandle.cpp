#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_LIBJPEG_HANDLE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_LIBJPEG_HANDLE_H_
#include <stddef.h>
#include <stdio.h>
#include <memory>
#include <string>
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg_status.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg.h"
namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {
class LibjpegHandle {
 public:
  static std::unique_ptr<LibjpegHandle> Create(Status& status);
  ~LibjpegHandle();
  LibjpegHandle(LibjpegHandle const&) = delete;
  LibjpegHandle& operator=(const LibjpegHandle&) = delete;
  LibjpegHandle(LibjpegHandle&& LibjpegHandle) = delete;
  LibjpegHandle& operator=(LibjpegHandle&& other) = delete;
  static const int kLibjpegVersion = 62;
  struct jpeg_error_mgr* (*jpeg_std_error_)(struct jpeg_error_mgr*);
  void (*jpeg_destroy_decompress_)(j_decompress_ptr);
  void (*jpeg_create_decompress_)(j_decompress_ptr, int, size_t);
  void (*jpeg_stdio_src_)(j_decompress_ptr, FILE*);
  int (*jpeg_read_header_)(j_decompress_ptr, boolean);
  boolean (*jpeg_start_decompress_)(j_decompress_ptr);
  unsigned int (*jpeg_read_scanlines_)(j_decompress_ptr, JSAMPARRAY,
                                       JDIMENSION);
  boolean (*jpeg_finish_decompress_)(j_decompress_ptr);
 private:
  LibjpegHandle() {}
  void* libjpeg_ = nullptr;
};
}  
}  
}  
#endif  