#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_FULLY_CONNECTED_4BIT_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_FULLY_CONNECTED_4BIT_H_
#include <stdint.h>
#ifndef TFLITE_MMAP_DISABLED
#include <sys/mman.h>
#endif
#include <cstdlib>
#include <memory>
#if defined(FC_4BIT_SSE) && defined(__SSSE3__)
#include "tensorflow/lite/kernels/internal/optimized/4bit/sse_fully_connected.h"
#elif defined(FC_4BIT_NEON) && (defined(__ARM_NEON__) || defined(__ARM_NEON))
#include "tensorflow/lite/kernels/internal/optimized/4bit/neon_fully_connected.h"
#else
#include "tensorflow/lite/kernels/internal/optimized/4bit/fully_connected_reference.h"
#endif
namespace tflite {
namespace optimized_4bit {
constexpr int FilterWidth = 4;
constexpr int FilterDepth = 32;
constexpr int kDefaultAlignmentPadding = 63;
struct Deleter {
  explicit Deleter(size_t size = 0) : size(size) {}
  void operator()(uint8_t* memory) {
    if (!memory) {
      return;
    }
#ifdef TFLITE_MMAP_DISABLED
    delete[] memory;
#else
    munmap(memory, size);
#endif
  }
  size_t size;
};
struct OpData4Bit {
  int rows_right = 1;
  int batch_size = 0;
  bool needs_prepack = true;
  uint8_t* prepacked_cache = nullptr;
  std::unique_ptr<uint8_t[], Deleter> prepacked_cache_buffer;
  size_t prepacked_cache_buffer_size = 0;
  void AllocatePackedRegion(size_t required_size) {
#ifdef TFLITE_MMAP_DISABLED
    uint8_t* region = new uint8_t[required_size];
    prepacked_cache_buffer =
        std::unique_ptr<uint8_t[], Deleter>(region, Deleter());
#else
    uint8_t* region = reinterpret_cast<uint8_t*>(
        mmap(nullptr, required_size, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    prepacked_cache_buffer =
        std::unique_ptr<uint8_t[], Deleter>(region, Deleter(required_size));
#ifdef MADV_MERGEABLE
    madvise(region, required_size, MADV_MERGEABLE);
#endif
#endif
    prepacked_cache = reinterpret_cast<uint8_t*>(
        (reinterpret_cast<uintptr_t>(prepacked_cache_buffer.get()) +
         kDefaultAlignmentPadding) &
        ~kDefaultAlignmentPadding);
    prepacked_cache_buffer_size = required_size;
  }
};
namespace api {
inline void Prepack(uint8_t* dest, const int8_t* tensor, int layout_rows,
                    int layout_cols, int src_rows, int src_cols, int width,
                    int depth) {
  optimized_4bit::Prepack(dest, tensor, layout_rows, layout_cols, src_rows,
                          src_cols, width, depth);
}
inline void BatchQuantizeFloats4Bit(const float* float_data_ptr, int n_batch,
                                    int n_data, int8_t* quantized_data_ptr,
                                    float* scaling_factors, int width,
                                    int depth, int32_t* input_offsets) {
  optimized_4bit::BatchQuantizeFloats4Bit(float_data_ptr, n_batch, n_data,
                                          quantized_data_ptr, scaling_factors,
                                          width, depth, input_offsets);
}
inline void AssignBiasAndComputeOffsets(const int32_t* input_offsets,
                                        const float* batch_scales,
                                        float* filter_scales,
                                        const float* bias_ptr,
                                        float* output_ptr, int output_depth,
                                        int batch_size) {
  optimized_4bit::AssignBiasAndComputeOffsets(
      input_offsets, batch_scales, filter_scales, bias_ptr, output_ptr,
      output_depth, batch_size);
}
inline void RunAndUnpack(int rhs_width, const uint8_t* lhs, const int8_t* rhs,
                         int32_t* dst, int output_depth, int batch_size,
                         int lhs_layout_rows, int lhs_layout_cols,
                         int rhs_layout_rows, int rhs_layout_cols,
                         int dst_layout_rows, int dst_layout_cols,
                         float* output_ptr, const float* scaling_factors,
                         const float* filter_scales) {
  optimized_4bit::RunAndUnpack(
      rhs_width, lhs, rhs, dst, output_depth, batch_size, lhs_layout_rows,
      lhs_layout_cols, rhs_layout_rows, rhs_layout_cols, dst_layout_rows,
      dst_layout_cols, output_ptr, scaling_factors, filter_scales);
}
}  
}  
}  
#endif  