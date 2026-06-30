#ifndef TENSORFLOW_COMPILER_MLIR_LITE_OFFSET_BUFFER_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_OFFSET_BUFFER_H_
#include <cstdint>
namespace tflite {
inline bool IsValidBufferOffset(const int64_t offset) { return offset > 1; }
}  
#endif  