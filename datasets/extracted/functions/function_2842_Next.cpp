#include "quiche/http2/core/array_output_buffer.h"
#include <cstdint>
namespace spdy {
void ArrayOutputBuffer::Next(char** data, int* size) {
  *data = current_;
  *size = capacity_ > 0 ? capacity_ : 0;
}
void ArrayOutputBuffer::AdvanceWritePtr(int64_t count) {
  current_ += count;
  capacity_ -= count;
}
uint64_t ArrayOutputBuffer::BytesFree() const { return capacity_; }
}  