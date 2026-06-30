#include "quiche/common/simple_buffer_allocator.h"
namespace quiche {
char* SimpleBufferAllocator::New(size_t size) { return new char[size]; }
char* SimpleBufferAllocator::New(size_t size, bool ) {
  return New(size);
}
void SimpleBufferAllocator::Delete(char* buffer) { delete[] buffer; }
}  