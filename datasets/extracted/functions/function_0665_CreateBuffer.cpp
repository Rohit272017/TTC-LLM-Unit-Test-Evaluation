#include "arolla/util/preallocated_buffers.h"
#include <cstring>
#include "arolla/util/memory.h"
namespace arolla {
namespace {
const void* CreateBuffer() {
  auto alloc = AlignedAlloc(Alignment{kZeroInitializedBufferAlignment},
                            kZeroInitializedBufferSize);
  std::memset(alloc.get(), 0, kZeroInitializedBufferSize);
  return alloc.release();
}
}  
const void* GetZeroInitializedBuffer() {
  static const void* const kBuffer = CreateBuffer();
  return kBuffer;
}
}  