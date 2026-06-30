#ifndef TENSORFLOW_CORE_PLATFORM_SNAPPY_H_
#define TENSORFLOW_CORE_PLATFORM_SNAPPY_H_
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/snappy.h"
#if !defined(PLATFORM_WINDOWS)
#include <sys/uio.h>
#else
namespace tensorflow {
using tsl::iovec;
}  
#endif
namespace tensorflow {
namespace port {
using tsl::port::Snappy_Compress;
using tsl::port::Snappy_CompressFromIOVec;
using tsl::port::Snappy_GetUncompressedLength;
using tsl::port::Snappy_Uncompress;
using tsl::port::Snappy_UncompressToIOVec;
}  
}  
#endif  