#ifndef TENSORFLOW_CORE_PLATFORM_NUMA_H_
#define TENSORFLOW_CORE_PLATFORM_NUMA_H_
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/numa.h"
namespace tensorflow {
namespace port {
using tsl::port::kNUMANoAffinity;
using tsl::port::NUMAEnabled;
using tsl::port::NUMAFree;
using tsl::port::NUMAGetMemAffinity;
using tsl::port::NUMAGetThreadNodeAffinity;
using tsl::port::NUMAMalloc;
using tsl::port::NUMANumNodes;
using tsl::port::NUMASetThreadNodeAffinity;
}  
}  
#endif  