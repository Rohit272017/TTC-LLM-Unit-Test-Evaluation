#ifndef TENSORFLOW_TSL_PLATFORM_NUMA_H_
#define TENSORFLOW_TSL_PLATFORM_NUMA_H_
#include "tsl/platform/platform.h"
#include "tsl/platform/types.h"
namespace tsl {
namespace port {
bool NUMAEnabled();
int NUMANumNodes();
static const int kNUMANoAffinity = -1;
void NUMASetThreadNodeAffinity(int node);
int NUMAGetThreadNodeAffinity();
void* NUMAMalloc(int node, size_t size, int minimum_alignment);
void NUMAFree(void* ptr, size_t size);
int NUMAGetMemAffinity(const void* ptr);
}  
}  
#endif  