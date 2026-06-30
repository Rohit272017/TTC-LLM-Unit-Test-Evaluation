#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef TF_USE_SNAPPY
#include "snappy.h"
#endif
#include <Windows.h>
#include <processthreadsapi.h>
#include <shlwapi.h>
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/demangle.h"
#include "tsl/platform/host_info.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/mem.h"
#include "tsl/platform/numa.h"
#include "tsl/platform/snappy.h"
#include "tsl/platform/types.h"
namespace tsl {
namespace port {
void InitMain(const char* usage, int* argc, char*** argv) {}
string Hostname() {
  char name[1024];
  DWORD name_size = sizeof(name);
  name[0] = 0;
  if (::GetComputerNameA(name, &name_size)) {
    name[name_size] = 0;
  }
  return name;
}
string JobName() {
  const char* job_name_cs = std::getenv("TF_JOB_NAME");
  if (job_name_cs != nullptr) {
    return string(job_name_cs);
  }
  return "";
}
int64_t JobUid() { return -1; }
int64_t TaskId() { return -1; }
IOStatistics GetIOStatistics() { return IOStatistics(); }
int NumSchedulableCPUs() {
  SYSTEM_INFO system_info;
  GetSystemInfo(&system_info);
  return system_info.dwNumberOfProcessors;
}
int MaxParallelism() { return NumSchedulableCPUs(); }
int MaxParallelism(int numa_node) {
  if (numa_node != port::kNUMANoAffinity) {
    return NumSchedulableCPUs() / port::NUMANumNodes();
  }
  return NumSchedulableCPUs();
}
int NumTotalCPUs() {
  return NumSchedulableCPUs();
}
int GetCurrentCPU() {
  return GetCurrentProcessorNumber();
}
bool NUMAEnabled() {
  return false;
}
int NUMANumNodes() { return 1; }
void NUMASetThreadNodeAffinity(int node) {}
int NUMAGetThreadNodeAffinity() { return kNUMANoAffinity; }
void* NUMAMalloc(int node, size_t size, int minimum_alignment) {
  return tsl::port::AlignedMalloc(size, minimum_alignment);
}
void NUMAFree(void* ptr, size_t size) { tsl::port::Free(ptr); }
int NUMAGetMemAffinity(const void* addr) { return kNUMANoAffinity; }
bool Snappy_Compress(const char* input, size_t length, string* output) {
#ifdef TF_USE_SNAPPY
  output->resize(snappy::MaxCompressedLength(length));
  size_t outlen;
  snappy::RawCompress(input, length, &(*output)[0], &outlen);
  output->resize(outlen);
  return true;
#else
  return false;
#endif
}
bool Snappy_CompressFromIOVec(const struct iovec* iov,
                              size_t uncompressed_length, string* output) {
#ifdef TF_USE_SNAPPY
  output->resize(snappy::MaxCompressedLength(uncompressed_length));
  size_t outlen;
  const snappy::iovec* snappy_iov = reinterpret_cast<const snappy::iovec*>(iov);
  snappy::RawCompressFromIOVec(snappy_iov, uncompressed_length, &(*output)[0],
                               &outlen);
  output->resize(outlen);
  return true;
#else
  return false;
#endif
}
bool Snappy_GetUncompressedLength(const char* input, size_t length,
                                  size_t* result) {
#ifdef TF_USE_SNAPPY
  return snappy::GetUncompressedLength(input, length, result);
#else
  return false;
#endif
}
bool Snappy_Uncompress(const char* input, size_t length, char* output) {
#ifdef TF_USE_SNAPPY
  return snappy::RawUncompress(input, length, output);
#else
  return false;
#endif
}
bool Snappy_UncompressToIOVec(const char* compressed, size_t compressed_length,
                              const struct iovec* iov, size_t iov_cnt) {
#ifdef TF_USE_SNAPPY
  const snappy::iovec* snappy_iov = reinterpret_cast<const snappy::iovec*>(iov);
  return snappy::RawUncompressToIOVec(compressed, compressed_length, snappy_iov,
                                      iov_cnt);
#else
  return false;
#endif
}
string Demangle(const char* mangled) { return mangled; }
double NominalCPUFrequency() {
  DWORD data;
  DWORD data_size = sizeof(data);
#pragma comment(lib, "shlwapi.lib")  
  if (SUCCEEDED(
          SHGetValueA(HKEY_LOCAL_MACHINE,
                      "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                      "~MHz", nullptr, &data, &data_size))) {
    return data * 1e6;  
  }
  return 1.0;
}
int NumHyperthreadsPerCore() {
  static const int ht_per_core = tsl::port::CPUIDNumSMT();
  return (ht_per_core > 0) ? ht_per_core : 1;
}
}  
}  
namespace tsl {
namespace port {
void* AlignedMalloc(size_t size, int minimum_alignment) {
  return _aligned_malloc(size, minimum_alignment);
}
void AlignedFree(void* aligned_memory) { _aligned_free(aligned_memory); }
void AlignedSizedFree(void* aligned_memory, size_t alignment, size_t size) {
  (void)alignment;
  (void)size;
  _aligned_free(aligned_memory);
}
void* Malloc(size_t size) { return malloc(size); }
void* Realloc(void* ptr, size_t size) { return realloc(ptr, size); }
void Free(void* ptr) { free(ptr); }
void MallocExtension_ReleaseToSystem(std::size_t num_bytes) {
}
std::size_t MallocExtension_GetAllocatedSize(const void* p) { return 0; }
MemoryInfo GetMemoryInfo() {
  MemoryInfo mem_info = {INT64_MAX, INT64_MAX};
  MEMORYSTATUSEX statex;
  statex.dwLength = sizeof(statex);
  if (GlobalMemoryStatusEx(&statex)) {
    mem_info.free = statex.ullAvailPhys;
    mem_info.total = statex.ullTotalPhys;
  }
  return mem_info;
}
MemoryBandwidthInfo GetMemoryBandwidthInfo() {
  MemoryBandwidthInfo membw_info = {INT64_MAX};
  return membw_info;
}
}  
}  