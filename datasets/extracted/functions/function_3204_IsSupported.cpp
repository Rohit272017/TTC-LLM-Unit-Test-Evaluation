#include "tensorflow/lite/profiling/memory_info.h"
#include <stddef.h>
#include <ostream>
#ifdef __linux__
#include <malloc.h>
#include <sys/resource.h>
#include <sys/time.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <malloc/malloc.h>
#endif
namespace tflite {
namespace profiling {
namespace memory {
const size_t MemoryUsage::kValueNotSet = 0;
bool MemoryUsage::IsSupported() {
#if defined(__linux__) || defined(__APPLE__)
  return true;
#endif
  return false;
}
MemoryUsage GetMemoryUsage() {
  MemoryUsage result;
#ifdef __linux__
  rusage res;
  if (getrusage(RUSAGE_SELF, &res) == 0) {
    result.mem_footprint_kb = res.ru_maxrss;
  }
#if defined(__NO_MALLINFO__)
  result.total_allocated_bytes = -1;
  result.in_use_allocated_bytes = -1;
#elif defined(__GLIBC__) && __GLIBC_MINOR__ >= 33
  const auto mem = mallinfo2();
  result.total_allocated_bytes = mem.arena;
  result.in_use_allocated_bytes = mem.uordblks;
#else
  const auto mem = mallinfo();
  result.total_allocated_bytes = mem.arena;
  result.in_use_allocated_bytes = mem.uordblks;
#endif  
#elif defined(__APPLE__)
  struct task_vm_info vm_info;
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  auto status = task_info(mach_task_self(), TASK_VM_INFO,
                          reinterpret_cast<task_info_t>(&vm_info), &count);
  if (status == KERN_SUCCESS) {
    result.mem_footprint_kb =
        static_cast<int64_t>(vm_info.phys_footprint / 1024.0);
  }
  struct mstats stats = mstats();
  result.total_allocated_bytes = stats.bytes_total;
  result.in_use_allocated_bytes = stats.bytes_used;
#endif  
  return result;
}
void MemoryUsage::AllStatsToStream(std::ostream* stream) const {
  *stream << "max resident set size/physical footprint = "
          << mem_footprint_kb / 1000.0 << " MB, total non-mmapped heap size = "
          << total_allocated_bytes / 1000.0 / 1000.0
          << " MB, in-use heap size = "
          << in_use_allocated_bytes / 1000.0 / 1000.0 << " MB";
}
}  
}  
}  