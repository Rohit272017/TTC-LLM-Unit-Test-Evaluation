#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_COMPILATION_TIMER_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_COMPILATION_TIMER_H_
#include <chrono>  
#include "tensorflow/core/platform/profile_utils/cpu_utils.h"
struct CompilationTimer {
  uint64_t start_cycles =
      tensorflow::profile_utils::CpuUtils::GetCurrentClockCycle();
  uint64_t ElapsedCycles() {
    return tensorflow::profile_utils::CpuUtils::GetCurrentClockCycle() -
           start_cycles;
  }
  int64_t ElapsedCyclesInMilliseconds() {
    std::chrono::duration<double> duration =
        tensorflow::profile_utils::CpuUtils::ConvertClockCycleToTime(
            ElapsedCycles());
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration)
        .count();
  }
};
#endif  