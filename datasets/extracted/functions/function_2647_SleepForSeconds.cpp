#include "tensorflow/lite/tools/benchmark/benchmark_utils.h"
#include "tensorflow/lite/profiling/time.h"
namespace tflite {
namespace benchmark {
namespace util {
void SleepForSeconds(double sleep_seconds) {
  if (sleep_seconds <= 0.0) {
    return;
  }
  tflite::profiling::time::SleepForMicros(
      static_cast<uint64_t>(sleep_seconds * 1e6));
}
}  
}  
}  