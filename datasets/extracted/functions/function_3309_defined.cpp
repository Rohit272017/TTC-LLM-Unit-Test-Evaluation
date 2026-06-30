#include "tensorflow/lite/profiling/time.h"
#if defined(_MSC_VER)
#include <chrono>  
#include <thread>  
#else
#include <sys/time.h>
#include <time.h>
#endif
namespace tflite {
namespace profiling {
namespace time {
#if defined(_MSC_VER)
uint64_t NowMicros() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}
void SleepForMicros(uint64_t micros) {
  std::this_thread::sleep_for(std::chrono::microseconds(micros));
}
#else
uint64_t NowMicros() {
#if defined(__APPLE__)
  return clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW) / 1e3;
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1e6 +
         static_cast<uint64_t>(ts.tv_nsec) / 1e3;
#endif  
}
void SleepForMicros(uint64_t micros) {
  timespec sleep_time;
  sleep_time.tv_sec = micros / 1e6;
  micros -= sleep_time.tv_sec * 1e6;
  sleep_time.tv_nsec = micros * 1e3;
  nanosleep(&sleep_time, nullptr);
}
#endif  
}  
}  
}  