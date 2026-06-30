#include "tsl/platform/cloud/time_util.h"
#include <time.h>
#include <cmath>
#include <cstdio>
#include <ctime>
#ifdef _WIN32
#define timegm _mkgmtime
#endif
#include "tsl/platform/errors.h"
namespace tsl {
namespace {
constexpr int64_t kNanosecondsPerSecond = 1000 * 1000 * 1000;
}  
absl::Status ParseRfc3339Time(const string& time, int64_t* mtime_nsec) {
  tm parsed{0};
  float seconds;
  if (sscanf(time.c_str(), "%4d-%2d-%2dT%2d:%2d:%fZ", &(parsed.tm_year),
             &(parsed.tm_mon), &(parsed.tm_mday), &(parsed.tm_hour),
             &(parsed.tm_min), &seconds) != 6) {
    return errors::Internal(
        strings::StrCat("Unrecognized RFC 3339 time format: ", time));
  }
  const int int_seconds = std::floor(seconds);
  parsed.tm_year -= 1900;  
  parsed.tm_mon -= 1;      
  parsed.tm_sec = int_seconds;
  *mtime_nsec = timegm(&parsed) * kNanosecondsPerSecond +
                static_cast<int64_t>(std::floor((seconds - int_seconds) *
                                                kNanosecondsPerSecond));
  return absl::OkStatus();
}
}  