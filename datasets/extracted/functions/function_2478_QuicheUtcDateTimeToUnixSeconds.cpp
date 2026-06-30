#ifndef QUICHE_COMMON_PLATFORM_API_QUICHE_TIME_UTILS_H_
#define QUICHE_COMMON_PLATFORM_API_QUICHE_TIME_UTILS_H_
#include <cstdint>
#include "quiche_platform_impl/quiche_time_utils_impl.h"
namespace quiche {
inline std::optional<int64_t> QuicheUtcDateTimeToUnixSeconds(
    int year, int month, int day, int hour, int minute, int second) {
  return QuicheUtcDateTimeToUnixSecondsImpl(year, month, day, hour, minute,
                                            second);
}
}  
#endif  