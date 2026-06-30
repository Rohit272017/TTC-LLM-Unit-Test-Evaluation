#include "internal/time.h"
#include <cstdint>
#include <string>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "internal/status_macros.h"
namespace cel::internal {
namespace {
std::string RawFormatTimestamp(absl::Time timestamp) {
  return absl::FormatTime("%Y-%m-%d%ET%H:%M:%E*SZ", timestamp,
                          absl::UTCTimeZone());
}
}  
absl::Status ValidateDuration(absl::Duration duration) {
  if (duration < MinDuration()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Duration \"", absl::FormatDuration(duration),
                     "\" below minimum allowed duration \"",
                     absl::FormatDuration(MinDuration()), "\""));
  }
  if (duration > MaxDuration()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Duration \"", absl::FormatDuration(duration),
                     "\" above maximum allowed duration \"",
                     absl::FormatDuration(MaxDuration()), "\""));
  }
  return absl::OkStatus();
}
absl::StatusOr<absl::Duration> ParseDuration(absl::string_view input) {
  absl::Duration duration;
  if (!absl::ParseDuration(input, &duration)) {
    return absl::InvalidArgumentError("Failed to parse duration from string");
  }
  return duration;
}
absl::StatusOr<std::string> FormatDuration(absl::Duration duration) {
  CEL_RETURN_IF_ERROR(ValidateDuration(duration));
  return absl::FormatDuration(duration);
}
std::string DebugStringDuration(absl::Duration duration) {
  return absl::FormatDuration(duration);
}
absl::Status ValidateTimestamp(absl::Time timestamp) {
  if (timestamp < MinTimestamp()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Timestamp \"", RawFormatTimestamp(timestamp),
                     "\" below minimum allowed timestamp \"",
                     RawFormatTimestamp(MinTimestamp()), "\""));
  }
  if (timestamp > MaxTimestamp()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Timestamp \"", RawFormatTimestamp(timestamp),
                     "\" above maximum allowed timestamp \"",
                     RawFormatTimestamp(MaxTimestamp()), "\""));
  }
  return absl::OkStatus();
}
absl::StatusOr<absl::Time> ParseTimestamp(absl::string_view input) {
  absl::Time timestamp;
  std::string err;
  if (!absl::ParseTime(absl::RFC3339_full, input, absl::UTCTimeZone(),
                       &timestamp, &err)) {
    return err.empty() ? absl::InvalidArgumentError(
                             "Failed to parse timestamp from string")
                       : absl::InvalidArgumentError(absl::StrCat(
                             "Failed to parse timestamp from string: ", err));
  }
  CEL_RETURN_IF_ERROR(ValidateTimestamp(timestamp));
  return timestamp;
}
absl::StatusOr<std::string> FormatTimestamp(absl::Time timestamp) {
  CEL_RETURN_IF_ERROR(ValidateTimestamp(timestamp));
  return RawFormatTimestamp(timestamp);
}
std::string FormatNanos(int32_t nanos) {
  constexpr int32_t kNanosPerMillisecond = 1000000;
  constexpr int32_t kNanosPerMicrosecond = 1000;
  if (nanos % kNanosPerMillisecond == 0) {
    return absl::StrFormat("%03d", nanos / kNanosPerMillisecond);
  } else if (nanos % kNanosPerMicrosecond == 0) {
    return absl::StrFormat("%06d", nanos / kNanosPerMicrosecond);
  }
  return absl::StrFormat("%09d", nanos);
}
absl::StatusOr<std::string> EncodeDurationToJson(absl::Duration duration) {
  CEL_RETURN_IF_ERROR(ValidateDuration(duration));
  std::string result;
  int64_t seconds = absl::IDivDuration(duration, absl::Seconds(1), &duration);
  int64_t nanos = absl::IDivDuration(duration, absl::Nanoseconds(1), &duration);
  if (seconds < 0 || nanos < 0) {
    result = "-";
    seconds = -seconds;
    nanos = -nanos;
  }
  absl::StrAppend(&result, seconds);
  if (nanos != 0) {
    absl::StrAppend(&result, ".", FormatNanos(nanos));
  }
  absl::StrAppend(&result, "s");
  return result;
}
absl::StatusOr<std::string> EncodeTimestampToJson(absl::Time timestamp) {
  static constexpr absl::string_view kTimestampFormat = "%E4Y-%m-%dT%H:%M:%S";
  CEL_RETURN_IF_ERROR(ValidateTimestamp(timestamp));
  absl::Time unix_seconds =
      absl::FromUnixSeconds(absl::ToUnixSeconds(timestamp));
  int64_t n = (timestamp - unix_seconds) / absl::Nanoseconds(1);
  std::string result =
      absl::FormatTime(kTimestampFormat, unix_seconds, absl::UTCTimeZone());
  if (n > 0) {
    absl::StrAppend(&result, ".", FormatNanos(n));
  }
  absl::StrAppend(&result, "Z");
  return result;
}
std::string DebugStringTimestamp(absl::Time timestamp) {
  return RawFormatTimestamp(timestamp);
}
}  