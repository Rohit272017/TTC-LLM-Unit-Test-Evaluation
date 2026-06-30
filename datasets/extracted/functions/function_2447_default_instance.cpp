#include "common/constant.h"
#include <cmath>
#include <cstdint>
#include <string>
#include "absl/base/no_destructor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "internal/strings.h"
namespace cel {
const BytesConstant& BytesConstant::default_instance() {
  static const absl::NoDestructor<BytesConstant> instance;
  return *instance;
}
const StringConstant& StringConstant::default_instance() {
  static const absl::NoDestructor<StringConstant> instance;
  return *instance;
}
const Constant& Constant::default_instance() {
  static const absl::NoDestructor<Constant> instance;
  return *instance;
}
std::string FormatNullConstant() { return "null"; }
std::string FormatBoolConstant(bool value) {
  return value ? std::string("true") : std::string("false");
}
std::string FormatIntConstant(int64_t value) { return absl::StrCat(value); }
std::string FormatUintConstant(uint64_t value) {
  return absl::StrCat(value, "u");
}
std::string FormatDoubleConstant(double value) {
  if (std::isfinite(value)) {
    if (std::floor(value) != value) {
      return absl::StrCat(value);
    }
    std::string stringified = absl::StrCat(value);
    if (!absl::StrContains(stringified, '.')) {
      absl::StrAppend(&stringified, ".0");
    }
    return stringified;
  }
  if (std::isnan(value)) {
    return "nan";
  }
  if (std::signbit(value)) {
    return "-infinity";
  }
  return "+infinity";
}
std::string FormatBytesConstant(absl::string_view value) {
  return internal::FormatBytesLiteral(value);
}
std::string FormatStringConstant(absl::string_view value) {
  return internal::FormatStringLiteral(value);
}
std::string FormatDurationConstant(absl::Duration value) {
  return absl::StrCat("duration(\"", absl::FormatDuration(value), "\")");
}
std::string FormatTimestampConstant(absl::Time value) {
  return absl::StrCat(
      "timestamp(\"",
      absl::FormatTime("%Y-%m-%d%ET%H:%M:%E*SZ", value, absl::UTCTimeZone()),
      "\")");
}
}  