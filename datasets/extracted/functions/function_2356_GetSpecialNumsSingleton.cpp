#include "tsl/platform/numbers.h"
#include <ctype.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <locale>
#include <unordered_map>
#include "double-conversion/double-conversion.h"
#include "tsl/platform/str_util.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/stringprintf.h"
#include "tsl/platform/types.h"
namespace tsl {
namespace {
template <typename T>
const std::unordered_map<std::string, T>* GetSpecialNumsSingleton() {
  static const std::unordered_map<std::string, T>* special_nums =
      CHECK_NOTNULL((new const std::unordered_map<std::string, T>{
          {"inf", std::numeric_limits<T>::infinity()},
          {"+inf", std::numeric_limits<T>::infinity()},
          {"-inf", -std::numeric_limits<T>::infinity()},
          {"infinity", std::numeric_limits<T>::infinity()},
          {"+infinity", std::numeric_limits<T>::infinity()},
          {"-infinity", -std::numeric_limits<T>::infinity()},
          {"nan", std::numeric_limits<T>::quiet_NaN()},
          {"+nan", std::numeric_limits<T>::quiet_NaN()},
          {"-nan", -std::numeric_limits<T>::quiet_NaN()},
      }));
  return special_nums;
}
template <typename T>
T locale_independent_strtonum(const char* str, const char** endptr) {
  auto special_nums = GetSpecialNumsSingleton<T>();
  std::stringstream s(str);
  std::string special_num_str;
  s >> special_num_str;
  for (size_t i = 0; i < special_num_str.length(); ++i) {
    special_num_str[i] =
        std::tolower(special_num_str[i], std::locale::classic());
  }
  auto entry = special_nums->find(special_num_str);
  if (entry != special_nums->end()) {
    *endptr = str + (s.eof() ? static_cast<std::iostream::pos_type>(strlen(str))
                             : s.tellg());
    return entry->second;
  } else {
    if (special_num_str.compare(0, 2, "0x") == 0 ||
        special_num_str.compare(0, 3, "-0x") == 0) {
      return strtol(str, const_cast<char**>(endptr), 16);
    }
  }
  s.str(str);
  s.clear();
  s.imbue(std::locale::classic());
  T result;
  s >> result;
  if (s.fail()) {
    if (result == std::numeric_limits<T>::max() ||
        result == std::numeric_limits<T>::infinity()) {
      result = std::numeric_limits<T>::infinity();
      s.clear(s.rdstate() & ~std::ios::failbit);
    } else if (result == -std::numeric_limits<T>::max() ||
               result == -std::numeric_limits<T>::infinity()) {
      result = -std::numeric_limits<T>::infinity();
      s.clear(s.rdstate() & ~std::ios::failbit);
    }
  }
  if (endptr) {
    *endptr =
        str +
        (s.fail() ? static_cast<std::iostream::pos_type>(0)
                  : (s.eof() ? static_cast<std::iostream::pos_type>(strlen(str))
                             : s.tellg()));
  }
  return result;
}
static inline const double_conversion::StringToDoubleConverter&
StringToFloatConverter() {
  static const double_conversion::StringToDoubleConverter converter(
      double_conversion::StringToDoubleConverter::ALLOW_LEADING_SPACES |
          double_conversion::StringToDoubleConverter::ALLOW_HEX |
          double_conversion::StringToDoubleConverter::ALLOW_TRAILING_SPACES |
          double_conversion::StringToDoubleConverter::ALLOW_CASE_INSENSIBILITY,
      0., 0., "inf", "nan");
  return converter;
}
}  
namespace strings {
size_t FastInt32ToBufferLeft(int32_t i, char* buffer) {
  uint32_t u = i;
  size_t length = 0;
  if (i < 0) {
    *buffer++ = '-';
    ++length;
    u = 0 - u;
  }
  length += FastUInt32ToBufferLeft(u, buffer);
  return length;
}
size_t FastUInt32ToBufferLeft(uint32_t i, char* buffer) {
  char* start = buffer;
  do {
    *buffer++ = ((i % 10) + '0');
    i /= 10;
  } while (i > 0);
  *buffer = 0;
  std::reverse(start, buffer);
  return buffer - start;
}
size_t FastInt64ToBufferLeft(int64_t i, char* buffer) {
  uint64_t u = i;
  size_t length = 0;
  if (i < 0) {
    *buffer++ = '-';
    ++length;
    u = 0 - u;
  }
  length += FastUInt64ToBufferLeft(u, buffer);
  return length;
}
size_t FastUInt64ToBufferLeft(uint64_t i, char* buffer) {
  char* start = buffer;
  do {
    *buffer++ = ((i % 10) + '0');
    i /= 10;
  } while (i > 0);
  *buffer = 0;
  std::reverse(start, buffer);
  return buffer - start;
}
static const double kDoublePrecisionCheckMax = DBL_MAX / 1.000000000000001;
size_t DoubleToBuffer(double value, char* buffer) {
  static_assert(DBL_DIG < 20, "DBL_DIG is too big");
  if (std::isnan(value)) {
    int snprintf_result = snprintf(buffer, kFastToBufferSize, "%snan",
                                   std::signbit(value) ? "-" : "");
    DCHECK(snprintf_result > 0 && snprintf_result < kFastToBufferSize);
    return snprintf_result;
  }
  if (std::abs(value) <= kDoublePrecisionCheckMax) {
    int snprintf_result =
        snprintf(buffer, kFastToBufferSize, "%.*g", DBL_DIG, value);
    DCHECK(snprintf_result > 0 && snprintf_result < kFastToBufferSize);
    if (locale_independent_strtonum<double>(buffer, nullptr) == value) {
      return snprintf_result;
    }
  }
  int snprintf_result =
      snprintf(buffer, kFastToBufferSize, "%.*g", DBL_DIG + 2, value);
  DCHECK(snprintf_result > 0 && snprintf_result < kFastToBufferSize);
  return snprintf_result;
}
namespace {
char SafeFirstChar(absl::string_view str) {
  if (str.empty()) return '\0';
  return str[0];
}
void SkipSpaces(absl::string_view* str) {
  while (isspace(SafeFirstChar(*str))) str->remove_prefix(1);
}
}  
bool safe_strto64(absl::string_view str, int64_t* value) {
  SkipSpaces(&str);
  int64_t vlimit = kint64max;
  int sign = 1;
  if (absl::ConsumePrefix(&str, "-")) {
    sign = -1;
    vlimit = kint64min;
  }
  if (!isdigit(SafeFirstChar(str))) return false;
  int64_t result = 0;
  if (sign == 1) {
    do {
      int digit = SafeFirstChar(str) - '0';
      if ((vlimit - digit) / 10 < result) {
        return false;
      }
      result = result * 10 + digit;
      str.remove_prefix(1);
    } while (isdigit(SafeFirstChar(str)));
  } else {
    do {
      int digit = SafeFirstChar(str) - '0';
      if ((vlimit + digit) / 10 > result) {
        return false;
      }
      result = result * 10 - digit;
      str.remove_prefix(1);
    } while (isdigit(SafeFirstChar(str)));
  }
  SkipSpaces(&str);
  if (!str.empty()) return false;
  *value = result;
  return true;
}
bool safe_strtou64(absl::string_view str, uint64_t* value) {
  SkipSpaces(&str);
  if (!isdigit(SafeFirstChar(str))) return false;
  uint64_t result = 0;
  do {
    int digit = SafeFirstChar(str) - '0';
    if ((kuint64max - digit) / 10 < result) {
      return false;
    }
    result = result * 10 + digit;
    str.remove_prefix(1);
  } while (isdigit(SafeFirstChar(str)));
  SkipSpaces(&str);
  if (!str.empty()) return false;
  *value = result;
  return true;
}
bool safe_strto32(absl::string_view str, int32_t* value) {
  SkipSpaces(&str);
  int64_t vmax = kint32max;
  int sign = 1;
  if (absl::ConsumePrefix(&str, "-")) {
    sign = -1;
    ++vmax;
  }
  if (!isdigit(SafeFirstChar(str))) return false;
  int64_t result = 0;
  do {
    result = result * 10 + SafeFirstChar(str) - '0';
    if (result > vmax) {
      return false;
    }
    str.remove_prefix(1);
  } while (isdigit(SafeFirstChar(str)));
  SkipSpaces(&str);
  if (!str.empty()) return false;
  *value = static_cast<int32_t>(result * sign);
  return true;
}
bool safe_strtou32(absl::string_view str, uint32_t* value) {
  SkipSpaces(&str);
  if (!isdigit(SafeFirstChar(str))) return false;
  int64_t result = 0;
  do {
    result = result * 10 + SafeFirstChar(str) - '0';
    if (result > kuint32max) {
      return false;
    }
    str.remove_prefix(1);
  } while (isdigit(SafeFirstChar(str)));
  SkipSpaces(&str);
  if (!str.empty()) return false;
  *value = static_cast<uint32_t>(result);
  return true;
}
bool safe_strtof(absl::string_view str, float* value) {
  int processed_characters_count = -1;
  auto len = str.size();
  if (len >= kFastToBufferSize) return false;
  if (len > std::numeric_limits<int>::max()) return false;
  *value = StringToFloatConverter().StringToFloat(
      str.data(), static_cast<int>(len), &processed_characters_count);
  return processed_characters_count > 0;
}
bool safe_strtod(absl::string_view str, double* value) {
  int processed_characters_count = -1;
  auto len = str.size();
  if (len >= kFastToBufferSize) return false;
  if (len > std::numeric_limits<int>::max()) return false;
  *value = StringToFloatConverter().StringToDouble(
      str.data(), static_cast<int>(len), &processed_characters_count);
  return processed_characters_count > 0;
}
size_t FloatToBuffer(float value, char* buffer) {
  static_assert(FLT_DIG < 10, "FLT_DIG is too big");
  if (std::isnan(value)) {
    int snprintf_result = snprintf(buffer, kFastToBufferSize, "%snan",
                                   std::signbit(value) ? "-" : "");
    DCHECK(snprintf_result > 0 && snprintf_result < kFastToBufferSize);
    return snprintf_result;
  }
  int snprintf_result =
      snprintf(buffer, kFastToBufferSize, "%.*g", FLT_DIG, value);
  DCHECK(snprintf_result > 0 && snprintf_result < kFastToBufferSize);
  float parsed_value;
  if (!safe_strtof(buffer, &parsed_value) || parsed_value != value) {
    snprintf_result =
        snprintf(buffer, kFastToBufferSize, "%.*g", FLT_DIG + 3, value);
    DCHECK(snprintf_result > 0 && snprintf_result < kFastToBufferSize);
  }
  return snprintf_result;
}
std::string FpToString(Fprint fp) {
  char buf[17];
  snprintf(buf, sizeof(buf), "%016llx", static_cast<long long>(fp));
  return std::string(buf);
}
bool StringToFp(const std::string& s, Fprint* fp) {
  char junk;
  uint64_t result;
  if (sscanf(s.c_str(), "%" SCNx64 "%c", &result, &junk) == 1) {
    *fp = result;
    return true;
  } else {
    return false;
  }
}
absl::string_view Uint64ToHexString(uint64_t v, char* buf) {
  static const char* hexdigits = "0123456789abcdef";
  const int num_byte = 16;
  buf[num_byte] = '\0';
  for (int i = num_byte - 1; i >= 0; i--) {
    buf[i] = hexdigits[v & 0xf];
    v >>= 4;
  }
  return absl::string_view(buf, num_byte);
}
bool HexStringToUint64(const absl::string_view& s, uint64_t* result) {
  uint64_t v = 0;
  if (s.empty()) {
    return false;
  }
  for (size_t i = 0; i < s.size(); i++) {
    char c = s[i];
    if (c >= '0' && c <= '9') {
      v = (v << 4) + (c - '0');
    } else if (c >= 'a' && c <= 'f') {
      v = (v << 4) + 10 + (c - 'a');
    } else if (c >= 'A' && c <= 'F') {
      v = (v << 4) + 10 + (c - 'A');
    } else {
      return false;
    }
  }
  *result = v;
  return true;
}
std::string HumanReadableNum(int64_t value) {
  std::string s;
  if (value < 0) {
    s += "-";
    value = -value;
  }
  if (value < 1000) {
    Appendf(&s, "%lld", static_cast<long long>(value));
  } else if (value >= static_cast<int64_t>(1e15)) {
    Appendf(&s, "%0.3G", static_cast<double>(value));
  } else {
    static const char units[] = "kMBT";
    const char* unit = units;
    while (value >= static_cast<int64_t>(1000000)) {
      value /= static_cast<int64_t>(1000);
      ++unit;
      CHECK(unit < units + TF_ARRAYSIZE(units));
    }
    Appendf(&s, "%.2f%c", value / 1000.0, *unit);
  }
  return s;
}
std::string HumanReadableNumBytes(int64_t num_bytes) {
  if (num_bytes == kint64min) {
    return "-8E";
  }
  const char* neg_str = (num_bytes < 0) ? "-" : "";
  if (num_bytes < 0) {
    num_bytes = -num_bytes;
  }
  if (num_bytes < 1024) {
    char buf[8];  
    snprintf(buf, sizeof(buf), "%s%lldB", neg_str,
             static_cast<long long>(num_bytes));
    return std::string(buf);
  }
  static const char units[] = "KMGTPE";  
  const char* unit = units;
  while (num_bytes >= static_cast<int64_t>(1024) * 1024) {
    num_bytes /= 1024;
    ++unit;
    CHECK(unit < units + TF_ARRAYSIZE(units));
  }
  char buf[16];
  snprintf(buf, sizeof(buf), ((*unit == 'K') ? "%s%.1f%ciB" : "%s%.2f%ciB"),
           neg_str, num_bytes / 1024.0, *unit);
  return std::string(buf);
}
std::string HumanReadableElapsedTime(double seconds) {
  std::string human_readable;
  if (seconds < 0) {
    human_readable = "-";
    seconds = -seconds;
  }
  const double microseconds = seconds * 1.0e6;
  if (microseconds < 999.5) {
    strings::Appendf(&human_readable, "%0.3g us", microseconds);
    return human_readable;
  }
  double milliseconds = seconds * 1e3;
  if (milliseconds >= .995 && milliseconds < 1) {
    milliseconds = 1.0;
  }
  if (milliseconds < 999.5) {
    strings::Appendf(&human_readable, "%0.3g ms", milliseconds);
    return human_readable;
  }
  if (seconds < 60.0) {
    strings::Appendf(&human_readable, "%0.3g s", seconds);
    return human_readable;
  }
  seconds /= 60.0;
  if (seconds < 60.0) {
    strings::Appendf(&human_readable, "%0.3g min", seconds);
    return human_readable;
  }
  seconds /= 60.0;
  if (seconds < 24.0) {
    strings::Appendf(&human_readable, "%0.3g h", seconds);
    return human_readable;
  }
  seconds /= 24.0;
  if (seconds < 30.0) {
    strings::Appendf(&human_readable, "%0.3g days", seconds);
    return human_readable;
  }
  if (seconds < 365.2425) {
    strings::Appendf(&human_readable, "%0.3g months", seconds / 30.436875);
    return human_readable;
  }
  seconds /= 365.2425;
  strings::Appendf(&human_readable, "%0.3g years", seconds);
  return human_readable;
}
}  
}  