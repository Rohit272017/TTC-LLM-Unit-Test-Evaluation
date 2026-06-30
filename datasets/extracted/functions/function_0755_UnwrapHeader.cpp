#include "validating_util.h"
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include "util/md5.h"
namespace i18n {
namespace addressinput {
namespace {
const char kTimestampPrefix[] = "timestamp=";
const size_t kTimestampPrefixLength = sizeof kTimestampPrefix - 1;
const char kChecksumPrefix[] = "checksum=";
const size_t kChecksumPrefixLength = sizeof kChecksumPrefix - 1;
const char kSeparator = '\n';
bool UnwrapHeader(const char* header_prefix,
                  size_t header_prefix_length,
                  std::string* data,
                  std::string* header_value) {
  assert(header_prefix != nullptr);
  assert(data != nullptr);
  assert(header_value != nullptr);
  if (data->compare(
          0, header_prefix_length, header_prefix, header_prefix_length) != 0) {
    return false;
  }
  std::string::size_type separator_position =
      data->find(kSeparator, header_prefix_length);
  if (separator_position == std::string::npos) {
    return false;
  }
  header_value->assign(
      *data, header_prefix_length, separator_position - header_prefix_length);
  data->erase(0, separator_position + 1);
  return true;
}
}  
void ValidatingUtil::Wrap(time_t timestamp, std::string* data) {
  assert(data != nullptr);
  char timestamp_string[2 + 3 * sizeof timestamp];
  int size = std::snprintf(timestamp_string, sizeof(timestamp_string), "%ld",
                           static_cast<long>(timestamp));
  assert(size > 0);
  assert(size < sizeof timestamp_string);
  (void)size;
  std::string header;
  header.append(kTimestampPrefix, kTimestampPrefixLength);
  header.append(timestamp_string);
  header.push_back(kSeparator);
  header.append(kChecksumPrefix, kChecksumPrefixLength);
  header.append(MD5String(*data));
  header.push_back(kSeparator);
  data->reserve(header.size() + data->size());
  data->insert(0, header);
}
bool ValidatingUtil::UnwrapTimestamp(std::string* data, time_t now) {
  assert(data != nullptr);
  if (now < 0) {
    return false;
  }
  std::string timestamp_string;
  if (!UnwrapHeader(
          kTimestampPrefix, kTimestampPrefixLength, data, &timestamp_string)) {
    return false;
  }
  time_t timestamp = atol(timestamp_string.c_str());
  if (timestamp < 0) {
    return false;
  }
  static const double kOneMonthInSeconds = 30.0 * 24.0 * 60.0 * 60.0;
  double age_in_seconds = difftime(now, timestamp);
  return !(age_in_seconds < 0.0) && age_in_seconds < kOneMonthInSeconds;
}
bool ValidatingUtil::UnwrapChecksum(std::string* data) {
  assert(data != nullptr);
  std::string checksum;
  if (!UnwrapHeader(kChecksumPrefix, kChecksumPrefixLength, data, &checksum)) {
    return false;
  }
  return checksum == MD5String(*data);
}
}  
}  