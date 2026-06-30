#include "tensorstore/util/status_testutil.h"
#include <ostream>
#include <regex>  
#include <string>
#include <system_error>  
#include <gmock/gmock.h>
#include "absl/status/status.h"
namespace tensorstore {
namespace internal_status {
namespace {
template <typename StringType>
class RegexMatchImpl : public ::testing::MatcherInterface<StringType> {
 public:
  RegexMatchImpl(const std::string& message_pattern)
      : message_pattern_(message_pattern) {}
  void DescribeTo(std::ostream* os) const override {
    *os << "message matches pattern ";
    ::testing::internal::UniversalPrint(message_pattern_, os);
  }
  void DescribeNegationTo(std::ostream* os) const override {
    *os << "message doesn't match pattern ";
    ::testing::internal::UniversalPrint(message_pattern_, os);
  }
  bool MatchAndExplain(
      StringType message,
      ::testing::MatchResultListener* result_listener) const override {
    return std::regex_match(message, std::regex(message_pattern_));
  }
 private:
  const std::string message_pattern_;
};
}  
}  
internal_status::StatusIsMatcher MatchesStatus(
    absl::StatusCode status_code, const std::string& message_pattern) {
  return internal_status::StatusIsMatcher(
      status_code, ::testing::Matcher<const std::string&>(
                       new internal_status::RegexMatchImpl<const std::string&>(
                           message_pattern)));
}
}  