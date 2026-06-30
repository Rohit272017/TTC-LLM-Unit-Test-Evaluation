#include "tensorflow/core/data/service/url.h"
#include <string>
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/regexp.h"
namespace tensorflow {
namespace data {
URL::URL(absl::string_view url) { Parse(url); }
void URL::Parse(absl::string_view url) {
  absl::string_view regexp = "(.*):([a-zA-Z0-9_]+|%port(_[a-zA-Z0-9_]+)?%)";
  if (!RE2::FullMatch(url, regexp, &host_, &port_)) {
    host_ = std::string(url);
    port_ = "";
  }
}
}  
}  