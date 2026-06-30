#include "tensorstore/util/quote_string.h"
#include <string>
#include <string_view>
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
namespace tensorstore {
std::string QuoteString(std::string_view s) {
  return absl::StrCat(
      "\"", absl::CHexEscape(absl::string_view(s.data(), s.size())), "\"");
}
}  