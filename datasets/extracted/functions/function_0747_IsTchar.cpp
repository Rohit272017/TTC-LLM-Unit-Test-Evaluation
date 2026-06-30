#include "tensorstore/internal/http/http_header.h"
#include <stddef.h>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include "absl/container/btree_map.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "re2/re2.h"
#include "tensorstore/internal/uri_utils.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/str_cat.h"
namespace tensorstore {
namespace internal_http {
namespace {
static inline constexpr internal::AsciiSet kTChar{
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    R"(!#$%&'*+-.)"};
inline bool IsTchar(char ch) { return kTChar.Test(ch); }
inline bool IsOWS(char ch) { return ch == ' ' || ch == '\t'; }
}  
absl::Status ValidateHttpHeader(std::string_view header) {
  static LazyRE2 kHeaderPattern = {
                                   "[!#\\$%&'*+\\-\\.\\^_`|~0-9a-zA-Z]+"
                                   ":"
                                   "[\t\x20-\x7e\x80-\xff]*",
                                   RE2::Latin1};
  if (!RE2::FullMatch(header, *kHeaderPattern)) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Invalid HTTP header: ", tensorstore::QuoteString(header)));
  }
  return absl::OkStatus();
}
size_t AppendHeaderData(absl::btree_multimap<std::string, std::string>& headers,
                        std::string_view data) {
  if (data.empty() || *data.rbegin() != '\n') return data.size();
  for (std::string_view field : absl::StrSplit(data, '\n', absl::SkipEmpty())) {
    if (field.empty() || *field.rbegin() != '\r') break;
    field.remove_suffix(1);
    while (!field.empty() && IsOWS(*field.rbegin())) field.remove_suffix(1);
    if (field.empty()) continue;
    auto it = field.begin();
    for (; it != field.end() && IsTchar(*it); ++it) {
    }
    if (it == field.begin() || it == field.end() || *it != ':') {
      continue;
    }
    std::string field_name = absl::AsciiStrToLower(
        std::string_view(field.data(), std::distance(field.begin(), it)));
    field.remove_prefix(field_name.size() + 1);
    while (!field.empty() && IsOWS(*field.begin())) field.remove_prefix(1);
    headers.emplace(std::move(field_name), std::string(field));
  }
  return data.size();
}
std::optional<std::tuple<size_t, size_t, size_t>> TryParseContentRangeHeader(
    const absl::btree_multimap<std::string, std::string>& headers) {
  auto it = headers.find("content-range");
  if (it == headers.end()) {
    return std::nullopt;
  }
  static LazyRE2 kContentRange1 = {R"(^bytes (\d+)-(\d+)/(\d+))"};
  static LazyRE2 kContentRange2 = {R"(^bytes (\d+)-(\d+)(/[*])?)"};
  std::tuple<size_t, size_t, size_t> result(0, 0, 0);
  if (RE2::FullMatch(it->second, *kContentRange1, &std::get<0>(result),
                     &std::get<1>(result), &std::get<2>(result))) {
    return result;
  }
  if (RE2::FullMatch(it->second, *kContentRange2, &std::get<0>(result),
                     &std::get<1>(result))) {
    return result;
  }
  return std::nullopt;
}
std::optional<bool> TryParseBoolHeader(
    const absl::btree_multimap<std::string, std::string>& headers,
    std::string_view header) {
  auto it = headers.find(header);
  bool result;
  if (it != headers.end() && absl::SimpleAtob(it->second, &result)) {
    return result;
  }
  return std::nullopt;
}
std::optional<size_t> TryGetContentLength(
    const absl::btree_multimap<std::string, std::string>& headers) {
  std::optional<size_t> content_length;
  if (headers.find("transfer-encoding") == headers.end() &&
      headers.find("content-encoding") == headers.end()) {
    content_length = TryParseIntHeader<size_t>(headers, "content-length");
  }
  if (!content_length) {
    auto content_range = TryParseContentRangeHeader(headers);
    if (content_range) {
      content_length =
          1 + std::get<1>(*content_range) - std::get<0>(*content_range);
    }
  }
  return content_length;
}
}  
}  