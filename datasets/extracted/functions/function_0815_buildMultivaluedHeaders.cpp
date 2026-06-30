#include "quiche/balsa/header_properties.h"
#include <array>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "quiche/common/quiche_text_utils.h"
namespace quiche::header_properties {
namespace {
using MultivaluedHeadersSet =
    absl::flat_hash_set<absl::string_view, StringPieceCaseHash,
                        StringPieceCaseEqual>;
MultivaluedHeadersSet* buildMultivaluedHeaders() {
  return new MultivaluedHeadersSet({
      "accept",
      "accept-charset",
      "accept-encoding",
      "accept-language",
      "accept-ranges",
      "access-control-allow-headers",
      "access-control-allow-methods",
      "access-control-expose-headers",
      "access-control-request-headers",
      "allow",
      "cache-control",
      "cdn-cache-control",
      "connection",
      "content-encoding",
      "content-language",
      "expect",
      "if-match",
      "if-none-match",
      "link",
      "pragma",
      "proxy-authenticate",
      "te",
      "sec-websocket-extensions",
      "set-cookie",
      "trailer",
      "transfer-encoding",
      "upgrade",
      "vary",
      "via",
      "warning",
      "www-authenticate",
      "x-forwarded-for",
      "x-go"  "ogle-cache-control",
  });
}
std::array<bool, 256> buildInvalidHeaderKeyCharLookupTable() {
  std::array<bool, 256> invalidCharTable;
  invalidCharTable.fill(false);
  for (uint8_t c : kInvalidHeaderKeyCharList) {
    invalidCharTable[c] = true;
  }
  return invalidCharTable;
}
std::array<bool, 256> buildInvalidHeaderKeyCharLookupTableAllowDoubleQuote() {
  std::array<bool, 256> invalidCharTable;
  invalidCharTable.fill(false);
  for (uint8_t c : kInvalidHeaderKeyCharListAllowDoubleQuote) {
    invalidCharTable[c] = true;
  }
  return invalidCharTable;
}
std::array<bool, 256> buildInvalidCharLookupTable() {
  std::array<bool, 256> invalidCharTable;
  invalidCharTable.fill(false);
  for (uint8_t c : kInvalidHeaderCharList) {
    invalidCharTable[c] = true;
  }
  return invalidCharTable;
}
std::array<bool, 256> buildInvalidPathCharLookupTable() {
  std::array<bool, 256> invalidCharTable;
  invalidCharTable.fill(true);
  for (uint8_t c : kValidPathCharList) {
    invalidCharTable[c] = false;
  }
  return invalidCharTable;
}
}  
bool IsMultivaluedHeader(absl::string_view header) {
  static const MultivaluedHeadersSet* const multivalued_headers =
      buildMultivaluedHeaders();
  return multivalued_headers->contains(header);
}
bool IsInvalidHeaderKeyChar(uint8_t c) {
  static const std::array<bool, 256> invalidHeaderKeyCharTable =
      buildInvalidHeaderKeyCharLookupTable();
  return invalidHeaderKeyCharTable[c];
}
bool IsInvalidHeaderKeyCharAllowDoubleQuote(uint8_t c) {
  static const std::array<bool, 256> invalidHeaderKeyCharTable =
      buildInvalidHeaderKeyCharLookupTableAllowDoubleQuote();
  return invalidHeaderKeyCharTable[c];
}
bool IsInvalidHeaderChar(uint8_t c) {
  static const std::array<bool, 256> invalidCharTable =
      buildInvalidCharLookupTable();
  return invalidCharTable[c];
}
bool HasInvalidHeaderChars(absl::string_view value) {
  for (const char c : value) {
    if (IsInvalidHeaderChar(c)) {
      return true;
    }
  }
  return false;
}
bool HasInvalidPathChar(absl::string_view value) {
  static const std::array<bool, 256> invalidCharTable =
      buildInvalidPathCharLookupTable();
  for (const char c : value) {
    if (invalidCharTable[c]) {
      return true;
    }
  }
  return false;
}
}  