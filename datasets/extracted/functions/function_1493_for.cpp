#ifndef QUICHE_QUIC_CORE_HTTP_HTTP_FRAMES_H_
#define QUICHE_QUIC_CORE_HTTP_HTTP_FRAMES_H_
#include <algorithm>
#include <cstdint>
#include <map>
#include <ostream>
#include <sstream>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "quiche/http2/core/spdy_protocol.h"
#include "quiche/quic/core/http/http_constants.h"
#include "quiche/quic/core/quic_types.h"
namespace quic {
enum class HttpFrameType {
  DATA = 0x0,
  HEADERS = 0x1,
  CANCEL_PUSH = 0x3,
  SETTINGS = 0x4,
  PUSH_PROMISE = 0x5,
  GOAWAY = 0x7,
  ORIGIN = 0xC,
  MAX_PUSH_ID = 0xD,
  ACCEPT_CH = 0x89,
  PRIORITY_UPDATE_REQUEST_STREAM = 0xF0700,
  WEBTRANSPORT_STREAM = 0x41,
  METADATA = 0x4d,
};
struct QUICHE_EXPORT DataFrame {
  absl::string_view data;
};
struct QUICHE_EXPORT HeadersFrame {
  absl::string_view headers;
};
using SettingsMap = absl::flat_hash_map<uint64_t, uint64_t>;
struct QUICHE_EXPORT SettingsFrame {
  SettingsMap values;
  bool operator==(const SettingsFrame& rhs) const {
    return values == rhs.values;
  }
  std::string ToString() const {
    std::string s;
    for (auto it : values) {
      std::string setting = absl::StrCat(
          H3SettingsToString(
              static_cast<Http3AndQpackSettingsIdentifiers>(it.first)),
          " = ", it.second, "; ");
      absl::StrAppend(&s, setting);
    }
    return s;
  }
  friend QUICHE_EXPORT std::ostream& operator<<(std::ostream& os,
                                                const SettingsFrame& s) {
    os << s.ToString();
    return os;
  }
};
struct QUICHE_EXPORT GoAwayFrame {
  uint64_t id;
  bool operator==(const GoAwayFrame& rhs) const { return id == rhs.id; }
};
struct QUICHE_EXPORT OriginFrame {
  std::vector<std::string> origins;
  bool operator==(const OriginFrame& rhs) const {
    return origins == rhs.origins;
  }
  std::string ToString() const {
    std::string result = "Origin Frame: {origins: ";
    for (const std::string& origin : origins) {
      absl::StrAppend(&result, "\n", origin);
    }
    result += "}";
    return result;
  }
  friend QUICHE_EXPORT std::ostream& operator<<(std::ostream& os,
                                                const OriginFrame& s) {
    os << s.ToString();
    return os;
  }
};
inline constexpr QuicByteCount kPriorityFirstByteLength = 1;
struct QUICHE_EXPORT PriorityUpdateFrame {
  uint64_t prioritized_element_id = 0;
  std::string priority_field_value;
  bool operator==(const PriorityUpdateFrame& rhs) const {
    return std::tie(prioritized_element_id, priority_field_value) ==
           std::tie(rhs.prioritized_element_id, rhs.priority_field_value);
  }
  std::string ToString() const {
    return absl::StrCat(
        "Priority Frame : {prioritized_element_id: ", prioritized_element_id,
        ", priority_field_value: ", priority_field_value, "}");
  }
  friend QUICHE_EXPORT std::ostream& operator<<(std::ostream& os,
                                                const PriorityUpdateFrame& s) {
    os << s.ToString();
    return os;
  }
};
struct QUICHE_EXPORT AcceptChFrame {
  std::vector<spdy::AcceptChOriginValuePair> entries;
  bool operator==(const AcceptChFrame& rhs) const {
    return entries.size() == rhs.entries.size() &&
           std::equal(entries.begin(), entries.end(), rhs.entries.begin());
  }
  std::string ToString() const {
    std::stringstream s;
    s << *this;
    return s.str();
  }
  friend QUICHE_EXPORT std::ostream& operator<<(std::ostream& os,
                                                const AcceptChFrame& frame) {
    os << "ACCEPT_CH frame with " << frame.entries.size() << " entries: ";
    for (auto& entry : frame.entries) {
      os << "origin: " << entry.origin << "; value: " << entry.value;
    }
    return os;
  }
};
}  
#endif  