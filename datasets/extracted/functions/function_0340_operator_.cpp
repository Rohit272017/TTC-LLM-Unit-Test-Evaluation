#include "quiche/http2/http2_structures.h"
#include <cstring>  
#include <ostream>
#include <sstream>
#include <string>
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
namespace http2 {
bool Http2FrameHeader::IsProbableHttpResponse() const {
  return (payload_length == 0x485454 &&      
          static_cast<char>(type) == 'P' &&  
          flags == '/');                     
}
std::string Http2FrameHeader::ToString() const {
  return absl::StrCat("length=", payload_length,
                      ", type=", Http2FrameTypeToString(type),
                      ", flags=", FlagsToString(), ", stream=", stream_id);
}
std::string Http2FrameHeader::FlagsToString() const {
  return Http2FrameFlagsToString(type, flags);
}
bool operator==(const Http2FrameHeader& a, const Http2FrameHeader& b) {
  return a.payload_length == b.payload_length && a.stream_id == b.stream_id &&
         a.type == b.type && a.flags == b.flags;
}
std::ostream& operator<<(std::ostream& out, const Http2FrameHeader& v) {
  return out << v.ToString();
}
bool operator==(const Http2PriorityFields& a, const Http2PriorityFields& b) {
  return a.stream_dependency == b.stream_dependency && a.weight == b.weight;
}
std::string Http2PriorityFields::ToString() const {
  std::stringstream ss;
  ss << "E=" << (is_exclusive ? "true" : "false")
     << ", stream=" << stream_dependency
     << ", weight=" << static_cast<uint32_t>(weight);
  return ss.str();
}
std::ostream& operator<<(std::ostream& out, const Http2PriorityFields& v) {
  return out << v.ToString();
}
bool operator==(const Http2RstStreamFields& a, const Http2RstStreamFields& b) {
  return a.error_code == b.error_code;
}
std::ostream& operator<<(std::ostream& out, const Http2RstStreamFields& v) {
  return out << "error_code=" << v.error_code;
}
bool operator==(const Http2SettingFields& a, const Http2SettingFields& b) {
  return a.parameter == b.parameter && a.value == b.value;
}
std::ostream& operator<<(std::ostream& out, const Http2SettingFields& v) {
  return out << "parameter=" << v.parameter << ", value=" << v.value;
}
bool operator==(const Http2PushPromiseFields& a,
                const Http2PushPromiseFields& b) {
  return a.promised_stream_id == b.promised_stream_id;
}
std::ostream& operator<<(std::ostream& out, const Http2PushPromiseFields& v) {
  return out << "promised_stream_id=" << v.promised_stream_id;
}
bool operator==(const Http2PingFields& a, const Http2PingFields& b) {
  static_assert((sizeof a.opaque_bytes) == Http2PingFields::EncodedSize(),
                "Why not the same size?");
  return 0 ==
         std::memcmp(a.opaque_bytes, b.opaque_bytes, sizeof a.opaque_bytes);
}
std::ostream& operator<<(std::ostream& out, const Http2PingFields& v) {
  return out << "opaque_bytes=0x"
             << absl::BytesToHexString(absl::string_view(
                    reinterpret_cast<const char*>(v.opaque_bytes),
                    sizeof v.opaque_bytes));
}
bool operator==(const Http2GoAwayFields& a, const Http2GoAwayFields& b) {
  return a.last_stream_id == b.last_stream_id && a.error_code == b.error_code;
}
std::ostream& operator<<(std::ostream& out, const Http2GoAwayFields& v) {
  return out << "last_stream_id=" << v.last_stream_id
             << ", error_code=" << v.error_code;
}
bool operator==(const Http2WindowUpdateFields& a,
                const Http2WindowUpdateFields& b) {
  return a.window_size_increment == b.window_size_increment;
}
std::ostream& operator<<(std::ostream& out, const Http2WindowUpdateFields& v) {
  return out << "window_size_increment=" << v.window_size_increment;
}
bool operator==(const Http2AltSvcFields& a, const Http2AltSvcFields& b) {
  return a.origin_length == b.origin_length;
}
std::ostream& operator<<(std::ostream& out, const Http2AltSvcFields& v) {
  return out << "origin_length=" << v.origin_length;
}
bool operator==(const Http2PriorityUpdateFields& a,
                const Http2PriorityUpdateFields& b) {
  return a.prioritized_stream_id == b.prioritized_stream_id;
}
std::string Http2PriorityUpdateFields::ToString() const {
  std::stringstream ss;
  ss << "prioritized_stream_id=" << prioritized_stream_id;
  return ss.str();
}
std::ostream& operator<<(std::ostream& out,
                         const Http2PriorityUpdateFields& v) {
  return out << v.ToString();
}
}  