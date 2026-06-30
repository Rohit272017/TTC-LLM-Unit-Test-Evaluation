#include "tensorflow/core/profiler/convert/trace_viewer/trace_events_to_json.h"
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/profiler/protobuf/trace_events.pb.h"
#include "tensorflow/core/profiler/protobuf/trace_events_raw.pb.h"
namespace tensorflow {
namespace profiler {
std::string JsonEscape(absl::string_view raw) {
  std::string escaped_string;
  const size_t length = raw.length();
  escaped_string.reserve((length + 1) * 2);
  escaped_string.push_back('"');
  for (size_t i = 0; i < length; ++i) {
    const unsigned char c = raw[i];
    if (c < 0x20) {
      escaped_string.push_back('\\');
      switch (c) {
        case '\b':
          escaped_string.push_back('b');
          break;
        case '\f':
          escaped_string.push_back('f');
          break;
        case '\n':
          escaped_string.push_back('n');
          break;
        case '\r':
          escaped_string.push_back('r');
          break;
        case '\t':
          escaped_string.push_back('t');
          break;
        default:
          absl::StrAppendFormat(&escaped_string, "u%04x",
                                static_cast<unsigned int>(c));
      }
      continue;
    }
    switch (c) {
      case '\"':
        escaped_string.append("\\\"");
        continue;
      case '\\':
        escaped_string.append("\\\\");
        continue;
      case '<':
      case '>':
      case '&': {
        absl::StrAppendFormat(&escaped_string, "\\u%04x",
                              static_cast<unsigned int>(c));
        continue;
      }
      case '\xe2': {
        if ((i + 2 < length) && (raw[i + 1] == '\x80')) {
          if (raw[i + 2] == '\xa8') {
            escaped_string.append("\\u2028");
            i += 2;
            continue;
          } else if (raw[i + 2] == '\xa9') {
            escaped_string.append("\\u2029");
            i += 2;
            continue;
          }
        }
        escaped_string.push_back(c);
        continue;
      }
    }
    escaped_string.push_back(c);
  }
  escaped_string.push_back('"');
  return escaped_string;
}
std::string ProtoString(const tsl::protobuf::Message& pb) {
  return JsonEscape(pb.DebugString());
}
std::map<uint64_t, uint64_t> BuildStackFrameReferences(const Trace& trace) {
  const auto& name_table = trace.name_table();
  std::map<uint64_t, uint64_t> output;
  for (const auto& [fp, name] : name_table) {
    if (!absl::StartsWith(name, "@@")) continue;
    output[fp] = 0;
  }
  uint64_t sf = 1;
  for (auto& it : output) {
    it.second = sf++;
  }
  return output;
}
}  
}  