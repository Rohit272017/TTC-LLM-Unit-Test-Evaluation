#ifndef XLA_DEBUG_OPTIONS_PARSERS_H_
#define XLA_DEBUG_OPTIONS_PARSERS_H_
#include <string>
#include <vector>
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "xla/xla.pb.h"
namespace xla {
template <typename T>
void parse_xla_backend_extra_options(T* extra_options_map,
                                     std::string comma_separated_values) {
  std::vector<std::string> extra_options_parts =
      absl::StrSplit(comma_separated_values, ',');
  for (const auto& part : extra_options_parts) {
    size_t eq_pos = part.find_first_of('=');
    if (eq_pos == std::string::npos) {
      (*extra_options_map)[part] = "";
    } else {
      std::string value = "";
      if (eq_pos + 1 < part.size()) {
        value = part.substr(eq_pos + 1);
      }
      (*extra_options_map)[part.substr(0, eq_pos)] = value;
    }
  }
}
}  
#endif  