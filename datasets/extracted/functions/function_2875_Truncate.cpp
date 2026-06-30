#include "arolla/util/string.h"
#include <cstddef>
#include <string>
#include "absl/log/check.h"
namespace arolla {
std::string Truncate(std::string str, size_t max_length) {
  DCHECK_GT(max_length, 3);
  if (str.size() > max_length) {
    str.resize(max_length);
    str.replace(max_length - 3, 3, "...");
  }
  return str;
}
}  