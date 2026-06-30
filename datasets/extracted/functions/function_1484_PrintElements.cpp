#ifndef QUICHE_COMMON_PRINT_ELEMENTS_H_
#define QUICHE_COMMON_PRINT_ELEMENTS_H_
#include <ostream>
#include <sstream>
#include <string>
#include "quiche/common/platform/api/quiche_export.h"
namespace quiche {
template <typename T>
QUICHE_NO_EXPORT inline std::string PrintElements(const T& container) {
  std::stringstream debug_string;
  debug_string << "{";
  auto it = container.cbegin();
  if (it != container.cend()) {
    debug_string << *it;
    ++it;
    while (it != container.cend()) {
      debug_string << ", " << *it;
      ++it;
    }
  }
  debug_string << "}";
  return debug_string.str();
}
}  
#endif  