#include "string_split.h"
#include <cassert>
#include <cstddef>
#include <string>
#include <vector>
namespace i18n {
namespace addressinput {
void SplitString(const std::string& str, char s, std::vector<std::string>* r) {
  assert(r != nullptr);
  r->clear();
  size_t last = 0;
  size_t c = str.size();
  for (size_t i = 0; i <= c; ++i) {
    if (i == c || str[i] == s) {
      std::string tmp(str, last, i - last);
      if (i != c || !r->empty() || !tmp.empty()) {
        r->push_back(tmp);
      }
      last = i + 1;
    }
  }
}
}  
}  