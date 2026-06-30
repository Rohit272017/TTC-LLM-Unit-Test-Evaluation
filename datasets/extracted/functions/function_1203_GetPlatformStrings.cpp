#include "tensorflow/core/platform/platform_strings.h"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
namespace tensorflow {
int GetPlatformStrings(const std::string& path,
                       std::vector<std::string>* found) {
  int result;
  FILE* ifp = fopen(path.c_str(), "rb");
  if (ifp != nullptr) {
    static const char prefix[] = TF_PLAT_STR_MAGIC_PREFIX_;
    int first_char = prefix[1];
    int last_char = -1;
    int c;
    while ((c = getc(ifp)) != EOF) {
      if (c == first_char && last_char == 0) {
        int i = 2;
        while (prefix[i] != 0 && (c = getc(ifp)) == prefix[i]) {
          i++;
        }
        if (prefix[i] == 0) {
          std::string str;
          while ((c = getc(ifp)) != EOF && c != 0) {
            str.push_back(c);
          }
          if (!str.empty()) {
            found->push_back(str);
          }
        }
      }
      last_char = c;
    }
    result = (ferror(ifp) == 0) ? 0 : errno;
    if (fclose(ifp) != 0) {
      result = errno;
    }
  } else {
    result = errno;
  }
  return result;
}
}  