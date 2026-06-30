#include "tensorflow/c/experimental/ops/gen/common/case_format.h"
#include "absl/strings/ascii.h"
#include "tensorflow/core/platform/types.h"
namespace tensorflow {
namespace generator {
namespace {
enum CaseFormatType {
  LOWER_CAMEL,
  UPPER_CAMEL,
  LOWER_SNAKE,
  UPPER_SNAKE,
};
string FormatStringCase(const string &str, CaseFormatType to,
                        const char delimiter = '_') {
  const bool from_snake = (str == absl::AsciiStrToUpper(str)) ||
                          (str == absl::AsciiStrToLower(str));
  const bool toUpper = (to == UPPER_CAMEL || to == UPPER_SNAKE);
  const bool toSnake = (to == LOWER_SNAKE || to == UPPER_SNAKE);
  string result;
  bool inputStart = true;
  bool wordStart = true;
  for (const char c : str) {
    if (c == delimiter) {
      if (wordStart) {
        result.push_back(delimiter);
      }
      wordStart = true;
      continue;
    }
    if (!from_snake && isupper(c)) {
      wordStart = true;
    }
    if (wordStart && toSnake && !inputStart) {
      result.push_back(delimiter);
    }
    const bool shouldCapIfSnake = toUpper;
    const bool shouldCapIfCamel = wordStart && (toUpper || !inputStart);
    if ((toSnake && shouldCapIfSnake) || (!toSnake && shouldCapIfCamel)) {
      result += toupper(c);
    } else {
      result += tolower(c);
    }
    wordStart = false;
    inputStart = false;
  }
  if (wordStart) {
    result.push_back(delimiter);
  }
  return result;
}
}  
string toLowerCamel(const string &s, const char delimiter) {
  return FormatStringCase(s, LOWER_CAMEL, delimiter);
}
string toLowerSnake(const string &s, const char delimiter) {
  return FormatStringCase(s, LOWER_SNAKE, delimiter);
}
string toUpperCamel(const string &s, const char delimiter) {
  return FormatStringCase(s, UPPER_CAMEL, delimiter);
}
string toUpperSnake(const string &s, const char delimiter) {
  return FormatStringCase(s, UPPER_SNAKE, delimiter);
}
}  
}  