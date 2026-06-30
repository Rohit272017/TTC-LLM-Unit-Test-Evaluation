#include "tsl/platform/str_util.h"
#include <cctype>
#include <cstdint>
#include <string>
#include "absl/strings/ascii.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/stringpiece.h"
namespace tsl {
namespace str_util {
size_t RemoveLeadingWhitespace(absl::string_view* text) {
  absl::string_view new_text = absl::StripLeadingAsciiWhitespace(*text);
  size_t count = text->size() - new_text.size();
  *text = new_text;
  return count;
}
size_t RemoveTrailingWhitespace(absl::string_view* text) {
  absl::string_view new_text = absl::StripTrailingAsciiWhitespace(*text);
  size_t count = text->size() - new_text.size();
  *text = new_text;
  return count;
}
size_t RemoveWhitespaceContext(absl::string_view* text) {
  absl::string_view new_text = absl::StripAsciiWhitespace(*text);
  size_t count = text->size() - new_text.size();
  *text = new_text;
  return count;
}
bool ConsumeLeadingDigits(absl::string_view* s, uint64_t* val) {
  const char* p = s->data();
  const char* limit = p + s->size();
  uint64_t v = 0;
  while (p < limit) {
    const char c = *p;
    if (c < '0' || c > '9') break;
    uint64_t new_v = (v * 10) + (c - '0');
    if (new_v / 8 < v) {
      return false;
    }
    v = new_v;
    p++;
  }
  if (p > s->data()) {
    s->remove_prefix(p - s->data());
    *val = v;
    return true;
  } else {
    return false;
  }
}
bool ConsumeNonWhitespace(absl::string_view* s, absl::string_view* val) {
  const char* p = s->data();
  const char* limit = p + s->size();
  while (p < limit) {
    const char c = *p;
    if (isspace(c)) break;
    p++;
  }
  const size_t n = p - s->data();
  if (n > 0) {
    *val = absl::string_view(s->data(), n);
    s->remove_prefix(n);
    return true;
  } else {
    *val = absl::string_view();
    return false;
  }
}
void TitlecaseString(string* s, absl::string_view delimiters) {
  bool upper = true;
  for (string::iterator ss = s->begin(); ss != s->end(); ++ss) {
    if (upper) {
      *ss = toupper(*ss);
    }
    upper = (delimiters.find(*ss) != absl::string_view::npos);
  }
}
string StringReplace(absl::string_view s, absl::string_view oldsub,
                     absl::string_view newsub, bool replace_all) {
  string res(s);
  size_t pos = 0;
  while ((pos = res.find(oldsub.data(), pos, oldsub.size())) != string::npos) {
    res.replace(pos, oldsub.size(), newsub.data(), newsub.size());
    pos += newsub.size();
    if (oldsub.empty()) {
      pos++;  
    }
    if (!replace_all) {
      break;
    }
  }
  return res;
}
size_t Strnlen(const char* str, const size_t string_max_len) {
  size_t len = 0;
  while (len < string_max_len && str[len] != '\0') {
    ++len;
  }
  return len;
}
string ArgDefCase(absl::string_view s) {
  const size_t n = s.size();
  size_t extra_us = 0;
  size_t to_skip = 0;
  for (size_t i = 0; i < n; ++i) {
    if (i == to_skip && !isalpha(s[i])) {
      ++to_skip;
      continue;
    }
    if (isupper(s[i]) && i != to_skip && i > 0 && isalnum(s[i - 1])) {
      ++extra_us;
    }
  }
  string result(n + extra_us - to_skip, '_');
  for (size_t i = to_skip, j = 0; i < n; ++i, ++j) {
    DCHECK_LT(j, result.size());
    char c = s[i];
    if (isalnum(c)) {
      if (isupper(c)) {
        if (i != to_skip) {
          DCHECK_GT(j, 0);
          if (result[j - 1] != '_') ++j;
        }
        result[j] = tolower(c);
      } else {
        result[j] = c;
      }
    }
  }
  return result;
}
}  
}  