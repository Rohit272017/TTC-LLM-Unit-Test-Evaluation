#include "post_box_matchers.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include <re2/re2.h>
#include "language.h"
#include "rule.h"
#include "util/re2ptr.h"
#include "util/size.h"
namespace i18n {
namespace addressinput {
namespace {
struct LanguageInfo {
  const char* language;
  const char* regexp;
  static bool less(const LanguageInfo& a, const LanguageInfo& b) {
      return strcmp(a.language, b.language) < 0;
  }
};
constexpr const LanguageInfo kLanguageInfoMap[] = {
      {"ar", R"(صندوق بريد|ص[-. ]ب)"},
      {"cs", R"((?i)p\.? ?p\.? \d)"},
      {"da", R"((?i)Postboks)"},
      {"de", R"((?i)Postfach)"},
      {"el", R"((?i)T\.? ?Θ\.? \d{2})"},
      {"en", R"(Private Bag|Post(?:al)? Box)"},
      {"es", R"((?i)(?:Apartado|Casillas) de correos?)"},
      {"fi", R"((?i)Postilokero|P\.?L\.? \d)"},
      {"fr", R"((?i)Bo(?:[iî]|î)te Postale|BP \d|CEDEX \d)"},
      {"hr", R"((?i)p\.? ?p\.? \d)"},
      {"hu", R"((?i)Postafi(?:[oó]|ó)k|Pf\.? \d)"},
      {"ja", R"(私書箱\d{1,5}号)"},
      {"nl", R"((?i)Postbus)"},
      {"no", R"((?i)Postboks)"},
      {"pl", R"((?i)Skr(?:\.?|ytka) poczt(?:\.?|owa))"},
      {"pt", R"((?i)Apartado)"},
      {"ru", R"((?i)абонентский ящик|[аa]"я (?:(?:№|#|N) ?)?\d)"},
      {"sv", R"((?i)Box \d)"},
      {"und", R"(P\.? ?O\.? Box)"},
      {"zh", R"(郵政信箱.{1,5}號|郵局第.{1,10}號信箱)"},
};
constexpr size_t kLanguageInfoMapSize = size(kLanguageInfoMap);
constexpr bool StrLessOrEqualConstexpr(const char* a, const char* b) {
  return (*a == '\0') ? true : (
      (*a == *b) ? StrLessOrEqualConstexpr(a + 1, b + 1) : (*a < *b));
}
static_assert(StrLessOrEqualConstexpr("", ""), "");
static_assert(StrLessOrEqualConstexpr("", "foo"), "");
static_assert(!StrLessOrEqualConstexpr("foo", ""), "");
static_assert(StrLessOrEqualConstexpr("foo", "foo"), "");
static_assert(!StrLessOrEqualConstexpr("foo", "bar"), "");
static_assert(StrLessOrEqualConstexpr("bar", "foo"), "");
static_assert(StrLessOrEqualConstexpr("foo", "foobar"), "");
static_assert(!StrLessOrEqualConstexpr("foobar", "foo"), "");
constexpr bool CheckLanguageInfoMapOrderConstexpr(size_t n = 0) {
  return !StrLessOrEqualConstexpr(kLanguageInfoMap[n].language,
                                  kLanguageInfoMap[n + 1].language) ? false : (
      (n + 2 < kLanguageInfoMapSize) ?
         CheckLanguageInfoMapOrderConstexpr(n + 1) : true);
}
static_assert(CheckLanguageInfoMapOrderConstexpr(),
              "kLanguageInfoMap is not correctly sorted!");
const LanguageInfo* FindLanguageInfoFor(const std::string& language) {
  const LanguageInfo* begin = kLanguageInfoMap;
  const LanguageInfo* end = begin + kLanguageInfoMapSize;
  LanguageInfo key = { language.c_str(), };
  const LanguageInfo* probe =
      std::lower_bound(begin, end, key, LanguageInfo::less);
  if (probe != end && language == probe->language) {
    return probe;
  }
  return nullptr;
}
class StaticRE2Array {
 public:
  StaticRE2Array() {
    for (size_t n = 0; n < kLanguageInfoMapSize; ++n) {
      re2s_[n].ptr = new RE2(kLanguageInfoMap[n].regexp);
    }
  }
  ~StaticRE2Array() {
    for (auto& entry : re2s_) {
      delete entry.ptr;
    }
  }
  const RE2PlainPtr* FindMatcherFor(const std::string& language) const {
    const LanguageInfo* info = FindLanguageInfoFor(language);
    if (!info) {
      return nullptr;
    }
    size_t idx = info - kLanguageInfoMap;
    assert(idx < kLanguageInfoMapSize);
    return &re2s_[idx];
  }
 private:
  RE2PlainPtr re2s_[kLanguageInfoMapSize];
};
}  
std::vector<const RE2PlainPtr*> PostBoxMatchers::GetMatchers(
    const Rule& country_rule) {
  static const StaticRE2Array kMatchers;
  std::vector<std::string> languages{"und"};
  for (const auto& language_tag : country_rule.GetLanguages()) {
    Language language(language_tag);
    languages.push_back(language.base);
  }
  std::vector<const RE2PlainPtr*> result;
  for (const auto& language_tag : languages) {
    const RE2PlainPtr* matcher = kMatchers.FindMatcherFor(language_tag);
    if (matcher != nullptr) {
      result.push_back(matcher);
    }
  }
  return result;
}
}  
}  