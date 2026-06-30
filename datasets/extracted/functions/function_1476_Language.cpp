#include "language.h"
#include <algorithm>
#include <cctype>
#include <string>
#include <vector>
#include "rule.h"
#include "util/string_split.h"
namespace i18n {
namespace addressinput {
Language::Language(const std::string& language_tag) : tag(language_tag),
                                                      base(),
                                                      has_latin_script(false) {
  static const char kSubtagsSeparator = '-';
  static const char kAlternativeSubtagsSeparator = '_';
  std::replace(
      tag.begin(), tag.end(), kAlternativeSubtagsSeparator, kSubtagsSeparator);
  std::string lowercase = tag;
  std::transform(
      lowercase.begin(), lowercase.end(), lowercase.begin(), tolower);
  base = lowercase.substr(0, lowercase.find(kSubtagsSeparator));
  static const char kLowercaseLatinScript[] = "latn";
  std::vector<std::string> subtags;
  SplitString(lowercase, kSubtagsSeparator, &subtags);
  has_latin_script =
      (subtags.size() > 1 && subtags[1] == kLowercaseLatinScript) ||
      (subtags.size() > 2 && subtags[2] == kLowercaseLatinScript);
}
Language::~Language() = default;
Language ChooseBestAddressLanguage(const Rule& address_region_rule,
                                   const Language& ui_language) {
  if (address_region_rule.GetLanguages().empty()) {
    return ui_language;
  }
  std::vector<Language> available_languages;
  for (const auto& language_tag : address_region_rule.GetLanguages()) {
    available_languages.emplace_back(language_tag);
  }
  if (ui_language.tag.empty()) {
    return available_languages.front();
  }
  bool has_latin_format = !address_region_rule.GetLatinFormat().empty();
  static const char kLatinScriptSuffix[] = "-Latn";
  Language latin_script_language(
      available_languages.front().base + kLatinScriptSuffix);
  if (has_latin_format && ui_language.has_latin_script) {
    return latin_script_language;
  }
  for (const auto& language : available_languages) {
    if (ui_language.base == language.base) {
      return language;
    }
  }
  return has_latin_format ? latin_script_language : available_languages.front();
}
}  
}  