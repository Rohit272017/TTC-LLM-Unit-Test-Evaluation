#include <libaddressinput/address_formatter.h>
#include <libaddressinput/address_data.h>
#include <libaddressinput/address_field.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>
#include "format_element.h"
#include "language.h"
#include "region_data_constants.h"
#include "rule.h"
#include "util/cctype_tolower_equal.h"
#include "util/size.h"
namespace i18n {
namespace addressinput {
namespace {
const char kCommaSeparator[] = ", ";
const char kSpaceSeparator[] = " ";
const char kArabicCommaSeparator[] = "، ";
const char kLanguagesThatUseSpace[][3] = {
    "th",
    "ko",
};
const char kLanguagesThatHaveNoSeparator[][3] = {
    "ja",
    "zh",  
};
const char kLanguagesThatUseAnArabicComma[][3] = {
    "ar",
    "fa",
    "ku",
    "ps",
    "ur",
};
std::string GetLineSeparatorForLanguage(const std::string& language_tag) {
  Language address_language(language_tag);
  if (address_language.has_latin_script) {
    return kCommaSeparator;
  }
  const std::string& base_language = address_language.base;
  using std::placeholders::_1;
  if (std::find_if(kLanguagesThatUseSpace,
                   kLanguagesThatUseSpace + size(kLanguagesThatUseSpace),
                   std::bind(&EqualToTolowerString, _1, base_language)) !=
      kLanguagesThatUseSpace + size(kLanguagesThatUseSpace)) {
    return kSpaceSeparator;
  } else if (std::find_if(
                 kLanguagesThatHaveNoSeparator,
                 kLanguagesThatHaveNoSeparator +
                     size(kLanguagesThatHaveNoSeparator),
                 std::bind(&EqualToTolowerString, _1, base_language)) !=
             kLanguagesThatHaveNoSeparator +
                 size(kLanguagesThatHaveNoSeparator)) {
    return "";
  } else if (std::find_if(
                 kLanguagesThatUseAnArabicComma,
                 kLanguagesThatUseAnArabicComma +
                     size(kLanguagesThatUseAnArabicComma),
                 std::bind(&EqualToTolowerString, _1, base_language)) !=
             kLanguagesThatUseAnArabicComma +
                 size(kLanguagesThatUseAnArabicComma)) {
    return kArabicCommaSeparator;
  }
  return kCommaSeparator;
}
void CombineLinesForLanguage(const std::vector<std::string>& lines,
                             const std::string& language_tag,
                             std::string* line) {
  line->clear();
  std::string separator = GetLineSeparatorForLanguage(language_tag);
  for (auto it = lines.begin(); it != lines.end(); ++it) {
    if (it != lines.begin()) {
      line->append(separator);
    }
    line->append(*it);
  }
}
}  
void GetFormattedNationalAddress(
    const AddressData& address_data, std::vector<std::string>* lines) {
  assert(lines != nullptr);
  lines->clear();
  Rule rule;
  rule.CopyFrom(Rule::GetDefault());
  rule.ParseSerializedRule(
      RegionDataConstants::GetRegionData(address_data.region_code));
  Language language(address_data.language_code);
  const std::vector<FormatElement>& format =
      language.has_latin_script && !rule.GetLatinFormat().empty()
          ? rule.GetLatinFormat()
          : rule.GetFormat();
  std::vector<FormatElement> pruned_format;
  for (auto element_it = format.begin();
       element_it != format.end();
       ++element_it) {
    if (element_it->IsNewline() ||
        (element_it->IsField() &&
         !address_data.IsFieldEmpty(element_it->GetField())) ||
        (!element_it->IsField() &&
         (element_it + 1 == format.end() ||
          !(element_it + 1)->IsField() ||
          !address_data.IsFieldEmpty((element_it + 1)->GetField())) &&
         (element_it == format.begin() ||
          !(element_it - 1)->IsField() ||
          (!pruned_format.empty() && pruned_format.back().IsField())))) {
      pruned_format.push_back(*element_it);
    }
  }
  std::string line;
  for (const auto& element : pruned_format) {
    if (element.IsNewline()) {
      if (!line.empty()) {
        lines->push_back(line);
        line.clear();
      }
    } else if (element.IsField()) {
      AddressField field = element.GetField();
      if (field == STREET_ADDRESS) {
        if (!address_data.IsFieldEmpty(field)) {
          line.append(address_data.address_line.front());
          if (address_data.address_line.size() > 1U) {
            lines->push_back(line);
            line.clear();
            const auto last_element_iterator =
                address_data.address_line.begin() +
                address_data.address_line.size() - 1;
            lines->insert(lines->end(), address_data.address_line.begin() + 1,
                          last_element_iterator);
            line.append(*last_element_iterator);
          }
        }
      } else {
        line.append(address_data.GetFieldValue(field));
      }
    } else {
      line.append(element.GetLiteral());
    }
  }
  if (!line.empty()) {
    lines->push_back(line);
  }
}
void GetFormattedNationalAddressLine(
    const AddressData& address_data, std::string* line) {
  std::vector<std::string> address_lines;
  GetFormattedNationalAddress(address_data, &address_lines);
  CombineLinesForLanguage(address_lines, address_data.language_code, line);
}
void GetStreetAddressLinesAsSingleLine(
    const AddressData& address_data, std::string* line) {
  CombineLinesForLanguage(
      address_data.address_line, address_data.language_code, line);
}
}  
}  