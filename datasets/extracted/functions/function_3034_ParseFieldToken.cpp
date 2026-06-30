#include "address_field_util.h"
#include <libaddressinput/address_field.h>
#include <algorithm>
#include <cassert>
#include <string>
#include <vector>
#include "format_element.h"
namespace i18n {
namespace addressinput {
namespace {
bool ParseFieldToken(char c, AddressField* field) {
  assert(field != nullptr);
  static const struct { char c; AddressField field; } kTokenMap[] = {
      { 'R', COUNTRY },
      { 'S', ADMIN_AREA },
      { 'C', LOCALITY },
      { 'D', DEPENDENT_LOCALITY },
      { 'X', SORTING_CODE },
      { 'Z', POSTAL_CODE },
      { 'A', STREET_ADDRESS },
      { 'O', ORGANIZATION },
      { 'N', RECIPIENT },
  };
  for (const auto& entry : kTokenMap) {
    if (c == entry.c) {
      *field = entry.field;
      return true;
    }
  }
  return false;
}
}  
void ParseFormatRule(const std::string& format,
                     std::vector<FormatElement>* elements) {
  assert(elements != nullptr);
  elements->clear();
  std::string::const_iterator prev = format.begin();
  for (std::string::const_iterator next = format.begin();
       next != format.end(); prev = ++next) {
    if ((next = std::find(next, format.end(), '%')) == format.end()) {
      break;
    }
    if (prev < next) {
      elements->emplace_back(std::string(prev, next));
    }
    if ((prev = ++next) == format.end()) {
      break;
    }
    AddressField field;
    if (*next == 'n') {
      elements->emplace_back();
    } else if (ParseFieldToken(*next, &field)) {
      elements->emplace_back(field);
    }  
  }
  if (prev != format.end()) {
    elements->emplace_back(std::string(prev, format.end()));
  }
}
void ParseAddressFieldsRequired(const std::string& required,
                                std::vector<AddressField>* fields) {
  assert(fields != nullptr);
  fields->clear();
  for (char c : required) {
    AddressField field;
    if (ParseFieldToken(c, &field)) {
      fields->push_back(field);
    }
  }
}
}  
}  