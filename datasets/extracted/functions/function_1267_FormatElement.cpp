#include "format_element.h"
#include <libaddressinput/address_field.h>
#include <cassert>
#include <ostream>
#include <string>
namespace i18n {
namespace addressinput {
FormatElement::FormatElement(AddressField field) : field_(field), literal_() {}
FormatElement::FormatElement(const std::string& literal)
    : field_(COUNTRY), literal_(literal) {
  assert(!literal.empty());
}
FormatElement::FormatElement() : field_(COUNTRY), literal_("\n") {}
bool FormatElement::operator==(const FormatElement& other) const {
  return field_ == other.field_ && literal_ == other.literal_;
}
}  
}  
std::ostream& operator<<(std::ostream& o,
                         const i18n::addressinput::FormatElement& element) {
  if (element.IsField()) {
    o << "Field: " << element.GetField();
  } else if (element.IsNewline()) {
    o << "Newline";
  } else {
    o << "Literal: " << element.GetLiteral();
  }
  return o;
}