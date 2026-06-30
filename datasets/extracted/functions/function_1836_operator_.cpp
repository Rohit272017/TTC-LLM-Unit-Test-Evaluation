#include <libaddressinput/address_field.h>
#include <cstddef>
#include <ostream>
#include "util/size.h"
using i18n::addressinput::AddressField;
using i18n::addressinput::COUNTRY;
using i18n::addressinput::RECIPIENT;
using i18n::addressinput::size;
std::ostream& operator<<(std::ostream& o, AddressField field) {
  static const char* const kFieldNames[] = {
      "COUNTRY",
      "ADMIN_AREA",
      "LOCALITY",
      "DEPENDENT_LOCALITY",
      "SORTING_CODE",
      "POSTAL_CODE",
      "STREET_ADDRESS",
      "ORGANIZATION",
      "RECIPIENT",
  };
  static_assert(COUNTRY == 0, "bad_base");
  static_assert(RECIPIENT == size(kFieldNames) - 1, "bad_length");
  if (field < 0 || static_cast<size_t>(field) >= size(kFieldNames)) {
    o << "[INVALID ENUM VALUE " << static_cast<int>(field) << "]";
  } else {
    o << kFieldNames[field];
  }
  return o;
}