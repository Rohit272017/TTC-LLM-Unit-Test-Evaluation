#include <libaddressinput/address_data.h>
#include <libaddressinput/address_field.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <ostream>
#include <string>
#include <vector>
#include <re2/re2.h>
#include "util/size.h"
namespace i18n {
namespace addressinput {
namespace {
std::string AddressData::*kStringField[] = {
    &AddressData::region_code,
    &AddressData::administrative_area,
    &AddressData::locality,
    &AddressData::dependent_locality,
    &AddressData::sorting_code,
    &AddressData::postal_code,
    nullptr,
    &AddressData::organization,
    &AddressData::recipient,
};
const std::vector<std::string> AddressData::*kVectorStringField[] = {
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    &AddressData::address_line,
    nullptr,
    nullptr,
};
static_assert(size(kStringField) == size(kVectorStringField),
              "field_mapping_array_size_mismatch");
bool IsStringEmpty(const std::string& str) {
  static const RE2 kMatcher(R"(\S)");
  return str.empty() || !RE2::PartialMatch(str, kMatcher);
}
}  
bool AddressData::IsFieldEmpty(AddressField field) const {
  assert(field >= 0);
  assert(static_cast<size_t>(field) < size(kStringField));
  if (kStringField[field] != nullptr) {
    const auto& value = GetFieldValue(field);
    return IsStringEmpty(value);
  } else {
    const auto& value = GetRepeatedFieldValue(field);
    return std::find_if_not(value.begin(), value.end(), IsStringEmpty) ==
           value.end();
  }
}
const std::string& AddressData::GetFieldValue(AddressField field) const {
  assert(field >= 0);
  assert(static_cast<size_t>(field) < size(kStringField));
  assert(kStringField[field] != nullptr);
  return this->*kStringField[field];
}
void AddressData::SetFieldValue(AddressField field, const std::string& value) {
  assert(field >= 0);
  assert(static_cast<size_t>(field) < size(kStringField));
  assert(kStringField[field] != nullptr);
  (this->*kStringField[field]).assign(value);
}
const std::vector<std::string>& AddressData::GetRepeatedFieldValue(
    AddressField field) const {
  assert(IsRepeatedFieldValue(field));
  return this->*kVectorStringField[field];
}
bool AddressData::operator==(const AddressData& other) const {
  return region_code == other.region_code &&
         address_line == other.address_line &&
         administrative_area == other.administrative_area &&
         locality == other.locality &&
         dependent_locality == other.dependent_locality &&
         postal_code == other.postal_code &&
         sorting_code == other.sorting_code &&
         language_code == other.language_code &&
         organization == other.organization &&
         recipient == other.recipient;
}
bool AddressData::IsRepeatedFieldValue(AddressField field) {
  assert(field >= 0);
  assert(static_cast<size_t>(field) < size(kVectorStringField));
  return kVectorStringField[field] != nullptr;
}
}  
}  
std::ostream& operator<<(std::ostream& o,
                         const i18n::addressinput::AddressData& address) {
  o << "region_code: \"" << address.region_code << "\"\n"
    "administrative_area: \"" << address.administrative_area << "\"\n"
    "locality: \"" << address.locality << "\"\n"
    "dependent_locality: \"" << address.dependent_locality << "\"\n"
    "postal_code: \"" << address.postal_code << "\"\n"
    "sorting_code: \"" << address.sorting_code << "\"\n";
  for (const auto& line : address.address_line) {
    o << "address_line: \"" << line << "\"\n";
  }
  o << "language_code: \"" << address.language_code << "\"\n"
    "organization: \"" << address.organization << "\"\n"
    "recipient: \"" << address.recipient << "\"\n";
  return o;
}