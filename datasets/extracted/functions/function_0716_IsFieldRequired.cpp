#include <libaddressinput/address_metadata.h>
#include <libaddressinput/address_field.h>
#include <algorithm>
#include <string>
#include "format_element.h"
#include "region_data_constants.h"
#include "rule.h"
namespace i18n {
namespace addressinput {
bool IsFieldRequired(AddressField field, const std::string& region_code) {
  if (field == COUNTRY) {
    return true;
  }
  Rule rule;
  rule.CopyFrom(Rule::GetDefault());
  if (!rule.ParseSerializedRule(
          RegionDataConstants::GetRegionData(region_code))) {
    return false;
  }
  return std::find(rule.GetRequired().begin(),
                   rule.GetRequired().end(),
                   field) != rule.GetRequired().end();
}
bool IsFieldUsed(AddressField field, const std::string& region_code) {
  if (field == COUNTRY) {
    return true;
  }
  Rule rule;
  rule.CopyFrom(Rule::GetDefault());
  if (!rule.ParseSerializedRule(
          RegionDataConstants::GetRegionData(region_code))) {
    return false;
  }
  return std::find(rule.GetFormat().begin(),
                   rule.GetFormat().end(),
                   FormatElement(field)) != rule.GetFormat().end();
}
}  
}  