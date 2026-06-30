#include "lookup_key.h"
#include <libaddressinput/address_data.h>
#include <libaddressinput/address_field.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <string>
#include "language.h"
#include "region_data_constants.h"
#include "rule.h"
#include "util/cctype_tolower_equal.h"
#include "util/size.h"
namespace i18n {
namespace addressinput {
namespace {
const char kSlashDelim[] = "/";
const char kDashDelim[] = "--";
const char kData[] = "data";
const char kUnknown[] = "ZZ";
bool ShouldSetLanguageForKey(const std::string& language_tag,
                             const std::string& region_code) {
  if (RegionDataConstants::GetMaxLookupKeyDepth(region_code) == 0) {
    return false;
  }
  Rule rule;
  rule.CopyFrom(Rule::GetDefault());
  if (!rule.ParseSerializedRule(
          RegionDataConstants::GetRegionData(region_code))) {
    return false;
  }
  const auto& languages = rule.GetLanguages();
  if (languages.empty() || languages[0] == language_tag) {
    return false;
  }
  using std::placeholders::_1;
  return std::find_if(languages.begin() + 1, languages.end(),
                      std::bind(&EqualToTolowerString, _1, language_tag)) !=
         languages.end();
}
}  
const AddressField LookupKey::kHierarchy[] = {
    COUNTRY,
    ADMIN_AREA,
    LOCALITY,
    DEPENDENT_LOCALITY,
};
LookupKey::LookupKey() = default;
LookupKey::~LookupKey() = default;
void LookupKey::FromAddress(const AddressData& address) {
  nodes_.clear();
  if (address.region_code.empty()) {
    nodes_.emplace(COUNTRY, kUnknown);
  } else {
    for (AddressField field : kHierarchy) {
      if (address.IsFieldEmpty(field)) {
        break;
      }
      const std::string& value = address.GetFieldValue(field);
      if (value.find('/') != std::string::npos) {
        break;
      }
      nodes_.emplace(field, value);
    }
  }
  Language address_language(address.language_code);
  std::string language_tag_no_latn = address_language.has_latin_script
                                         ? address_language.base
                                         : address_language.tag;
  if (ShouldSetLanguageForKey(language_tag_no_latn, address.region_code)) {
    language_ = language_tag_no_latn;
  }
}
void LookupKey::FromLookupKey(const LookupKey& parent,
                              const std::string& child_node) {
  assert(parent.nodes_.size() < size(kHierarchy));
  assert(!child_node.empty());
  if (this != &parent) nodes_ = parent.nodes_;
  AddressField child_field = kHierarchy[nodes_.size()];
  nodes_.emplace(child_field, child_node);
}
std::string LookupKey::ToKeyString(size_t max_depth) const {
  assert(max_depth < size(kHierarchy));
  std::string key_string(kData);
  for (size_t i = 0; i <= max_depth; ++i) {
    AddressField field = kHierarchy[i];
    auto it = nodes_.find(field);
    if (it == nodes_.end()) {
      break;
    }
    key_string.append(kSlashDelim);
    key_string.append(it->second);
  }
  if (!language_.empty()) {
    key_string.append(kDashDelim);
    key_string.append(language_);
  }
  return key_string;
}
const std::string& LookupKey::GetRegionCode() const {
  auto it = nodes_.find(COUNTRY);
  assert(it != nodes_.end());
  return it->second;
}
size_t LookupKey::GetDepth() const {
  size_t depth = nodes_.size() - 1;
  assert(depth < size(kHierarchy));
  return depth;
}
}  
}  