#include <libaddressinput/address_normalizer.h>
#include <libaddressinput/address_data.h>
#include <libaddressinput/address_field.h>
#include <libaddressinput/preload_supplier.h>
#include <cassert>
#include <cstddef>
#include <string>
#include <vector>
#include "lookup_key.h"
#include "rule.h"
#include "util/size.h"
#include "util/string_compare.h"
namespace i18n {
namespace addressinput {
AddressNormalizer::AddressNormalizer(const PreloadSupplier* supplier)
    : supplier_(supplier),
      compare_(new StringCompare) {
  assert(supplier_ != nullptr);
}
AddressNormalizer::~AddressNormalizer() = default;
void AddressNormalizer::Normalize(AddressData* address) const {
  assert(address != nullptr);
  assert(supplier_->IsLoaded(address->region_code));
  AddressData region_address;
  region_address.region_code = address->region_code;
  LookupKey parent_key;
  parent_key.FromAddress(region_address);
  const Rule* parent_rule = supplier_->GetRule(parent_key);
  assert(parent_rule != nullptr);
  std::vector<std::string> languages(parent_rule->GetLanguages());
  if (languages.empty()) {
    languages.emplace_back("");
  } else {
    languages[0] = "";  
  }
  LookupKey lookup_key;
  for (size_t depth = 1; depth < size(LookupKey::kHierarchy); ++depth) {
    AddressField field = LookupKey::kHierarchy[depth];
    if (address->IsFieldEmpty(field)) {
      return;
    }
    const std::string& field_value = address->GetFieldValue(field);
    bool no_match_found_yet = true;
    for (const auto& sub_key : parent_rule->GetSubKeys()) {
      if (!no_match_found_yet) {
        break;
      }
      for (const std::string& language_tag : languages) {
        lookup_key.set_language(language_tag);
        lookup_key.FromLookupKey(parent_key, sub_key);
        const Rule* rule = supplier_->GetRule(lookup_key);
        if (rule == nullptr) continue;
        bool matches_latin_name =
            compare_->NaturalEquals(field_value, rule->GetLatinName());
        bool matches_local_name_id =
            compare_->NaturalEquals(field_value, sub_key) ||
            compare_->NaturalEquals(field_value, rule->GetName());
        if (matches_latin_name || matches_local_name_id) {
          address->SetFieldValue(
              field, matches_latin_name ? rule->GetLatinName() : sub_key);
          no_match_found_yet = false;
          parent_key.FromLookupKey(parent_key, sub_key);
          parent_rule = supplier_->GetRule(parent_key);
          assert(parent_rule != nullptr);
          break;
        }
      }
    }
    if (no_match_found_yet) {
      return;  
    }
  }
}
}  
}  