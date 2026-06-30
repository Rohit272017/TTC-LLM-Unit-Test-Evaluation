#include <libaddressinput/region_data_builder.h>
#include <libaddressinput/address_data.h>
#include <libaddressinput/preload_supplier.h>
#include <libaddressinput/region_data.h>
#include <cassert>
#include <cstddef>
#include <string>
#include <vector>
#include "language.h"
#include "lookup_key.h"
#include "region_data_constants.h"
#include "rule.h"
#include "util/size.h"
namespace i18n {
namespace addressinput {
namespace {
const size_t kLookupKeysMaxDepth = size(LookupKey::kHierarchy) - 1;
void BuildRegionTreeRecursively(
    const std::map<std::string, const Rule*>& rules,
    std::map<std::string, const Rule*>::const_iterator hint,
    const LookupKey& parent_key,
    RegionData* parent_region,
    const std::vector<std::string>& keys,
    bool prefer_latin_name,
    size_t region_max_depth) {
  assert(parent_region != nullptr);
  LookupKey lookup_key;
  for (const auto& key : keys) {
    lookup_key.FromLookupKey(parent_key, key);
    const std::string lookup_key_string =
        lookup_key.ToKeyString(kLookupKeysMaxDepth);
    ++hint;
    if (hint == rules.end() || hint->first != lookup_key_string) {
      hint = rules.find(lookup_key_string);
      if (hint == rules.end()) {
        return;
      }
    }
    const Rule* rule = hint->second;
    assert(rule != nullptr);
    const std::string& local_name = rule->GetName().empty()
        ? key : rule->GetName();
    const std::string& name =
        prefer_latin_name && !rule->GetLatinName().empty()
            ? rule->GetLatinName() : local_name;
    RegionData* region = parent_region->AddSubRegion(key, name);
    if (!rule->GetSubKeys().empty() &&
        region_max_depth > parent_key.GetDepth()) {
      BuildRegionTreeRecursively(rules,
                                 hint,
                                 lookup_key,
                                 region,
                                 rule->GetSubKeys(),
                                 prefer_latin_name,
                                 region_max_depth);
    }
  }
}
RegionData* BuildRegion(const std::map<std::string, const Rule*>& rules,
                        const std::string& region_code,
                        const Language& language) {
  AddressData address;
  address.region_code = region_code;
  LookupKey lookup_key;
  lookup_key.FromAddress(address);
  auto hint = rules.find(lookup_key.ToKeyString(kLookupKeysMaxDepth));
  assert(hint != rules.end());
  const Rule* rule = hint->second;
  assert(rule != nullptr);
  auto* region = new RegionData(region_code);
  size_t region_max_depth =
      RegionDataConstants::GetMaxLookupKeyDepth(region_code);
  if (region_max_depth > 0) {
    BuildRegionTreeRecursively(rules,
                               hint,
                               lookup_key,
                               region,
                               rule->GetSubKeys(),
                               language.has_latin_script,
                               region_max_depth);
  }
  return region;
}
}  
RegionDataBuilder::RegionDataBuilder(PreloadSupplier* supplier)
    : supplier_(supplier),
      cache_() {
  assert(supplier_ != nullptr);
}
RegionDataBuilder::~RegionDataBuilder() {
  for (const auto& outer : cache_) {
    assert(outer.second != nullptr);
    for (const auto& inner : *outer.second) {
      delete inner.second;
    }
    delete outer.second;
  }
}
const RegionData& RegionDataBuilder::Build(
    const std::string& region_code,
    const std::string& ui_language_tag,
    std::string* best_region_tree_language_tag) {
  assert(supplier_->IsLoaded(region_code));
  assert(best_region_tree_language_tag != nullptr);
  auto region_it = cache_.find(region_code);
  if (region_it == cache_.end()) {
    region_it = cache_.emplace(region_code, new LanguageRegionMap).first;
  }
  Rule rule;
  rule.ParseSerializedRule(RegionDataConstants::GetRegionData(region_code));
  static const Language kUndefinedLanguage("und");
  const Language best_language =
      rule.GetLanguages().empty()
          ? kUndefinedLanguage
          : ChooseBestAddressLanguage(rule, Language(ui_language_tag));
  *best_region_tree_language_tag = best_language.tag;
  auto language_it = region_it->second->find(best_language.tag);
  if (language_it == region_it->second->end()) {
    const auto& rules = supplier_->GetRulesForRegion(region_code);
    language_it = region_it->second
                      ->emplace(best_language.tag,
                                BuildRegion(rules, region_code, best_language))
                      .first;
  }
  return *language_it->second;
}
}  
}  