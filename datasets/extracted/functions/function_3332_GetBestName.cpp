#include <libaddressinput/address_input_helper.h>
#include <libaddressinput/address_data.h>
#include <libaddressinput/address_field.h>
#include <libaddressinput/address_metadata.h>
#include <libaddressinput/preload_supplier.h>
#include <cassert>
#include <cstddef>
#include <string>
#include <vector>
#include <re2/re2.h>
#include "language.h"
#include "lookup_key.h"
#include "region_data_constants.h"
#include "rule.h"
#include "util/re2ptr.h"
#include "util/size.h"
namespace i18n {
namespace addressinput {
struct Node {
  const Node* parent;
  const Rule* rule;
};
namespace {
const char kLookupKeySeparator = '/';
const size_t kHierarchyDepth = size(LookupKey::kHierarchy);
std::string GetBestName(const Language& language, const Rule& rule) {
  if (language.has_latin_script) {
    const std::string& name = rule.GetLatinName();
    if (!name.empty()) {
      return name;
    }
  }
  const std::string& id = rule.GetId();
  std::string::size_type pos = id.rfind(kLookupKeySeparator);
  assert(pos != std::string::npos);
  return id.substr(pos + 1);
}
void FillAddressFromMatchedRules(
    const std::vector<Node>* hierarchy,
    AddressData* address) {
  assert(hierarchy != nullptr);
  assert(address != nullptr);
  Language language(address->language_code);
  for (size_t depth = kHierarchyDepth - 1; depth > 0; --depth) {
    if (hierarchy[depth].size() == 1) {
      for (const Node* node = &hierarchy[depth].front();
           node != nullptr; node = node->parent, --depth) {
        const Rule* rule = node->rule;
        assert(rule != nullptr);
        AddressField field = LookupKey::kHierarchy[depth];
        if (address->IsFieldEmpty(field)) {
          address->SetFieldValue(field, GetBestName(language, *rule));
        }
      }
      break;
    }
  }
}
}  
AddressInputHelper::AddressInputHelper(PreloadSupplier* supplier)
    : supplier_(supplier) {
  assert(supplier_ != nullptr);
}
void AddressInputHelper::FillAddress(AddressData* address) const {
  assert(address != nullptr);
  const std::string& region_code = address->region_code;
  if (!RegionDataConstants::IsSupported(region_code)) {
    return;
  }
  AddressData lookup_key_address;
  lookup_key_address.region_code = region_code;
  LookupKey lookup_key;
  lookup_key.FromAddress(lookup_key_address);
  const Rule* region_rule = supplier_->GetRule(lookup_key);
  assert(region_rule != nullptr);
  const RE2ptr* postal_code_reg_exp = region_rule->GetPostalCodeMatcher();
  if (postal_code_reg_exp != nullptr) {
    if (address->postal_code.empty()) {
      address->postal_code = region_rule->GetSolePostalCode();
    }
    if (!address->postal_code.empty() &&
        RE2::FullMatch(address->postal_code, *postal_code_reg_exp->ptr)) {
      std::vector<Node> hierarchy[kHierarchyDepth];
      CheckChildrenForPostCodeMatches(*address, lookup_key, nullptr, hierarchy);
      FillAddressFromMatchedRules(hierarchy, address);
    }
  }
}
void AddressInputHelper::CheckChildrenForPostCodeMatches(
    const AddressData& address,
    const LookupKey& lookup_key,
    const Node* parent,
    std::vector<Node>* hierarchy) const {
  const Rule* rule = supplier_->GetRule(lookup_key);
  assert(rule != nullptr);
  const RE2ptr* postal_code_prefix = rule->GetPostalCodeMatcher();
  if (postal_code_prefix == nullptr ||
      RE2::PartialMatch(address.postal_code, *postal_code_prefix->ptr)) {
    size_t depth = lookup_key.GetDepth();
    assert(depth < size(LookupKey::kHierarchy));
    hierarchy[depth].emplace_back();
    Node* node = &hierarchy[depth].back();
    node->parent = parent;
    node->rule = rule;
    if (depth < size(LookupKey::kHierarchy) - 1 &&
        IsFieldUsed(LookupKey::kHierarchy[depth + 1], address.region_code)) {
      for (const auto& sub_key : rule->GetSubKeys()) {
        LookupKey child_key;
        child_key.FromLookupKey(lookup_key, sub_key);
        CheckChildrenForPostCodeMatches(address, child_key, node, hierarchy);
      }
    }
  }
}
}  
}  