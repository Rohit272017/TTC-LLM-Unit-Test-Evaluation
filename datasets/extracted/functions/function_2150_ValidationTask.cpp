#include "validation_task.h"
#include <libaddressinput/address_data.h>
#include <libaddressinput/address_field.h>
#include <libaddressinput/address_metadata.h>
#include <libaddressinput/address_problem.h>
#include <libaddressinput/address_validator.h>
#include <libaddressinput/callback.h>
#include <libaddressinput/supplier.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <string>
#include <re2/re2.h>
#include "lookup_key.h"
#include "post_box_matchers.h"
#include "rule.h"
#include "util/re2ptr.h"
#include "util/size.h"
namespace i18n {
namespace addressinput {
ValidationTask::ValidationTask(const AddressData& address, bool allow_postal,
                               bool require_name, const FieldProblemMap* filter,
                               FieldProblemMap* problems,
                               const AddressValidator::Callback& validated)
    : address_(address),
      allow_postal_(allow_postal),
      require_name_(require_name),
      filter_(filter),
      problems_(problems),
      validated_(validated),
      supplied_(BuildCallback(this, &ValidationTask::Validate)),
      lookup_key_(new LookupKey),
      max_depth_(size(LookupKey::kHierarchy)) {
  assert(problems_ != nullptr);
  assert(supplied_ != nullptr);
  assert(lookup_key_ != nullptr);
}
ValidationTask::~ValidationTask() = default;
void ValidationTask::Run(Supplier* supplier) {
  assert(supplier != nullptr);
  problems_->clear();
  lookup_key_->FromAddress(address_);
  max_depth_ = supplier->GetLoadedRuleDepth(lookup_key_->ToKeyString(0));
  supplier->SupplyGlobally(*lookup_key_, *supplied_);
}
void ValidationTask::Validate(bool success,
                              const LookupKey& lookup_key,
                              const Supplier::RuleHierarchy& hierarchy) {
  assert(&lookup_key == lookup_key_.get());  
  if (success) {
    if (address_.IsFieldEmpty(COUNTRY)) {
      ReportProblemMaybe(COUNTRY, MISSING_REQUIRED_FIELD);
    } else if (hierarchy.rule[0] == nullptr) {
      ReportProblemMaybe(COUNTRY, UNKNOWN_VALUE);
    } else {
      const std::string& region_code = address_.region_code;
      CheckUnexpectedField(region_code);
      CheckMissingRequiredField(region_code);
      CheckUnknownValue(hierarchy);
      CheckPostalCodeFormatAndValue(hierarchy);
      CheckUsesPoBox(hierarchy);
      CheckUnsupportedField();
    }
  }
  validated_(success, address_, *problems_);
  delete this;
}
void ValidationTask::CheckUnexpectedField(
    const std::string& region_code) const {
  static const AddressField kFields[] = {
      ADMIN_AREA,
      LOCALITY,
      DEPENDENT_LOCALITY,
      SORTING_CODE,
      POSTAL_CODE,
      STREET_ADDRESS,
      ORGANIZATION,
      RECIPIENT,
  };
  for (AddressField field : kFields) {
    if (!address_.IsFieldEmpty(field) && !IsFieldUsed(field, region_code)) {
      ReportProblemMaybe(field, UNEXPECTED_FIELD);
    }
  }
}
void ValidationTask::CheckMissingRequiredField(
    const std::string& region_code) const {
  static const AddressField kFields[] = {
      ADMIN_AREA,
      LOCALITY,
      DEPENDENT_LOCALITY,
      SORTING_CODE,
      POSTAL_CODE,
      STREET_ADDRESS,
  };
  for (AddressField field : kFields) {
    if (address_.IsFieldEmpty(field) && IsFieldRequired(field, region_code)) {
      ReportProblemMaybe(field, MISSING_REQUIRED_FIELD);
    }
  }
  if (require_name_ && address_.IsFieldEmpty(RECIPIENT)) {
    ReportProblemMaybe(RECIPIENT, MISSING_REQUIRED_FIELD);
  }
}
void ValidationTask::CheckUnknownValue(
    const Supplier::RuleHierarchy& hierarchy) const {
  for (size_t depth = 1; depth < size(LookupKey::kHierarchy); ++depth) {
    AddressField field = LookupKey::kHierarchy[depth];
    if (!(address_.IsFieldEmpty(field) ||
          hierarchy.rule[depth - 1] == nullptr ||
          hierarchy.rule[depth - 1]->GetSubKeys().empty() ||
          hierarchy.rule[depth] != nullptr)) {
      ReportProblemMaybe(field, UNKNOWN_VALUE);
    }
  }
}
void ValidationTask::CheckUnsupportedField() const {
  for (size_t depth = max_depth_; depth < size(LookupKey::kHierarchy);
       ++depth) {
    ReportProblemMaybe(LookupKey::kHierarchy[depth], UNSUPPORTED_FIELD);
  }
}
void ValidationTask::CheckPostalCodeFormatAndValue(
    const Supplier::RuleHierarchy& hierarchy) const {
  assert(hierarchy.rule[0] != nullptr);
  const Rule& country_rule = *hierarchy.rule[0];
  if (!(ShouldReport(POSTAL_CODE, INVALID_FORMAT) ||
        ShouldReport(POSTAL_CODE, MISMATCHING_VALUE))) {
    return;
  }
  if (address_.IsFieldEmpty(POSTAL_CODE)) {
    return;
  } else if (std::find(problems_->begin(), problems_->end(),
                       FieldProblemMap::value_type(POSTAL_CODE,
                                                   UNEXPECTED_FIELD))
             != problems_->end()) {
    return;  
  }
  const RE2ptr* format_ptr = country_rule.GetPostalCodeMatcher();
  if (format_ptr != nullptr &&
      !RE2::FullMatch(address_.postal_code, *format_ptr->ptr) &&
      ShouldReport(POSTAL_CODE, INVALID_FORMAT)) {
    ReportProblem(POSTAL_CODE, INVALID_FORMAT);
    return;
  }
  if (!ShouldReport(POSTAL_CODE, MISMATCHING_VALUE)) {
    return;
  }
  for (size_t depth = size(LookupKey::kHierarchy) - 1; depth > 0; --depth) {
    if (hierarchy.rule[depth] != nullptr) {
      const RE2ptr* prefix_ptr = hierarchy.rule[depth]->GetPostalCodeMatcher();
      if (prefix_ptr != nullptr) {
        if (!RE2::PartialMatch(address_.postal_code, *prefix_ptr->ptr)) {
          ReportProblem(POSTAL_CODE, MISMATCHING_VALUE);
        }
        return;
      }
    }
  }
}
void ValidationTask::CheckUsesPoBox(
    const Supplier::RuleHierarchy& hierarchy) const {
  assert(hierarchy.rule[0] != nullptr);
  const Rule& country_rule = *hierarchy.rule[0];
  if (allow_postal_ ||
      !ShouldReport(STREET_ADDRESS, USES_P_O_BOX) ||
      address_.IsFieldEmpty(STREET_ADDRESS)) {
    return;
  }
  const auto matchers = PostBoxMatchers::GetMatchers(country_rule);
  for (const auto& line : address_.address_line) {
    for (auto ptr : matchers) {
      assert(ptr != nullptr);
      if (RE2::PartialMatch(line, *ptr->ptr)) {
        ReportProblem(STREET_ADDRESS, USES_P_O_BOX);
        return;
      }
    }
  }
}
void ValidationTask::ReportProblem(AddressField field,
                                   AddressProblem problem) const {
  problems_->emplace(field, problem);
}
void ValidationTask::ReportProblemMaybe(AddressField field,
                                        AddressProblem problem) const {
  if (ShouldReport(field, problem)) {
    ReportProblem(field, problem);
  }
}
bool ValidationTask::ShouldReport(AddressField field,
                                  AddressProblem problem) const {
  return filter_ == nullptr || filter_->empty() ||
         std::find(filter_->begin(),
                   filter_->end(),
                   FieldProblemMap::value_type(field, problem)) !=
             filter_->end();
}
}  
}  