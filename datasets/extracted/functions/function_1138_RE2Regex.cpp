#include "arolla/qtype/strings/regex.h"
#include <memory>
#include <string>
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "arolla/qtype/simple_qtype.h"
#include "arolla/util/fingerprint.h"
#include "arolla/util/repr.h"
#include "re2/re2.h"
namespace arolla {
namespace {
class RE2Regex final : public Regex {
 public:
  explicit RE2Regex(absl::string_view pattern) : re2_(pattern, RE2::Quiet) {}
  bool ok() const { return re2_.ok(); }
  absl::string_view error() const { return re2_.error(); }
  absl::string_view pattern() const final { return re2_.pattern(); }
  int NumberOfCapturingGroups() const final {
    return re2_.NumberOfCapturingGroups();
  }
  bool PartialMatch(absl::string_view text) const final {
    return re2_.PartialMatch(text, re2_);
  }
  bool PartialMatch(absl::string_view text, std::string* match) const final {
    return RE2::PartialMatch(text, re2_, match);
  }
 private:
  RE2 re2_;
};
}  
absl::StatusOr<absl::Nonnull<RegexPtr>> CompileRegex(
    absl::string_view pattern) {
  auto result = std::make_shared<RE2Regex>(pattern);
  if (result->ok()) {
    return result;
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "invalid regular expression: `", pattern, "`; ", result->error()));
}
void FingerprintHasherTraits<RegexPtr>::operator()(
    FingerprintHasher* hasher, const RegexPtr& value) const {
  if (value != nullptr) {
    hasher->Combine(value->pattern());
  }
}
ReprToken ReprTraits<RegexPtr>::operator()(const RegexPtr& value) const {
  if (value == nullptr) {
    return {"regex{}"};
  }
  return {absl::StrCat("regex{`", value->pattern(), "`}")};
}
AROLLA_DEFINE_SIMPLE_QTYPE(REGEX, RegexPtr)
}  