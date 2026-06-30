#include "string_compare.h"
#include <cassert>
#include <string>
#include <re2/re2.h>
#include "lru_cache_using_std.h"
namespace {
std::string ComputeMinPossibleMatch(const std::string& str) {
  std::string min, max;
  RE2::Options options;
  options.set_literal(true);
  options.set_case_sensitive(false);
  RE2 matcher(str, options);
  bool success = matcher.PossibleMatchRange(&min, &max, str.size());
  assert(success);
  (void)success;  
  return min;
}
}  
namespace i18n {
namespace addressinput {
class StringCompare::Impl {
  enum { MAX_CACHE_SIZE = 1 << 15 };
 public:
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl() : min_possible_match_(&ComputeMinPossibleMatch, MAX_CACHE_SIZE) {
    options_.set_literal(true);
    options_.set_case_sensitive(false);
  }
  ~Impl() = default;
  bool NaturalEquals(const std::string& a, const std::string& b) const {
    RE2 matcher(b, options_);
    return RE2::FullMatch(a, matcher);
  }
  bool NaturalLess(const std::string& a, const std::string& b) const {
    const std::string& min_a(min_possible_match_(a));
    const std::string& min_b(min_possible_match_(b));
    return min_a < min_b;
  }
 private:
  RE2::Options options_;
  mutable lru_cache_using_std<std::string, std::string> min_possible_match_;
};
StringCompare::StringCompare() : impl_(new Impl) {}
StringCompare::~StringCompare() = default;
bool StringCompare::NaturalEquals(const std::string& a,
                                  const std::string& b) const {
  return impl_->NaturalEquals(a, b);
}
bool StringCompare::NaturalLess(const std::string& a,
                                const std::string& b) const {
  return impl_->NaturalLess(a, b);
}
}  
}  