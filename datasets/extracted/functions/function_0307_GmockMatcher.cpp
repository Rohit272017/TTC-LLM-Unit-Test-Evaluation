#ifndef XLA_SERVICE_PATTERN_MATCHER_GMOCK_H_
#define XLA_SERVICE_PATTERN_MATCHER_GMOCK_H_
#include <ostream>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/layout.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/test.h"
#include "tsl/platform/test.h"
namespace xla {
namespace pattern_matcher_gmock_detail {
template <typename Pattern>
class GmockMatcher {
 public:
  explicit GmockMatcher(Pattern p) : pattern_(std::move(p)) {}
  bool MatchAndExplain(const Layout& l,
                       ::testing::MatchResultListener* listener) const {
    return MatchAndExplainImpl(&l, listener);
  }
  bool MatchAndExplain(const Layout* l,
                       ::testing::MatchResultListener* listener) const {
    return MatchAndExplainImpl(l, listener);
  }
  bool MatchAndExplain(Layout* l,
                       ::testing::MatchResultListener* listener) const {
    return MatchAndExplainImpl(l, listener);
  }
  bool MatchAndExplain(const Shape& s,
                       ::testing::MatchResultListener* listener) const {
    return MatchAndExplainImpl(&s, listener);
  }
  bool MatchAndExplain(const Shape* s,
                       ::testing::MatchResultListener* listener) const {
    return MatchAndExplainImpl(s, listener);
  }
  bool MatchAndExplain(Shape* s,
                       ::testing::MatchResultListener* listener) const {
    return MatchAndExplainImpl(s, listener);
  }
  bool MatchAndExplain(const HloInstruction& instr,
                       ::testing::MatchResultListener* listener) const {
    return MatchAndExplainImpl(&instr, listener);
  }
  bool MatchAndExplain(const HloInstruction* instr,
                       ::testing::MatchResultListener* listener) const {
    return MatchAndExplainImpl(instr, listener);
  }
  bool MatchAndExplain(HloInstruction* instr,
                       ::testing::MatchResultListener* listener) const {
    return MatchAndExplainImpl(instr, listener);
  }
  void DescribeTo(std::ostream* os) const { pattern_.DescribeTo(os); }
  void DescribeNegationTo(std::ostream* os) const {
    *os << "is NOT: ";
    DescribeTo(os);
  }
 private:
  template <typename T>
  bool MatchAndExplainImpl(T* t,
                           ::testing::MatchResultListener* listener) const {
    MatchOption options{true, false,
                        listener->stream()};
    return Match(t, pattern_, options);
  }
  Pattern pattern_;
};
}  
template <typename Pattern>
::testing::PolymorphicMatcher<
    pattern_matcher_gmock_detail::GmockMatcher<Pattern>>
GmockMatch(Pattern&& p) {
  return ::testing::MakePolymorphicMatcher(
      pattern_matcher_gmock_detail::GmockMatcher<Pattern>(
          std::forward<Pattern>(p)));
}
}  
#endif  