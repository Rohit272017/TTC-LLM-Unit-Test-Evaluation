#include "eval/eval/regex_match_step.h"
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "common/casting.h"
#include "common/value.h"
#include "eval/eval/attribute_trail.h"
#include "eval/eval/direct_expression_step.h"
#include "eval/eval/evaluator_core.h"
#include "eval/eval/expression_step_base.h"
#include "internal/status_macros.h"
#include "re2/re2.h"
namespace google::api::expr::runtime {
namespace {
using ::cel::BoolValue;
using ::cel::Cast;
using ::cel::ErrorValue;
using ::cel::InstanceOf;
using ::cel::StringValue;
using ::cel::UnknownValue;
using ::cel::Value;
inline constexpr int kNumRegexMatchArguments = 1;
inline constexpr size_t kRegexMatchStepSubject = 0;
struct MatchesVisitor final {
  const RE2& re;
  bool operator()(const absl::Cord& value) const {
    if (auto flat = value.TryFlat(); flat.has_value()) {
      return RE2::PartialMatch(*flat, re);
    }
    return RE2::PartialMatch(static_cast<std::string>(value), re);
  }
  bool operator()(absl::string_view value) const {
    return RE2::PartialMatch(value, re);
  }
};
class RegexMatchStep final : public ExpressionStepBase {
 public:
  RegexMatchStep(int64_t expr_id, std::shared_ptr<const RE2> re2)
      : ExpressionStepBase(expr_id, true),
        re2_(std::move(re2)) {}
  absl::Status Evaluate(ExecutionFrame* frame) const override {
    if (!frame->value_stack().HasEnough(kNumRegexMatchArguments)) {
      return absl::Status(absl::StatusCode::kInternal,
                          "Insufficient arguments supplied for regular "
                          "expression match");
    }
    auto input_args = frame->value_stack().GetSpan(kNumRegexMatchArguments);
    const auto& subject = input_args[kRegexMatchStepSubject];
    if (!subject->Is<cel::StringValue>()) {
      return absl::Status(absl::StatusCode::kInternal,
                          "First argument for regular "
                          "expression match must be a string");
    }
    bool match = subject.GetString().NativeValue(MatchesVisitor{*re2_});
    frame->value_stack().Pop(kNumRegexMatchArguments);
    frame->value_stack().Push(frame->value_factory().CreateBoolValue(match));
    return absl::OkStatus();
  }
 private:
  const std::shared_ptr<const RE2> re2_;
};
class RegexMatchDirectStep final : public DirectExpressionStep {
 public:
  RegexMatchDirectStep(int64_t expr_id,
                       std::unique_ptr<DirectExpressionStep> subject,
                       std::shared_ptr<const RE2> re2)
      : DirectExpressionStep(expr_id),
        subject_(std::move(subject)),
        re2_(std::move(re2)) {}
  absl::Status Evaluate(ExecutionFrameBase& frame, Value& result,
                        AttributeTrail& attribute) const override {
    AttributeTrail subject_attr;
    CEL_RETURN_IF_ERROR(subject_->Evaluate(frame, result, subject_attr));
    if (InstanceOf<ErrorValue>(result) ||
        cel::InstanceOf<UnknownValue>(result)) {
      return absl::OkStatus();
    }
    if (!InstanceOf<StringValue>(result)) {
      return absl::Status(absl::StatusCode::kInternal,
                          "First argument for regular "
                          "expression match must be a string");
    }
    bool match = Cast<StringValue>(result).NativeValue(MatchesVisitor{*re2_});
    result = BoolValue(match);
    return absl::OkStatus();
  }
 private:
  std::unique_ptr<DirectExpressionStep> subject_;
  const std::shared_ptr<const RE2> re2_;
};
}  
std::unique_ptr<DirectExpressionStep> CreateDirectRegexMatchStep(
    int64_t expr_id, std::unique_ptr<DirectExpressionStep> subject,
    std::shared_ptr<const RE2> re2) {
  return std::make_unique<RegexMatchDirectStep>(expr_id, std::move(subject),
                                                std::move(re2));
}
absl::StatusOr<std::unique_ptr<ExpressionStep>> CreateRegexMatchStep(
    std::shared_ptr<const RE2> re2, int64_t expr_id) {
  return std::make_unique<RegexMatchStep>(expr_id, std::move(re2));
}
}  