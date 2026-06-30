#include "eval/eval/compiler_constant_step.h"
#include "absl/status/status.h"
#include "common/value.h"
#include "eval/eval/attribute_trail.h"
#include "eval/eval/evaluator_core.h"
namespace google::api::expr::runtime {
using ::cel::Value;
absl::Status DirectCompilerConstantStep::Evaluate(
    ExecutionFrameBase& frame, Value& result, AttributeTrail& attribute) const {
  result = value_;
  return absl::OkStatus();
}
absl::Status CompilerConstantStep::Evaluate(ExecutionFrame* frame) const {
  frame->value_stack().Push(value_);
  return absl::OkStatus();
}
}  