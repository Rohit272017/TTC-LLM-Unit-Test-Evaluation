#include "arolla/expr/eval/compile_std_function_operator.h"
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "arolla/expr/eval/executable_builder.h"
#include "arolla/expr/expr_node.h"
#include "arolla/expr/expr_operator_signature.h"
#include "arolla/expr/operators/std_function_operator.h"
#include "arolla/memory/frame.h"
#include "arolla/qexpr/bound_operators.h"
#include "arolla/qexpr/eval_context.h"
#include "arolla/qtype/typed_ref.h"
#include "arolla/qtype/typed_slot.h"
#include "arolla/util/status_macros_backport.h"
namespace arolla::expr::eval_internal {
absl::Status CompileStdFunctionOperator(
    const expr_operators::StdFunctionOperator& std_function_op,
    absl::Span<const TypedSlot> input_slots, TypedSlot output_slot,
    ExecutableBuilder& executable_builder, ExprNodePtr node) {
  RETURN_IF_ERROR(ValidateDepsCount(std_function_op.signature(),
                                    input_slots.size(),
                                    absl::StatusCode::kFailedPrecondition));
  auto fn = std_function_op.GetEvalFn();
  int64_t ip = executable_builder.AddEvalOp(
      MakeBoundOperator([fn, output_slot,
                         input_slots = std::vector(input_slots.begin(),
                                                   input_slots.end())](
                            EvaluationContext* ctx, FramePtr frame) {
        std::vector<TypedRef> inputs;
        inputs.reserve(input_slots.size());
        for (const auto input_slot : input_slots) {
          inputs.push_back(TypedRef::FromSlot(input_slot, frame));
        }
        ASSIGN_OR_RETURN(auto res, fn(inputs), ctx->set_status(std::move(_)));
        if (res.GetType() != output_slot.GetType()) {
          ctx->set_status(absl::InvalidArgumentError(absl::StrFormat(
              "expected the result to have qtype %s, got %s",
              output_slot.GetType()->name(), res.GetType()->name())));
          return;
        }
        ctx->set_status(res.CopyToSlot(output_slot, frame));
      }),
      FormatOperatorCall(std_function_op.display_name(), input_slots,
                         {output_slot}),
      std::string(std_function_op.display_name()));
  executable_builder.RegisterStacktrace(ip, node);
  return absl::OkStatus();
}
}  