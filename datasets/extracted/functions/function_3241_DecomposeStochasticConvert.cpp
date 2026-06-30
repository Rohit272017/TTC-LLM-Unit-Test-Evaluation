#include "xla/service/stochastic_convert_decomposer.h"
#include <cstdint>
#include <limits>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/shape_inference.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace xla {
absl::Status DecomposeStochasticConvert(HloComputation* comp,
                                        HloInstruction* instruction) {
  CHECK(instruction->opcode() == HloOpcode::kStochasticConvert)
      << "requires a stochastic_convert instruction to decompose, but got: "
      << instruction->opcode();
  CHECK(instruction->operand_count() == 2)
      << "requires 2 operands for stochastic convert, but got: "
      << instruction->operand_count();
  HloInstruction* operand = instruction->mutable_operand(0);
  HloInstruction* random = instruction->mutable_operand(1);
  PrimitiveType from_type = operand->shape().element_type();
  PrimitiveType random_type = random->shape().element_type();
  PrimitiveType to_type = instruction->shape().element_type();
  TF_RETURN_IF_ERROR(ShapeInference::InferStochasticConvertShape(
                         operand->shape(), random->shape(), to_type)
                         .status());
  VLOG(1) << "Decomposing instruction: " << instruction->ToString();
  if (primitive_util::IsSignedIntegralType(to_type)) {
    TF_ASSIGN_OR_RETURN(HloInstruction * operand_sign,
                        MakeUnaryHlo(HloOpcode::kSign, operand));
    TF_ASSIGN_OR_RETURN(HloInstruction * should_neg,
                        MakeCompareHlo(Comparison::Direction::kLt, operand_sign,
                                       MakeScalarLike(operand_sign, 0)));
    TF_ASSIGN_OR_RETURN(HloInstruction * operand_abs,
                        MakeUnaryHlo(HloOpcode::kAbs, operand));
    TF_ASSIGN_OR_RETURN(HloInstruction * truncated_fp,
                        MakeUnaryHlo(HloOpcode::kFloor, operand_abs));
    TF_ASSIGN_OR_RETURN(
        HloInstruction * fractional,
        MakeBinaryHlo(HloOpcode::kSubtract, operand_abs, truncated_fp));
    if (from_type == F16) {
      fractional = MakeConvertToHlo(fractional, F32);
    }
    TF_ASSIGN_OR_RETURN(
        HloInstruction * fixed_fractional,
        MakeBinaryHlo(
            HloOpcode::kMultiply, fractional,
            MakeScalarLike(fractional, IPow<double>(2, primitive_util::BitWidth(
                                                           random_type)))));
    TF_ASSIGN_OR_RETURN(
        HloInstruction * should_round_up,
        MakeCompareHlo(Comparison::Direction::kLt, random,
                       MakeConvertToHlo(fixed_fractional, random_type)));
    HloInstruction* truncated_int = MakeConvertToHlo(truncated_fp, to_type);
    TF_ASSIGN_OR_RETURN(
        truncated_int,
        MakeSelectHlo(should_round_up,
                      MakeBinaryHlo(HloOpcode::kAdd, truncated_int,
                                    MakeScalarLike(truncated_int, 1))
                          .value(),
                      truncated_int));
    TF_ASSIGN_OR_RETURN(
        HloInstruction * result,
        MakeSelectHlo(should_neg,
                      MakeUnaryHlo(HloOpcode::kNegate, truncated_int).value(),
                      truncated_int));
    auto to_bits = primitive_util::BitWidth(to_type);
    auto min = static_cast<int64_t>(
        (static_cast<uint64_t>(1) + ~static_cast<uint64_t>(1))
        << (to_bits - 1));
    TF_ASSIGN_OR_RETURN(HloInstruction * is_min,
                        MakeCompareHlo(Comparison::Direction::kLe, operand,
                                       MakeScalarLike(operand, min)));
    TF_ASSIGN_OR_RETURN(
        result, MakeSelectHlo(is_min, MakeScalarLike(result, min), result));
    auto max =
        static_cast<int64_t>((static_cast<uint64_t>(1) << (to_bits - 1)) - 1);
    TF_ASSIGN_OR_RETURN(HloInstruction * is_max,
                        MakeCompareHlo(Comparison::Direction::kGe, operand,
                                       MakeScalarLike(operand, max)));
    TF_ASSIGN_OR_RETURN(
        result, MakeSelectHlo(is_max, MakeScalarLike(result, max), result));
    TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(result));
    TF_RETURN_IF_ERROR(comp->RemoveInstruction(instruction));
    return absl::OkStatus();
  }
  return Internal("Unsupported stochastic convert: from %s to %s",
                       PrimitiveType_Name(from_type),
                       PrimitiveType_Name(to_type));
}
absl::StatusOr<bool> StochasticConvertDecomposer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() != HloOpcode::kStochasticConvert) {
        continue;
      }
      TF_RETURN_IF_ERROR(DecomposeStochasticConvert(computation, instruction));
      changed = true;
    }
  }
  return changed;
}
}  