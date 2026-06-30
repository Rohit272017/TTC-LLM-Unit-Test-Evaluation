#include "xla/service/gpu/transforms/algebraic_simplifier.h"
#include "absl/log/check.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/fusions/triton/triton_support_legacy.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/xla_data.pb.h"
namespace xla::gpu {
bool GpuAlgebraicSimplifierVisitor::ShouldStrengthReduceDotToReduce(
    const HloInstruction* hlo) {
  if (!options_.enable_dot_strength_reduction()) {
    return false;
  }
  const HloDotInstruction* dot = DynCast<HloDotInstruction>(hlo);
  if (dot == nullptr) {
    return false;
  }
  const HloInstruction* lhs = dot->operand(0);
  const HloInstruction* rhs = dot->operand(1);
  DotDimensionNumbers dnums = dot->dot_dimension_numbers();
  bool lhs_is_vector = (dnums.lhs_batch_dimensions_size() +
                            dnums.lhs_contracting_dimensions_size() ==
                        lhs->shape().rank());
  bool rhs_is_vector = (dnums.rhs_batch_dimensions_size() +
                            dnums.rhs_contracting_dimensions_size() ==
                        rhs->shape().rank());
  if (lhs_is_vector && rhs_is_vector) {
    return true;
  }
  absl::StatusOr<bool> is_too_small =
      IsMatrixMultiplicationTooSmallForRewriting(*hlo, 10000000);
  CHECK_OK(is_too_small.status());
  if (is_too_small.value()) {
    return true;
  }
  return !legacy_triton::CanTritonHandleGEMM(*dot, compute_capability_);
}
}  