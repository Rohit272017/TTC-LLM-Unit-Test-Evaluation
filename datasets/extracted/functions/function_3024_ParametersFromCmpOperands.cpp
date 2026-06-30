#include "xla/service/gpu/transforms/sort_rewriter.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/runtime/cub_sort_thunk.h"
#include "xla/service/stable_sort_expander.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace gpu {
namespace {
struct SortComputationAnalysis {
  int key_operand;  
  bool descending;
};
std::pair<int64_t, int64_t> ParametersFromCmpOperands(
    const HloCompareInstruction* cmp_op) {
  if (cmp_op == nullptr) {
    return std::pair<int64_t, int64_t>(-1, -1);
  }
  const HloParameterInstruction* param0 =
      DynCast<HloParameterInstruction>(cmp_op->operand(0));
  const HloParameterInstruction* param1 =
      DynCast<HloParameterInstruction>(cmp_op->operand(1));
  return (param0 && param1) ? std::make_pair(param0->parameter_number(),
                                             param1->parameter_number())
                            : std::pair<int64_t, int64_t>(-1, -1);
}
std::optional<SortComputationAnalysis> AnalyzeCompareOp(
    const HloInstruction* maybe_compare_op) {
  const HloCompareInstruction* compare =
      DynCast<HloCompareInstruction>(maybe_compare_op);
  if (compare == nullptr || compare->direction() == ComparisonDirection::kEq ||
      compare->direction() == ComparisonDirection::kNe) {
    return std::nullopt;
  }
  auto [index0, index1] = ParametersFromCmpOperands(compare);
  if (index0 == -1 || index1 == -1) {
    return std::nullopt;
  }
  int first_index = std::min(index0, index1);
  if (first_index % 2 != 0 || std::max(index0, index1) != first_index + 1) {
    return std::nullopt;
  }
  bool descending = compare->direction() == ComparisonDirection::kGt ||
                    compare->direction() == ComparisonDirection::kGe;
  bool reverse = first_index != index0;
  return SortComputationAnalysis{first_index / 2, descending != reverse};
}
std::optional<SortComputationAnalysis> AnalyzeComplexSortComputation(
    const HloSortInstruction& sort_op) {
  auto computation = sort_op.called_computations().front();
  if (computation->num_parameters() != 4) {
    return std::nullopt;
  }
  int64_t iota_operand_index =
      StableSortExpander::IotaOperandIndexForStableSort(sort_op);
  if (iota_operand_index < 0) {
    return std::nullopt;
  }
  auto root = computation->root_instruction();
  if (root->opcode() != HloOpcode::kSelect) {
    return std::nullopt;
  }
  auto iota_cmp = DynCast<HloCompareInstruction>(root->operand(1));
  auto [iotap0, iotap1] = ParametersFromCmpOperands(iota_cmp);
  if (iota_cmp == nullptr ||
      iota_cmp->direction() != ComparisonDirection::kLt ||
      iotap0 != iota_operand_index * 2 ||
      iotap1 != iota_operand_index * 2 + 1) {
    return std::nullopt;
  }
  auto eq_cmp = DynCast<HloCompareInstruction>(root->operand(0));
  if (eq_cmp == nullptr || eq_cmp->direction() != ComparisonDirection::kEq) {
    return std::nullopt;
  }
  auto [p0, p1] = ParametersFromCmpOperands(eq_cmp);
  if (p0 < 0 || p1 < 0) {
    auto cmp = DynCast<HloCompareInstruction>(eq_cmp->operand(0));
    auto cmp_reverse = DynCast<HloCompareInstruction>(eq_cmp->operand(1));
    auto [a, b] = ParametersFromCmpOperands(cmp);
    auto [p, q] = ParametersFromCmpOperands(cmp_reverse);
    if (cmp == nullptr || cmp_reverse == nullptr || a < 0 || b < 0 || a != q ||
        b != p || cmp->direction() != cmp_reverse->direction() ||
        cmp->direction() == Comparison::Direction::kEq ||
        cmp->direction() == Comparison::Direction::kNe) {
      return std::nullopt;
    }
  }
  return AnalyzeCompareOp(root->operand(2));
}
std::optional<SortComputationAnalysis> AnalyzeSortOp(
    const HloSortInstruction& sort_op) {
  auto computation = sort_op.called_computations().front();
  auto result = AnalyzeCompareOp(computation->root_instruction());
  if (!result.has_value()) {
    result = AnalyzeComplexSortComputation(sort_op);
  }
  return result;
}
absl::StatusOr<std::unique_ptr<CubSortRunnerInterface>> CreateRunner(
    HloSortInstruction* sort_op, const SortComputationAnalysis& sort_config) {
  int value_index = 1 - sort_config.key_operand;
  return CubSortRunnerInterface::Create(
      sort_op->operand(sort_config.key_operand)->shape().element_type(),
      sort_op->operand_count() == 2
          ? std::optional(sort_op->operand(value_index)->shape().element_type())
          : std::nullopt);
}
bool IsCubCompatibleSort(HloSortInstruction* sort_op) {
  VLOG(1) << "Sort instruction: " << sort_op->name();
  if (sort_op->operand_count() != 1 && sort_op->operand_count() != 2) {
    VLOG(2) << "Unsupported operand count: " << sort_op->operand_count();
    return false;
  }
  const Shape& operand_shape = sort_op->operand(0)->shape();
  if (sort_op->sort_dimension() != operand_shape.rank() - 1) {
    VLOG(2) << "Sort dimension should be the minor one";
    return false;
  }
  if (Product(operand_shape.dimensions()) < SortRewriter::SortSizeThreshold()) {
    VLOG(2) << "Tensor shape size is too small to see an improvement";
    return false;
  }
  auto sort_config = AnalyzeSortOp(*sort_op);
  if (!sort_config.has_value()) {
    VLOG(2) << "Only simple compare computations are supported";
    return false;
  }
  if (!CreateRunner(sort_op, *sort_config).ok()) {
    VLOG(2) << "Unsupported operand types (no compiled CUB kernels)";
    return false;
  }
  VLOG(2) << "Sort operation is compatible";
  return true;
}
HloInstruction* UnpackResultPair(HloSortInstruction* sort_op,
                                 HloInstruction* custom_call, bool swap) {
  HloComputation* parent = sort_op->parent();
  HloInstruction* gte0 =
      parent->AddInstruction(HloInstruction::CreateGetTupleElement(
          sort_op->operand(0)->shape(), custom_call, swap ? 1 : 0));
  HloInstruction* gte1 =
      parent->AddInstruction(HloInstruction::CreateGetTupleElement(
          sort_op->operand(1)->shape(), custom_call, swap ? 0 : 1));
  return parent->AddInstruction(HloInstruction::CreateTuple({gte0, gte1}));
}
}  
absl::StatusOr<bool> SortRewriter::RunOnInstruction(
    HloSortInstruction* sort_op) {
  SortComputationAnalysis sort_config = AnalyzeSortOp(*sort_op).value();
  const Shape& operand_shape = sort_op->operand(0)->shape();
  int64_t batch_size = Product(operand_shape.dimensions()) /
                       operand_shape.dimensions(sort_op->sort_dimension());
  TF_ASSIGN_OR_RETURN(auto runner, CreateRunner(sort_op, sort_config));
  TF_ASSIGN_OR_RETURN(
      int64_t scratch_size,
      runner->GetScratchSize(Product(operand_shape.dimensions()), batch_size));
  if (batch_size > 1) {
    scratch_size += sizeof(int) - scratch_size % sizeof(int);
    scratch_size += (batch_size + 1) * sizeof(int);
  }
  HloInstruction* keys = sort_op->mutable_operand(0);
  HloInstruction* values = nullptr;
  if (sort_op->operand_count() == 2) {
    values = sort_op->mutable_operand(1);
    if (sort_config.key_operand == 1) {
      std::swap(keys, values);
    }
  }
  std::vector<Shape> shapes{keys->shape()};
  std::vector<HloInstruction*> operands{keys};
  if (values != nullptr) {
    shapes.push_back(values->shape());
    operands.push_back(values);
  }
  shapes.push_back(ShapeUtil::MakeShape(U8, {scratch_size}));
  Shape call_shape = ShapeUtil::MakeTupleShape(absl::MakeSpan(shapes));
  HloInstruction* custom_call =
      sort_op->parent()->AddInstruction(HloInstruction::CreateCustomCall(
          call_shape, absl::MakeSpan(operands), kCubDeviceRadixSortTarget));
  xla::SortOptions backend_config;
  backend_config.set_descending(sort_config.descending);
  TF_RETURN_IF_ERROR(custom_call->set_backend_config(backend_config));
  HloInstruction* replacement;
  if (sort_op->operand_count() == 1) {
    replacement =
        sort_op->parent()->AddInstruction(HloInstruction::CreateGetTupleElement(
            sort_op->shape(), custom_call, 0));
  } else {
    replacement = UnpackResultPair(sort_op, custom_call,
                                   sort_config.key_operand == 1);
  }
  TF_RETURN_IF_ERROR(
      sort_op->parent()->ReplaceInstruction(sort_op, replacement));
  return true;
}
absl::StatusOr<bool> SortRewriter::RunOnComputation(
    HloComputation* computation) {
  std::vector<HloSortInstruction*> sort_ops;
  for (auto* inst : computation->instructions()) {
    HloSortInstruction* sort = DynCast<HloSortInstruction>(inst);
    if (sort != nullptr && IsCubCompatibleSort(sort)) {
      sort_ops.push_back(sort);
    }
  }
  bool changed = false;
  for (auto* sort : sort_ops) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnInstruction(sort));
    changed |= result;
  }
  return changed;
}
absl::StatusOr<bool> SortRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(2, "SortRewriter::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  XLA_VLOG_LINES(2, "SortRewriter::Run(), after:\n" + module->ToString());
  return changed;
}
}  
}  