#include "xla/service/select_and_scatter_expander.h"
#include <numeric>
#include <vector>
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/literal_util.h"
#include "xla/service/call_inliner.h"
namespace xla {
absl::StatusOr<HloInstruction*> SelectAndScatterExpander::ExpandInstruction(
    HloInstruction* instruction) {
  auto* computation = instruction->parent();
  auto* sas = Cast<HloSelectAndScatterInstruction>(instruction);
  auto* operand = sas->mutable_operand(0);
  auto operand_shape = operand->shape();
  auto* source = sas->mutable_operand(1);
  auto* select = sas->select();
  auto* init_value = sas->mutable_operand(2);
  const auto iota_shape = ShapeUtil::ChangeElementType(operand_shape, S32);
  const auto scalar_operand =
      ShapeUtil::MakeScalarShape(operand->shape().element_type());
  const auto scalar_iota =
      ShapeUtil::MakeScalarShape(iota_shape.element_type());
  const auto source_shape = source->shape();
  const Shape iota_shape_reduced =
      ShapeUtil::ChangeElementType(source_shape, S32);
  std::vector<HloInstruction*> iotas;
  iotas.reserve(operand_shape.rank());
  for (int i = 0; i < operand_shape.rank(); ++i) {
    iotas.push_back(
        computation->AddInstruction(HloInstruction::CreateIota(iota_shape, i)));
  }
  HloComputation* new_comp = [&]() -> HloComputation* {
    HloComputation::Builder builder(
        absl::StrCat(select->name(), ".reduce_window"));
    auto rhs_begin = static_cast<int64_t>(iotas.size() + 1);
    auto first_iota_index = 1;
    auto* neg_one = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(-1)));
    auto* first_lhs_iota =
        builder.AddInstruction(HloInstruction::CreateParameter(
            first_iota_index, scalar_iota, "iota_lhs"));
    auto* first_rhs_iota =
        builder.AddInstruction(HloInstruction::CreateParameter(
            first_iota_index + rhs_begin, scalar_iota, "iota_lhs"));
    auto* lhs_first_in_window =
        builder.AddInstruction(HloInstruction::CreateCompare(
            sas->select()->root_instruction()->shape(), first_lhs_iota, neg_one,
            Comparison::Direction::kNe, {}));
    auto* rhs_first_in_window =
        builder.AddInstruction(HloInstruction::CreateCompare(
            sas->select()->root_instruction()->shape(), first_rhs_iota, neg_one,
            Comparison::Direction::kNe, {}));
    auto rhs_not_first_in_window = builder.AddInstruction(
        HloInstruction::CreateUnary(sas->select()->root_instruction()->shape(),
                                    HloOpcode::kNot, rhs_first_in_window));
    auto* operand_lhs = builder.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_operand, "operand_lhs"));
    auto* operand_rhs = builder.AddInstruction(HloInstruction::CreateParameter(
        rhs_begin, scalar_operand, "operand_rhs"));
    auto* call = builder.AddInstruction(
        HloInstruction::CreateCall(sas->select()->root_instruction()->shape(),
                                   {operand_lhs, operand_rhs}, sas->select()));
    auto* pred = builder.AddInstruction(HloInstruction::CreateBinary(
        call->shape(), HloOpcode::kAnd, call, lhs_first_in_window));
    pred = builder.AddInstruction(HloInstruction::CreateBinary(
        call->shape(), HloOpcode::kOr, pred, rhs_not_first_in_window));
    std::vector<HloInstruction*> result_tuple;
    result_tuple.push_back(builder.AddInstruction(HloInstruction::CreateTernary(
        scalar_operand, HloOpcode::kSelect, pred, operand_lhs, operand_rhs)));
    for (auto i = first_iota_index; i < rhs_begin; ++i) {
      xla::HloInstruction *iota_lhs, *iota_rhs;
      if (i == first_iota_index) {
        iota_lhs = first_lhs_iota;
        iota_rhs = first_rhs_iota;
      } else {
        iota_lhs = builder.AddInstruction(
            HloInstruction::CreateParameter(i, scalar_iota, "iota_lhs"));
        iota_rhs = builder.AddInstruction(HloInstruction::CreateParameter(
            i + rhs_begin, scalar_iota, "iota_rhs"));
      }
      result_tuple.push_back(
          builder.AddInstruction(HloInstruction::CreateTernary(
              scalar_iota, HloOpcode::kSelect, pred, iota_lhs, iota_rhs)));
    }
    builder.AddInstruction(HloInstruction::CreateTuple(result_tuple));
    auto* result = select->parent()->AddEmbeddedComputation(builder.Build());
    if (!CallInliner::Inline(call).ok()) {
      return nullptr;
    }
    return result;
  }();
  if (!new_comp) {
    return nullptr;
  }
  auto num_reduce_values = iotas.size() + 1;
  std::vector<HloInstruction*> ops;
  ops.reserve(num_reduce_values);
  ops.push_back(operand);
  ops.insert(ops.end(), iotas.begin(), iotas.end());
  auto* neg_one = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(-1)));
  std::vector<HloInstruction*> reduce_init_values;
  reduce_init_values.reserve(num_reduce_values);
  reduce_init_values.push_back(init_value);
  for (auto i = 0; i < iotas.size(); ++i) {
    reduce_init_values.push_back(neg_one);
  }
  std::vector<xla::Shape> shapes;
  shapes.reserve(num_reduce_values);
  shapes.push_back(source->shape());
  for (auto i = 0; i < iotas.size(); ++i) {
    shapes.push_back(iota_shape_reduced);
  }
  auto* reduce_window =
      computation->AddInstruction(HloInstruction::CreateReduceWindow(
          ShapeUtil::MakeTupleShape(shapes), ops, reduce_init_values,
          sas->window(), new_comp));
  std::vector<HloInstruction*> iota_indices;
  std::vector<int64_t> broadcasted_iota_dims;
  broadcasted_iota_dims.reserve(iota_shape_reduced.rank() + 1);
  broadcasted_iota_dims.insert(broadcasted_iota_dims.end(),
                               iota_shape_reduced.dimensions().begin(),
                               iota_shape_reduced.dimensions().end());
  broadcasted_iota_dims.push_back(1);
  auto broadcasted_iota_shape = ShapeUtil::MakeShape(
      iota_shape_reduced.element_type(), broadcasted_iota_dims);
  for (int i = 1; i < num_reduce_values; ++i) {
    auto* element = computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(reduce_window, i));
    iota_indices.push_back(computation->AddInstruction(
        HloInstruction::CreateReshape(broadcasted_iota_shape, element)));
  }
  std::vector<int64_t> scatter_dims(operand->shape().rank());
  std::iota(scatter_dims.begin(), scatter_dims.end(), 0);
  auto* broadcasted_init_value = computation->AddInstruction(
      HloInstruction::CreateBroadcast(instruction->shape(), init_value, {}));
  std::vector<int64_t> concatenated_iotas_dims;
  concatenated_iotas_dims.reserve(iota_indices.front()->shape().rank());
  concatenated_iotas_dims.insert(concatenated_iotas_dims.end(),
                                 broadcasted_iota_dims.begin(),
                                 broadcasted_iota_dims.end());
  concatenated_iotas_dims.back() = static_cast<int64_t>(iota_indices.size());
  auto* indices = computation->AddInstruction(HloInstruction::CreateConcatenate(
      ShapeUtil::MakeShape(iota_shape.element_type(), concatenated_iotas_dims),
      iota_indices, iota_shape.rank()));
  ScatterDimensionNumbers dim_nums =
      HloScatterInstruction::MakeScatterDimNumbers(
          {},
          scatter_dims,
          scatter_dims,
          source->shape().rank());
  return computation->AddInstruction(HloInstruction::CreateScatter(
      sas->shape(), broadcasted_init_value,
      indices, source,
      sas->scatter(), dim_nums,
      false, false));
}
bool SelectAndScatterExpander::InstructionMatchesPattern(
    HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kSelectAndScatter;
}
}  