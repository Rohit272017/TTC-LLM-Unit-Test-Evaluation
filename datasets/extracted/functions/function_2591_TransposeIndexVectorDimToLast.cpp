#include "xla/service/gather_expander.h"
#include <utility>
#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/while_util.h"
#include "xla/util.h"
namespace xla {
namespace {
absl::StatusOr<HloInstruction*> TransposeIndexVectorDimToLast(
    HloInstruction* start_indices, int64_t index_vector_dim) {
  const Shape& start_indices_shape = start_indices->shape();
  if (start_indices_shape.dimensions_size() == index_vector_dim) {
    return start_indices;
  }
  if (index_vector_dim == (start_indices_shape.dimensions_size() - 1)) {
    return start_indices;
  }
  std::vector<int64_t> permutation;
  permutation.reserve(start_indices_shape.dimensions_size());
  for (int64_t i = 0, e = start_indices_shape.dimensions_size(); i < e; i++) {
    if (i != index_vector_dim) {
      permutation.push_back(i);
    }
  }
  permutation.push_back(index_vector_dim);
  return MakeTransposeHlo(start_indices, permutation);
}
absl::StatusOr<HloInstruction*> CanonicalizeGatherIndices(
    HloInstruction* start_indices, int64_t index_vector_dim) {
  TF_ASSIGN_OR_RETURN(
      HloInstruction * transposed_start_indices,
      TransposeIndexVectorDimToLast(start_indices, index_vector_dim));
  bool indices_are_scalar =
      index_vector_dim == start_indices->shape().dimensions_size();
  const int64_t index_dims_in_start_indices = indices_are_scalar ? 0 : 1;
  const Shape& shape = transposed_start_indices->shape();
  if (shape.dimensions_size() == index_dims_in_start_indices) {
    return PrependDegenerateDims(transposed_start_indices, 1);
  } else {
    return CollapseFirstNDims(
        transposed_start_indices,
        shape.dimensions_size() - index_dims_in_start_indices);
  }
}
absl::StatusOr<HloInstruction*> AdjustBatchDimsInAccumulator(
    const Shape& start_indices_shape, HloInstruction* accumulator,
    int64_t index_vector_dim) {
  std::vector<int64_t> batch_dim_bounds;
  batch_dim_bounds.reserve(start_indices_shape.dimensions_size());
  for (int64_t i = 0, e = start_indices_shape.dimensions_size(); i < e; i++) {
    if (i != index_vector_dim) {
      batch_dim_bounds.push_back(start_indices_shape.dimensions(i));
    }
  }
  if (batch_dim_bounds.empty()) {
    return ElideDegenerateDims(accumulator, {0});
  }
  return ExpandFirstDimIntoNDims(accumulator, batch_dim_bounds);
}
absl::StatusOr<HloInstruction*> ExpandIndexVectorIntoOperandSpace(
    HloInstruction* index_vector, const GatherDimensionNumbers& dim_numbers,
    int64_t operand_rank) {
  HloComputation* computation = index_vector->parent();
  const Shape& index_shape = index_vector->shape();
  if (operand_rank == 0) {
    return computation->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateFromDimensions(index_shape.element_type(), {0})));
  }
  HloInstruction* zero =
      computation->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateFromDimensions(index_shape.element_type(), {1})));
  std::vector<HloInstruction*> expanded_index_components;
  for (int i = 0; i < operand_rank; i++) {
    int64_t index_vector_dim_index =
        FindIndex(dim_numbers.start_index_map(), i);
    if (index_vector_dim_index != dim_numbers.start_index_map_size()) {
      TF_ASSIGN_OR_RETURN(
          HloInstruction * component_to_concat,
          MakeSliceHlo(index_vector, {index_vector_dim_index},
                       {index_vector_dim_index + 1},
                       {1}));
      expanded_index_components.push_back(component_to_concat);
    } else {
      expanded_index_components.push_back(zero);
    }
  }
  return MakeConcatHlo(expanded_index_components, 0);
}
absl::StatusOr<std::vector<HloInstruction*>> GatherLoopBody(
    const HloInstruction& gather, HloInstruction* induction_var,
    const std::vector<HloInstruction*>& incoming_loop_state) {
  const GatherDimensionNumbers& dim_numbers = gather.gather_dimension_numbers();
  CHECK_EQ(incoming_loop_state.size(), 3);
  HloInstruction* const operand = incoming_loop_state[0];
  HloInstruction* const start_indices = incoming_loop_state[1];
  HloInstruction* const output_accumulator = incoming_loop_state[2];
  bool has_scalar_indices = start_indices->shape().dimensions_size() == 1;
  CHECK_EQ(has_scalar_indices,
           dim_numbers.index_vector_dim() ==
               gather.operand(1)->shape().dimensions_size());
  HloInstruction* induction_var_as_vector =
      MakeBroadcastHlo(induction_var, {},
                       {1});
  HloInstruction* index_vector;
  if (has_scalar_indices) {
    TF_ASSIGN_OR_RETURN(
        index_vector,
        MakeDynamicSliceHlo(start_indices, induction_var_as_vector, {1}));
  } else {
    TF_ASSIGN_OR_RETURN(
        HloInstruction * index_into_start_indices,
        PadVectorWithZeros(induction_var_as_vector,
                           0, 1));
    int64_t index_vector_size = start_indices->shape().dimensions(1);
    TF_ASSIGN_OR_RETURN(
        HloInstruction * index_vector_2d,
        MakeDynamicSliceHlo(start_indices, index_into_start_indices,
                            {1, index_vector_size}));
    TF_ASSIGN_OR_RETURN(index_vector,
                        ElideDegenerateDims(index_vector_2d, {0}));
  }
  TF_ASSIGN_OR_RETURN(
      HloInstruction * gathered_slice_start,
      ExpandIndexVectorIntoOperandSpace(index_vector, dim_numbers,
                                        operand->shape().dimensions_size()));
  TF_ASSIGN_OR_RETURN(HloInstruction * gathered_slice,
                      MakeDynamicSliceHlo(operand, gathered_slice_start,
                                          gather.gather_slice_sizes()));
  TF_ASSIGN_OR_RETURN(
      HloInstruction* const gathered_slice_with_dims_collapsed,
      ElideDegenerateDims(gathered_slice, dim_numbers.collapsed_slice_dims()));
  TF_ASSIGN_OR_RETURN(
      HloInstruction* const gathered_slice_for_update,
      PrependDegenerateDims(gathered_slice_with_dims_collapsed, 1));
  TF_ASSIGN_OR_RETURN(
      HloInstruction* const index_vector_into_accumulator,
      PadVectorWithZeros(
          induction_var_as_vector, 0,
          gathered_slice_with_dims_collapsed->shape().dimensions_size()));
  TF_ASSIGN_OR_RETURN(
      HloInstruction* const updated_accumulator,
      MakeDynamicUpdateSliceHlo(output_accumulator, gathered_slice_for_update,
                                index_vector_into_accumulator));
  return absl::StatusOr<std::vector<HloInstruction*>>{
      {operand, start_indices, updated_accumulator}};
}
HloInstruction* CreateGatherLoopAccumulatorInitValue(
    HloComputation* computation, PrimitiveType element_type,
    absl::Span<const int64_t> slice_sizes, int64_t gather_loop_trip_count,
    const GatherDimensionNumbers& dim_numbers) {
  std::vector<int64_t> accumulator_state_shape_dims;
  accumulator_state_shape_dims.reserve(1 + slice_sizes.size());
  accumulator_state_shape_dims.push_back(gather_loop_trip_count);
  for (int64_t i = 0; i < slice_sizes.size(); i++) {
    if (!absl::c_binary_search(dim_numbers.collapsed_slice_dims(), i)) {
      accumulator_state_shape_dims.push_back(slice_sizes[i]);
    }
  }
  return BroadcastZeros(computation, element_type,
                        accumulator_state_shape_dims);
}
absl::StatusOr<HloInstruction*> PermuteBatchAndOffsetDims(
    HloInstruction* accumulator, absl::Span<const int64_t> offset_dims,
    int64_t output_rank) {
  std::vector<int64_t> permutation;
  permutation.reserve(output_rank);
  int64_t batch_idx_counter = 0;
  int64_t offset_idx_counter = output_rank - offset_dims.size();
  for (int64_t i = 0; i < output_rank; i++) {
    bool is_offset_dim = absl::c_binary_search(offset_dims, i);
    if (is_offset_dim) {
      permutation.push_back(offset_idx_counter++);
    } else {
      permutation.push_back(batch_idx_counter++);
    }
  }
  return MakeTransposeHlo(accumulator, permutation);
}
int64_t GatherLoopTripCount(HloInstruction* gather_instr) {
  HloInstruction* start_indices = gather_instr->mutable_operand(1);
  const Shape& start_indices_shape = start_indices->shape();
  const GatherDimensionNumbers& dim_numbers =
      gather_instr->gather_dimension_numbers();
  int64_t trip_count = 1;
  for (int64_t i = 0, e = start_indices_shape.dimensions_size(); i < e; i++) {
    if (i != dim_numbers.index_vector_dim()) {
      trip_count *= start_indices_shape.dimensions(i);
    }
  }
  return trip_count;
}
int64_t GatherIsBroadcast(HloInstruction* gather_instr) {
  return absl::c_equal(gather_instr->gather_slice_sizes(),
                       gather_instr->operand(0)->shape().dimensions());
}
}  
absl::StatusOr<HloInstruction*> GatherExpander::ExpandInstruction(
    HloInstruction* gather_instr) {
  CHECK(!ShapeUtil::IsZeroElementArray(gather_instr->shape()));
  if (GatherIsBroadcast(gather_instr)) {
    if (ShapeUtil::IsZeroElementArray(gather_instr->operand(0)->shape())) {
      return MakeScalarLike(gather_instr, 0);
    }
    Shape broadcast_operand_shape = ShapeUtil::DeleteDimensions(
        gather_instr->gather_dimension_numbers().collapsed_slice_dims(),
        gather_instr->operand(0)->shape());
    TF_ASSIGN_OR_RETURN(HloInstruction * broadcast_operand,
                        MakeReshapeHlo(broadcast_operand_shape,
                                       gather_instr->mutable_operand(0)));
    gather_instr->SetupDerivedInstruction(broadcast_operand);
    HloInstruction* broadcast =
        MakeBroadcastHlo(broadcast_operand,
                         gather_instr->gather_dimension_numbers().offset_dims(),
                         gather_instr->shape());
    gather_instr->SetupDerivedInstruction(broadcast);
    return broadcast;
  }
  HloComputation* computation = gather_instr->parent();
  HloInstruction* operand = gather_instr->mutable_operand(0);
  HloInstruction* start_indices = gather_instr->mutable_operand(1);
  const Shape& output_shape = gather_instr->shape();
  int64_t output_rank = output_shape.dimensions_size();
  const GatherDimensionNumbers& dim_numbers =
      gather_instr->gather_dimension_numbers();
  int64_t gather_loop_trip_count = GatherLoopTripCount(gather_instr);
  if (!IsInt32(gather_loop_trip_count)) {
    return Unimplemented(
        "Gather operations with more than 2147483647 gather indices are not "
        "supported. This error occurred for %s.",
        gather_instr->ToString());
  }
  TF_ASSIGN_OR_RETURN(
      HloInstruction * canonical_start_indices,
      CanonicalizeGatherIndices(start_indices, dim_numbers.index_vector_dim()));
  CHECK_EQ(gather_loop_trip_count,
           canonical_start_indices->shape().dimensions(0));
  HloInstruction* accumulator_init = CreateGatherLoopAccumulatorInitValue(
      computation, output_shape.element_type(),
      gather_instr->gather_slice_sizes(), gather_loop_trip_count,
      gather_instr->gather_dimension_numbers());
  absl::StatusOr<std::vector<HloInstruction*>> gather_loop_result_or_error =
      WhileUtil::MakeCountedLoop(
          computation, gather_loop_trip_count,
          {operand, canonical_start_indices, accumulator_init},
          [&](HloInstruction* indvar,
              const std::vector<HloInstruction*>& loop_state) {
            return GatherLoopBody(*gather_instr, indvar, loop_state);
          },
          gather_instr->metadata());
  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> gather_loop_result,
                      gather_loop_result_or_error);
  HloInstruction* accumulator_result = gather_loop_result.back();
  TF_ASSIGN_OR_RETURN(
      HloInstruction* const accumulator_with_batch_dims_decanonicalized,
      AdjustBatchDimsInAccumulator(start_indices->shape(), accumulator_result,
                                   dim_numbers.index_vector_dim()));
  return PermuteBatchAndOffsetDims(accumulator_with_batch_dims_decanonicalized,
                                   dim_numbers.offset_dims(), output_rank);
}
bool GatherExpander::InstructionMatchesPattern(HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kGather &&
         !ShapeUtil::IsZeroElementArray(inst->shape()) &&
         (mode_ == kEliminateAllGathers || GatherLoopTripCount(inst) == 1 ||
          absl::c_equal(inst->gather_slice_sizes(),
                        inst->operand(0)->shape().dimensions()));
}
}  