#include "xla/service/reduce_window_rewriter.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/window_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace xla {
static size_t FlattenShapeIndex(const ShapeIndex& shape_index) {
  if (shape_index.empty()) {
    return 0;
  }
  CHECK_EQ(shape_index.size(), 1);
  return shape_index.back();
}
static Shape ShapeAtIndex(const Shape& shape, const ShapeIndex& shape_index) {
  if (shape_index.empty()) {
    return shape;
  }
  CHECK_EQ(shape_index.size(), 1);
  return ShapeUtil::GetTupleElementShape(shape, shape_index.back());
}
static HloInstruction* GetAtIndex(HloInstruction* hlo,
                                  const ShapeIndex& shape_index) {
  if (shape_index.empty()) {
    return hlo;
  }
  CHECK_EQ(shape_index.size(), 1);
  return hlo->parent()->AddInstruction(HloInstruction::CreateGetTupleElement(
      ShapeAtIndex(hlo->shape(), shape_index), hlo, shape_index.back()));
}
absl::Status ReduceWindowRewriter::ReplaceReduceWindowWithReshape(
    HloReduceWindowInstruction* reduce_window) {
  VLOG(2) << "Converting R1 reduce window: " << reduce_window->ToString();
  std::vector<Shape> r2_output_shapes;
  ShapeUtil::ForEachSubshape(
      reduce_window->shape(),
      [&](const Shape& subshape, const ShapeIndex& shape_index) {
        if (!ShapeUtil::IsLeafIndex(reduce_window->shape(), shape_index)) {
          return;
        }
        Shape r2_output_shape = subshape;
        ShapeUtil::AppendMajorDimension(1, &r2_output_shape);
        UpdateLayout(&r2_output_shape);
        r2_output_shapes.push_back(r2_output_shape);
        VLOG(2) << "ReduceWindowRewriter: Converting R2 result to R1: "
                << ShapeUtil::HumanStringWithLayout(r2_output_shape);
      });
  Window r2_window = reduce_window->window();
  WindowDimension* dim = r2_window.add_dimensions();
  dim->set_size(1);
  dim->set_stride(1);
  dim->set_base_dilation(1);
  dim->set_window_dilation(1);
  std::vector<HloInstruction*> r2_operands;
  for (HloInstruction* operand : reduce_window->inputs()) {
    Shape r2_input_shape = operand->shape();
    ShapeUtil::AppendMajorDimension(1, &r2_input_shape);
    UpdateLayout(&r2_input_shape);
    VLOG(2) << "ReduceWindowRewriter: Converting R1 operand to R2: "
            << ShapeUtil::HumanStringWithLayout(r2_input_shape);
    HloInstruction* r2_operand = operand->parent()->AddInstruction(
        HloInstruction::CreateReshape(r2_input_shape, operand));
    VLOG(2) << "R2 new operand: " << r2_operand->ToString();
    r2_operands.push_back(r2_operand);
  }
  HloInstruction* new_reduce_window = reduce_window->parent()->AddInstruction(
      HloInstruction::CreateReduceWindow(
          reduce_window->shape().IsTuple()
              ? ShapeUtil::MakeTupleShape(r2_output_shapes)
              : r2_output_shapes[0],
          r2_operands, reduce_window->init_values(), r2_window,
          reduce_window->to_apply()));
  VLOG(2) << "R2 resulting reduce window: " << new_reduce_window->ToString();
  std::vector<HloInstruction*> final_reshapes;
  ShapeUtil::ForEachSubshape(
      reduce_window->shape(),
      [&](const Shape& subshape, const ShapeIndex& shape_index) {
        if (!ShapeUtil::IsLeafIndex(reduce_window->shape(), shape_index)) {
          return;
        }
        HloInstruction* final_reshape =
            new_reduce_window->parent()->AddInstruction(
                HloInstruction::CreateReshape(
                    subshape, GetAtIndex(new_reduce_window, shape_index)));
        final_reshapes.push_back(final_reshape);
      });
  HloInstruction* result;
  if (reduce_window->shape().IsTuple()) {
    result = new_reduce_window->parent()->AddInstruction(
        HloInstruction::CreateTuple(final_reshapes));
  } else {
    CHECK_EQ(final_reshapes.size(), 1);
    result = final_reshapes[0];
  }
  TF_RETURN_IF_ERROR(reduce_window->ReplaceAllUsesWith(result));
  TF_RETURN_IF_ERROR(
      new_reduce_window->parent()->RemoveInstruction(reduce_window));
  return absl::OkStatus();
}
absl::StatusOr<bool> ReduceWindowRewriter::TryOptimizeCumSumOrProd(
    HloReduceWindowInstruction* reduce_window) {
  const Shape& operand_shape = reduce_window->inputs().front()->shape();
  int64_t rank = operand_shape.rank();
  const Window& window = reduce_window->window();
  int64_t scan_dim_num = -1;
  for (int i = 0; i < rank; ++i) {
    const WindowDimension& window_dim = window.dimensions(i);
    if (window_util::IsTrivialWindowDimension(window_dim)) {
      continue;
    }
    if (scan_dim_num != -1) {
      return false;
    }
    scan_dim_num = i;
  }
  if (scan_dim_num == -1) {
    return false;
  }
  const int64_t scan_length = operand_shape.dimensions(scan_dim_num);
  absl::Span<HloInstruction* const> init_values = reduce_window->init_values();
  const WindowDimension& scan_window_dim = window.dimensions(scan_dim_num);
  bool forward_scan = (scan_window_dim.padding_low() == scan_length - 1 ||
                       scan_window_dim.padding_low() == scan_length) &&
                      scan_window_dim.padding_high() == 0;
  bool reverse_scan = (scan_window_dim.padding_high() == scan_length - 1 ||
                       scan_window_dim.padding_high() == scan_length) &&
                      scan_window_dim.padding_low() == 0;
  if (scan_window_dim.stride() != 1 || scan_window_dim.size() != scan_length ||
      (!forward_scan && !reverse_scan) || scan_window_dim.window_reversal() ||
      scan_window_dim.base_dilation() != 1 ||
      scan_window_dim.window_dilation() != 1) {
    return false;
  }
  bool is_exclusive = forward_scan
                          ? (scan_window_dim.padding_low() == scan_length)
                          : (scan_window_dim.padding_high() == scan_length);
  if (scan_length <= base_length_) {
    return false;
  }
  if (reduce_window->to_apply()->root_instruction()->shape().IsTuple() &&
      reduce_window->to_apply()->root_instruction()->opcode() !=
          HloOpcode::kTuple) {
    return false;
  }
  VLOG(2) << "Rewriting Scan: " << reduce_window->ToString();
  HloComputation* parent = reduce_window->parent();
  std::vector<HloInstruction*> sources(reduce_window->inputs().begin(),
                                       reduce_window->inputs().end());
  std::vector<int64_t> permutation(rank);
  absl::c_iota(permutation, 0);
  permutation[scan_dim_num] = rank - 1;
  permutation[rank - 1] = scan_dim_num;
  if (scan_dim_num != rank - 1) {
    for (size_t i = 0; i < sources.size(); ++i) {
      sources[i] = parent->AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::PermuteDimensions(permutation, sources[i]->shape()),
          sources[i], permutation));
    }
  }
  const int64_t padded_length = RoundUpTo(scan_length, base_length_);
  if (scan_length != padded_length) {
    for (size_t i = 0; i < sources.size(); ++i) {
      auto* source = sources[i];
      Shape padded_shape = source->shape();
      padded_shape.set_dimensions(rank - 1, padded_length);
      UpdateLayout(&padded_shape);
      auto padding_config = MakeNoPaddingConfig(rank);
      padding_config.mutable_dimensions(rank - 1)->set_edge_padding_high(
          padded_length - scan_length);
      sources[i] = parent->AddInstruction(HloInstruction::CreatePad(
          padded_shape, source, init_values[i], padding_config));
    }
  }
  const int64_t num_columns = padded_length / base_length_;
  std::vector<HloInstruction*> tiled_sources;
  std::vector<Shape> tiled_shapes;
  for (size_t i = 0; i < sources.size(); ++i) {
    auto* source = sources[i];
    Shape tiled_shape = source->shape();
    tiled_shape.set_dimensions(rank - 1, num_columns);
    UpdateLayout(&tiled_shape);
    ShapeUtil::AppendMajorDimension(base_length_, &tiled_shape);
    tiled_shapes.push_back(tiled_shape);
    tiled_sources.push_back(parent->AddInstruction(
        HloInstruction::CreateReshape(tiled_shape, source)));
  }
  Window outer_window =
      window_util::MakeWindow(std::vector<int64_t>(rank + 1, 1));
  outer_window.mutable_dimensions(rank)->set_size(base_length_);
  if (forward_scan) {
    outer_window.mutable_dimensions(rank)->set_padding_low(base_length_ - 1);
  } else {
    outer_window.mutable_dimensions(rank)->set_padding_high(base_length_ - 1);
  }
  auto outer_reduce_window =
      parent->AddInstruction(HloInstruction::CreateReduceWindow(
          reduce_window->shape().IsTuple()
              ? ShapeUtil::MakeTupleShape(tiled_shapes)
              : tiled_shapes[0],
          tiled_sources, init_values, outer_window, reduce_window->to_apply()));
  std::vector<Shape> column_shapes;
  std::vector<HloInstruction*> last_cols;
  ShapeUtil::ForEachSubshape(
      outer_reduce_window->shape(),
      [&](const Shape& subshape, const ShapeIndex& shape_index) {
        if (!ShapeUtil::IsLeafIndex(outer_reduce_window->shape(),
                                    shape_index)) {
          return;
        }
        Shape column_shape = subshape;
        column_shape.set_dimensions(rank, 1);
        UpdateLayout(&column_shape);
        std::vector<int64_t> col_slice_starts(rank + 1, 0);
        std::vector<int64_t> col_slice_limits(
            SpanToVector(subshape.dimensions()));
        if (forward_scan) {
          col_slice_starts[rank] = base_length_ - 1;
        } else {
          col_slice_limits[rank] = 1;
        }
        auto last_col = parent->AddInstruction(HloInstruction::CreateSlice(
            column_shape, GetAtIndex(outer_reduce_window, shape_index),
            col_slice_starts, col_slice_limits,
            std::vector<int64_t>(rank + 1, 1)));
        column_shape.DeleteDimension(rank);
        last_col = parent->AddInstruction(
            HloInstruction::CreateReshape(column_shape, last_col));
        last_cols.push_back(last_col);
        column_shape.set_dimensions(rank - 1, num_columns + 1);
        UpdateLayout(&column_shape);
        column_shapes.push_back(column_shape);
      });
  Window inner_window = window_util::MakeWindow(std::vector<int64_t>(rank, 1));
  inner_window.mutable_dimensions(rank - 1)->set_size(num_columns);
  if (forward_scan) {
    inner_window.mutable_dimensions(rank - 1)->set_padding_low(num_columns);
  } else {
    inner_window.mutable_dimensions(rank - 1)->set_padding_high(num_columns);
  }
  auto inner_reduce_window =
      parent->AddInstruction(HloInstruction::CreateReduceWindow(
          reduce_window->shape().IsTuple()
              ? ShapeUtil::MakeTupleShape(column_shapes)
              : column_shapes[0],
          last_cols, init_values, inner_window, reduce_window->to_apply()));
  std::vector<int64_t> exclusive_slice_starts(rank, 0);
  std::vector<int64_t> exclusive_slice_limits =
      SpanToVector(column_shapes[0].dimensions());
  if (forward_scan) {
    exclusive_slice_limits[rank - 1] = num_columns;
  } else {
    exclusive_slice_starts[rank - 1] = 1;
    exclusive_slice_limits[rank - 1] = num_columns + 1;
  }
  std::vector<HloInstruction*> inner_scan_components;
  ShapeUtil::ForEachSubshape(
      inner_reduce_window->shape(),
      [&](const Shape& subshape, const ShapeIndex& shape_index) {
        if (!ShapeUtil::IsLeafIndex(inner_reduce_window->shape(),
                                    shape_index)) {
          return;
        }
        size_t idx = FlattenShapeIndex(shape_index);
        auto last_col = last_cols[idx];
        auto* inner_slice = parent->AddInstruction(HloInstruction::CreateSlice(
            last_col->shape(), GetAtIndex(inner_reduce_window, shape_index),
            exclusive_slice_starts, exclusive_slice_limits,
            std::vector<int64_t>(rank, 1)));
        std::vector<int64_t> rank_iota(rank);
        absl::c_iota(rank_iota, 0);
        auto* inner_scan_component =
            parent->AddInstruction(HloInstruction::CreateBroadcast(
                tiled_shapes[idx], inner_slice, rank_iota));
        inner_scan_components.push_back(inner_scan_component);
      });
  std::vector<HloInstruction*> map_operands;
  ShapeUtil::ForEachSubshape(
      outer_reduce_window->shape(),
      [&](const Shape& subshape, const ShapeIndex& shape_index) {
        if (!ShapeUtil::IsLeafIndex(outer_reduce_window->shape(),
                                    shape_index)) {
          return;
        }
        map_operands.push_back(GetAtIndex(outer_reduce_window, shape_index));
      });
  map_operands.insert(map_operands.end(), inner_scan_components.begin(),
                      inner_scan_components.end());
  std::vector<HloInstruction*> scans;
  auto status = ShapeUtil::ForEachSubshapeWithStatus(
      outer_reduce_window->shape(),
      [&](const Shape& subshape,
          const ShapeIndex& shape_index) -> absl::Status {
        if (!ShapeUtil::IsLeafIndex(outer_reduce_window->shape(),
                                    shape_index)) {
          return absl::OkStatus();
        }
        size_t idx = FlattenShapeIndex(shape_index);
        auto source = sources[idx];
        HloComputation* map_computation;
        auto reduce_function_root =
            reduce_window->to_apply()->root_instruction();
        if (reduce_function_root->shape().IsTuple()) {
          TF_RET_CHECK(reduce_function_root->opcode() == HloOpcode::kTuple);
          auto* map_computation_root = reduce_function_root->operand(idx);
          absl::flat_hash_map<const HloInstruction*,
                              std::unique_ptr<HloInstruction>>
              replacements;
          replacements[reduce_function_root] = nullptr;
          map_computation = parent->parent()->AddEmbeddedComputation(
              reduce_window->to_apply()->CloneWithReplacements(
                  &replacements,
                  {}, nullptr, "clone",
                  map_computation_root));
        } else {
          map_computation = reduce_window->to_apply();
        }
        auto scan = parent->AddInstruction(HloInstruction::CreateMap(
            ShapeAtIndex(outer_reduce_window->shape(), shape_index),
            map_operands, map_computation));
        scan = parent->AddInstruction(
            HloInstruction::CreateReshape(source->shape(), scan));
        if (scan_dim_num != rank - 1) {
          scan = parent->AddInstruction(HloInstruction::CreateTranspose(
              ShapeUtil::PermuteDimensions(permutation, source->shape()), scan,
              permutation));
        }
        if (padded_length != scan_length) {
          scan = parent->AddInstruction(HloInstruction::CreateSlice(
              operand_shape, scan, std::vector<int64_t>(rank, 0),
              operand_shape.dimensions(), std::vector<int64_t>(rank, 1)));
        }
        if (is_exclusive) {
          auto padding_config = MakeNoPaddingConfig(rank);
          if (forward_scan) {
            padding_config.mutable_dimensions(scan_dim_num)
                ->set_edge_padding_low(1);
          } else {
            padding_config.mutable_dimensions(scan_dim_num)
                ->set_edge_padding_high(1);
          }
          scan = parent->AddInstruction(HloInstruction::CreatePad(
              ShapeAtIndex(reduce_window->shape(), shape_index), scan,
              init_values[idx], padding_config));
        }
        scans.push_back(scan);
        return absl::OkStatus();
      });
  TF_RETURN_IF_ERROR(status);
  HloInstruction* scan;
  if (reduce_window->shape().IsTuple()) {
    scan = parent->AddInstruction(HloInstruction::CreateTuple(scans));
  } else {
    CHECK_EQ(scans.size(), 1);
    scan = scans[0];
  }
  TF_RETURN_IF_ERROR(reduce_window->ReplaceAllUsesWith(scan));
  TF_RETURN_IF_ERROR(parent->RemoveInstruction(reduce_window));
  return true;
}
absl::StatusOr<bool> ReduceWindowRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (const auto& computation : module->computations(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      HloReduceWindowInstruction* reduce_window =
          DynCast<HloReduceWindowInstruction>(instruction);
      if (!reduce_window) {
        continue;
      }
      TF_ASSIGN_OR_RETURN(bool made_change,
                          TryOptimizeCumSumOrProd(reduce_window));
      if (made_change) {
        changed = true;
        continue;
      }
      if (reduce_window->inputs().front()->shape().rank() != 1) {
        continue;
      }
      TF_RETURN_IF_ERROR(ReplaceReduceWindowWithReshape(reduce_window));
      changed = true;
    }
  }
  return changed;
}
}  