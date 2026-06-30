#include "xla/service/hlo_cost_analysis.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/lib/gtl/map_util.h"
#include "xla/util.h"
#include "xla/window_util.h"
#include "tsl/platform/errors.h"
namespace xla {
HloCostAnalysis::HloCostAnalysis(const Options& options) : options_(options) {}
HloCostAnalysis::HloCostAnalysis(ShapeSizeFunction shape_size,
                                 const Properties& per_second_rates,
                                 const Properties& min_latencies_seconds)
    : HloCostAnalysis(
          Options{shape_size, per_second_rates, min_latencies_seconds}) {}
absl::Status HloCostAnalysis::Preprocess(const HloInstruction* hlo) {
  current_properties_ = Properties();
  current_should_compute_bottleneck_time_ = true;
  float bytes_accessed = GetShapeSize(hlo->shape());
  current_properties_.set_output_bytes_accessed(GetShapeSize(hlo->shape()));
  for (int64_t i = 0; i < hlo->operand_count(); ++i) {
    const HloInstruction* operand = hlo->operand(i);
    bytes_accessed += GetShapeSize(operand->shape());
    current_properties_.set_operand_bytes_accessed(
        i, GetShapeSize(operand->shape()));
    current_properties_.set_operand_utilization(i, 1.0);
  }
  current_properties_[kBytesAccessedKey] = bytes_accessed;
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::Postprocess(const HloInstruction* hlo) {
  if (current_should_compute_bottleneck_time_) {
    float optimal_seconds = 0.0f;
    current_properties_.ForEach([&](absl::string_view key, float val) {
      if (key == kOptimalSecondsKey) {
        return;
      }
      float per_second_rate = options_.per_second_rate(key);
      if (per_second_rate != 0) {
        float time_for_key =
            std::max(val / per_second_rate, options_.min_latency_seconds(key));
        optimal_seconds = std::max(optimal_seconds, time_for_key);
      }
    });
    current_properties_[kOptimalSecondsKey] = optimal_seconds;
  }
  current_properties_.ForEach(
      [&](absl::string_view key, float val) { properties_sum_[key] += val; });
  auto [it_ignored, inserted] =
      hlo_properties_.emplace(hlo, std::move(current_properties_));
  current_properties_ = Properties();
  TF_RET_CHECK(inserted);
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::RemoveInstruction(HloInstruction* instruction) {
  auto it = hlo_properties_.find(instruction);
  if (it != hlo_properties_.end()) {
    current_properties_ = it->second;
    current_properties_.ForEach(
        [&](absl::string_view key, float val) { properties_sum_[key] -= val; });
    hlo_properties_.erase(instruction);
  }
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::RevisitInstruction(HloInstruction* instruction) {
  TF_RETURN_IF_ERROR(RemoveInstruction(instruction));
  TF_RETURN_IF_ERROR(Preprocess(instruction));
  TF_RETURN_IF_ERROR(instruction->Visit(this));
  TF_RETURN_IF_ERROR(Postprocess(instruction));
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleElementwiseOp(
    const HloInstruction* hlo_instruction) {
  const auto& shape = hlo_instruction->shape();
  auto computation_count = ShapeUtil::ElementsIn(shape);
  auto opcode = hlo_instruction->opcode();
  if (opcode == HloOpcode::kErf || opcode == HloOpcode::kExp ||
      opcode == HloOpcode::kLog || opcode == HloOpcode::kLogistic ||
      opcode == HloOpcode::kPower || opcode == HloOpcode::kSqrt ||
      opcode == HloOpcode::kCbrt || opcode == HloOpcode::kRsqrt ||
      opcode == HloOpcode::kTanh || opcode == HloOpcode::kSin ||
      opcode == HloOpcode::kCos || opcode == HloOpcode::kExpm1 ||
      opcode == HloOpcode::kLog1p || opcode == HloOpcode::kAtan2 ||
      opcode == HloOpcode::kTan) {
    current_properties_[kTranscendentalsKey] = computation_count;
  } else {
    current_properties_[kFlopsKey] = computation_count;
  }
  return absl::OkStatus();
}
 float HloCostAnalysis::GetPropertyForHlo(
    const HloInstruction& hlo, absl::string_view key,
    const HloToProperties& hlo_to_properties) {
  auto it = hlo_to_properties.find(&hlo);
  if (it == hlo_to_properties.end()) {
    return 0.0f;
  }
  return it->second[key];
}
int64_t HloCostAnalysis::GetShapeSize(const Shape& shape) const {
  if (!LayoutUtil::HasLayout(shape)) {
    return 0;
  }
  if (LayoutUtil::IsSparseArray(shape)) {
    return 0;
  }
  return options_.shape_size(shape);
}
int64_t HloCostAnalysis::FusionParameterReadBytes(
    const HloInstruction* hlo) const {
  CHECK(hlo->IsFused() && (hlo->opcode() == HloOpcode::kParameter ||
                           hlo->opcode() == HloOpcode::kGetTupleElement));
  auto handle_slice = [this](const HloInstruction* hlo,
                             const HloInstruction* user) -> int64_t {
    return GetShapeSize(user->shape());
  };
  auto handle_dynamic_slice = [this](const HloInstruction* hlo,
                                     const HloInstruction* user,
                                     bool& seen_trivial_user) -> int64_t {
    if (hlo == user->operand(0)) {
      return GetShapeSize(user->shape());
    }
    if (!seen_trivial_user) {
      seen_trivial_user = true;
      return GetShapeSize(hlo->shape());
    }
    return 0;
  };
  auto handle_dynamic_update_slice =
      [this](const HloInstruction* hlo, const HloInstruction* user,
             bool& seen_trivial_user) -> int64_t {
    if (hlo != user->operand(0) && !seen_trivial_user) {
      seen_trivial_user = true;
      return GetShapeSize(hlo->shape());
    }
    return 0;
  };
  int64_t size = 0;
  bool seen_trivial_user = false;
  for (const HloInstruction* user : hlo->users()) {
    switch (user->opcode()) {
      case HloOpcode::kFusion: {
        for (int64_t idx : user->OperandIndices(hlo)) {
          bool nested_seen_trivial_user = false;
          const auto& fusion_users = user->users();
          const HloInstruction* root_instruction =
              user->fused_instructions_computation()->root_instruction();
          const bool fusion_is_simple =
              user->fused_parameter(idx) == root_instruction->operand(0);
          for (const HloInstruction* fusion_user : fusion_users) {
            if (fusion_is_simple &&
                fusion_user->opcode() == HloOpcode::kSlice) {
              size += handle_slice(user, fusion_user);
            } else if (fusion_is_simple &&
                       fusion_user->opcode() == HloOpcode::kDynamicSlice) {
              size += handle_dynamic_slice(user, fusion_user,
                                           nested_seen_trivial_user);
            } else if (fusion_is_simple && fusion_user->opcode() ==
                                               HloOpcode::kDynamicUpdateSlice) {
              size += handle_dynamic_update_slice(user, fusion_user,
                                                  nested_seen_trivial_user);
            } else if (!nested_seen_trivial_user) {
              nested_seen_trivial_user = true;
              size += FusionParameterReadBytes(user->fused_parameter(idx));
            }
          }
        }
        break;
      }
      case HloOpcode::kSlice:
        size += handle_slice(hlo, user);
        break;
      case HloOpcode::kDynamicSlice:
        size += handle_dynamic_slice(hlo, user, seen_trivial_user);
        break;
      case HloOpcode::kDynamicUpdateSlice:
        size += handle_dynamic_update_slice(hlo, user, seen_trivial_user);
        break;
      case HloOpcode::kBroadcast:
      case HloOpcode::kReshape:
        size += GetShapeSize(hlo->shape());
        break;
      default:
        if (!seen_trivial_user) {
          seen_trivial_user = true;
          size += GetShapeSize(hlo->shape());
        }
    }
  }
  return size;
}
absl::Status HloCostAnalysis::FusionCalculateUtilizations(
    const HloInstruction* fusion) {
  for (const HloInstruction* instr :
       fusion->fused_instructions_computation()->instructions()) {
    if (ShouldFilterFusionInstruction(fusion, instr)) {
      hlo_properties_[instr][kUtilizationKey] = 0.f;
    } else {
      hlo_properties_[instr][kUtilizationKey] = 1.f;
    }
  }
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleElementwiseUnary(
    const HloInstruction* hlo) {
  return HandleElementwiseOp(hlo);
}
absl::Status HloCostAnalysis::HandleElementwiseBinary(
    const HloInstruction* hlo) {
  return HandleElementwiseOp(hlo);
}
absl::Status HloCostAnalysis::HandleCompare(const HloInstruction* compare) {
  return HandleElementwiseOp(compare);
}
absl::Status HloCostAnalysis::HandleClamp(const HloInstruction* clamp) {
  return HandleElementwiseOp(clamp);
}
absl::Status HloCostAnalysis::HandleReducePrecision(const HloInstruction* hlo) {
  return HandleElementwiseOp(hlo);
}
absl::Status HloCostAnalysis::HandleParameter(const HloInstruction*) {
  current_should_compute_bottleneck_time_ = false;
  current_properties_[kBytesAccessedKey] = 0;
  current_properties_.set_output_bytes_accessed(0);
  current_properties_[kOptimalSecondsKey] = 0;
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleConstant(const HloInstruction*) {
  current_should_compute_bottleneck_time_ = false;
  current_properties_[kBytesAccessedKey] = 0;
  current_properties_.set_output_bytes_accessed(0);
  current_properties_[kOptimalSecondsKey] = 0;
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleIota(const HloInstruction*) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleGetTupleElement(
    const HloInstruction* get_tuple_element) {
  current_should_compute_bottleneck_time_ = false;
  current_properties_[kBytesAccessedKey] = 0;
  current_properties_.set_output_bytes_accessed(0);
  current_properties_.set_operand_bytes_accessed(0, 0);
  current_properties_[kOptimalSecondsKey] = 0;
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleSelect(const HloInstruction* hlo) {
  return HandleElementwiseOp(hlo);
}
absl::Status HloCostAnalysis::HandleReverse(const HloInstruction*) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleSlice(const HloInstruction* slice) {
  const int64_t output_shape_size = GetShapeSize(slice->shape());
  const int64_t num_input_elements =
      ShapeUtil::ElementsIn(slice->operand(0)->shape());
  const int64_t num_output_elements = ShapeUtil::ElementsIn(slice->shape());
  current_properties_[kBytesAccessedKey] = output_shape_size * 2;
  current_properties_.set_output_bytes_accessed(output_shape_size);
  current_properties_.set_operand_bytes_accessed(0, output_shape_size);
  current_properties_.set_operand_utilization(
      0, 1.0 * num_output_elements / num_input_elements);
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleDynamicSlice(
    const HloInstruction* dynamic_slice) {
  const int64_t output_shape_size = GetShapeSize(dynamic_slice->shape());
  const int64_t start_indices_shape_size =
      GetShapeSize(dynamic_slice->operand(1)->shape());
  const int64_t num_input_elements =
      ShapeUtil::ElementsIn(dynamic_slice->operand(0)->shape());
  const int64_t num_output_elements =
      ShapeUtil::ElementsIn(dynamic_slice->shape());
  current_properties_[kBytesAccessedKey] =
      output_shape_size * 2 + start_indices_shape_size;
  current_properties_.set_output_bytes_accessed(output_shape_size);
  current_properties_.set_operand_bytes_accessed(0, output_shape_size);
  current_properties_.set_operand_bytes_accessed(1, start_indices_shape_size);
  current_properties_.set_operand_utilization(
      0, 1.0 * num_output_elements / num_input_elements);
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleDynamicUpdateSlice(
    const HloInstruction* dynamic_update_slice) {
  const int64_t update_shape_size =
      GetShapeSize(dynamic_update_slice->operand(1)->shape());
  const int64_t start_indices_shape_size =
      GetShapeSize(dynamic_update_slice->operand(2)->shape());
  const int64_t num_update_elements =
      ShapeUtil::ElementsIn(dynamic_update_slice->operand(1)->shape());
  const int64_t num_output_elements =
      ShapeUtil::ElementsIn(dynamic_update_slice->shape());
  current_properties_[kBytesAccessedKey] =
      update_shape_size * 2 + start_indices_shape_size;
  current_properties_.set_output_bytes_accessed(update_shape_size);
  current_properties_.set_operand_bytes_accessed(0, 0);
  current_properties_.set_operand_bytes_accessed(1, update_shape_size);
  current_properties_.set_operand_bytes_accessed(2, start_indices_shape_size);
  current_properties_.set_operand_utilization(
      0,
      1.0 * (num_output_elements - num_update_elements) / num_output_elements);
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleTuple(const HloInstruction* tuple) {
  current_properties_[kBytesAccessedKey] = GetShapeSize(tuple->shape());
  current_properties_.set_output_bytes_accessed(GetShapeSize(tuple->shape()));
  for (int i = 0; i < tuple->operand_count(); ++i) {
    current_properties_.set_operand_bytes_accessed(i, 0);
  }
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleConcatenate(const HloInstruction*) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleConvert(const HloInstruction* convert) {
  return HandleElementwiseOp(convert);
}
absl::Status HloCostAnalysis::HandleCopy(const HloInstruction*) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleDomain(const HloInstruction* domain) {
  current_should_compute_bottleneck_time_ = false;
  current_properties_[kBytesAccessedKey] = 0;
  current_properties_.set_output_bytes_accessed(0);
  for (int i = 0; i < domain->operand_count(); ++i) {
    current_properties_.set_operand_bytes_accessed(i, 0);
  }
  current_properties_[kOptimalSecondsKey] = 0;
  return absl::OkStatus();
}
int64_t HloCostAnalysis::GetDotFlops(const Shape& lhs_shape,
                                     const Shape& result_shape,
                                     const DotDimensionNumbers& dnums) {
  int64_t reduction_width = 1;
  for (auto dim : dnums.lhs_contracting_dimensions()) {
    reduction_width *= lhs_shape.dimensions(dim);
  }
  return kFmaFlops * ShapeUtil::ElementsIn(result_shape) * reduction_width;
}
absl::Status HloCostAnalysis::HandleDot(const HloInstruction* dot) {
  current_properties_[kFlopsKey] = GetDotFlops(
      dot->operand(0)->shape(), dot->shape(), dot->dot_dimension_numbers());
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleInfeed(const HloInstruction* infeed) {
  int64_t size = 0;
  ShapeUtil::ForEachLeafShape(
      infeed->shape(), [&](const Shape& sub_shape, const ShapeIndex& index) {
        size += GetShapeSize(sub_shape);
        current_properties_.set_output_bytes_accessed(index,
                                                      GetShapeSize(sub_shape));
      });
  current_properties_.set_output_bytes_accessed(size);
  current_properties_[kBytesAccessedKey] = size;
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleOutfeed(const HloInstruction* outfeed) {
  current_properties_[kBytesAccessedKey] = 0;
  for (int64_t i = 0; i < outfeed->operand_count(); ++i) {
    const HloInstruction* operand = outfeed->operand(i);
    int64_t size = 0;
    ShapeUtil::ForEachLeafShape(
        operand->shape(), [&](const Shape& sub_shape, const ShapeIndex& index) {
          size += GetShapeSize(sub_shape);
          current_properties_.set_operand_bytes_accessed(
              i, index, GetShapeSize(sub_shape));
        });
    current_properties_.set_operand_bytes_accessed(i, size);
    current_properties_[kBytesAccessedKey] += size;
  }
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleMap(const HloInstruction* map) {
  TF_ASSIGN_OR_RETURN(const Properties sub_properties,
                      ProcessSubcomputation(map->to_apply()));
  const int64_t element_count = ShapeUtil::ElementsIn(map->shape());
  sub_properties.ForEach([&](absl::string_view key, float val) {
    if (KeyToCopyFromSubcomputation(key)) {
      current_properties_[key] = val * element_count;
    }
  });
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleReduce(const HloInstruction* reduce) {
  HloComputation* function = reduce->to_apply();
  TF_ASSIGN_OR_RETURN(const Properties sub_properties,
                      ProcessSubcomputation(function));
  auto arg = reduce->operand(0);
  auto output_shape = reduce->shape().IsArray()
                          ? reduce->shape()
                          : reduce->shape().tuple_shapes(0);
  int64_t reduction_count =
      ShapeUtil::ElementsIn(arg->shape()) - ShapeUtil::ElementsIn(output_shape);
  sub_properties.ForEach([&](absl::string_view key, float val) {
    if (KeyToCopyFromSubcomputation(key)) {
      current_properties_[key] = val * reduction_count;
    }
  });
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleReduceWindow(
    const HloInstruction* reduce_window) {
  const Window& window = reduce_window->window();
  auto function = reduce_window->to_apply();
  TF_ASSIGN_OR_RETURN(Properties sub_properties,
                      ProcessSubcomputation(function));
  int64_t window_element_count = 1;
  for (const auto& dimension : window.dimensions()) {
    window_element_count *= dimension.size();
  }
  const int64_t input_element_count =
      ShapeUtil::ElementsIn(reduce_window->operand(0)->shape());
  const int64_t output_element_count =
      ShapeUtil::ElementsIn(reduce_window->shape().IsArray()
                                ? reduce_window->shape()
                                : reduce_window->shape().tuple_shapes(0));
  int64_t reduction_count = (window_element_count - 1) * output_element_count;
  bool optimized_rw = false;
  int64_t logical_reduction_dim = -1;
  int64_t num_reduction_dimensions = absl::c_count_if(
      window.dimensions(),
      [](const WindowDimension& dim) { return (dim.size() != 1); });
  int64_t num_padded_dimensions =
      absl::c_count_if(window.dimensions(), [](const WindowDimension& dim) {
        return (dim.padding_low() != 0 || dim.padding_high() != 0);
      });
  if (num_reduction_dimensions == 1 && num_padded_dimensions == 1 &&
      reduce_window->shape().IsArray()) {
    auto reduction_dim =
        absl::c_find_if(window.dimensions(), [](const WindowDimension& dim) {
          return (dim.size() != 1 && dim.padding_low() != 0 &&
                  dim.padding_high() != 0 &&
                  dim.padding_low() == dim.padding_high() &&
                  dim.size() == 2 * dim.padding_low() + 1);
        });
    if (reduction_dim != window.dimensions().end()) {
      logical_reduction_dim = reduction_dim - window.dimensions().begin();
      optimized_rw =
          reduction_dim->padding_low() ==
          reduce_window->shape().dimensions(logical_reduction_dim) - 1;
    }
  }
  if (optimized_rw) {
    window_element_count =
        reduce_window->shape().dimensions(logical_reduction_dim);
    reduction_count = (output_element_count / window_element_count) +
                      (window_element_count - 1);
    VLOG(3) << "Reduction count: " << reduction_count
            << " reported for reduce-window:\n"
            << reduce_window->ToString();
  }
  if (options_.count_multiple_input_accesses) {
    current_properties_.set_operand_utilization(0, 1.0 * output_element_count *
                                                       window_element_count /
                                                       input_element_count);
    current_properties_.set_operand_bytes_accessed(
        0, output_element_count * window_element_count *
               ShapeUtil::ByteSizeOfPrimitiveType(
                   reduce_window->operand(0)->shape().element_type()));
  }
  sub_properties.ForEach([&](absl::string_view key, float val) {
    if (KeyToCopyFromSubcomputation(key)) {
      current_properties_[key] = val * reduction_count;
    }
  });
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleSelectAndScatter(
    const HloInstruction* instruction) {
  TF_ASSIGN_OR_RETURN(Properties select_properties,
                      ProcessSubcomputation(instruction->select()));
  TF_ASSIGN_OR_RETURN(Properties scatter_properties,
                      ProcessSubcomputation(instruction->scatter()));
  const auto source = instruction->operand(1);
  const auto source_element_count = ShapeUtil::ElementsIn(source->shape());
  int64_t window_element_count = 1;
  for (const auto& dimension : instruction->window().dimensions()) {
    window_element_count *= dimension.size();
  }
  const int64_t select_count =
      source_element_count * (window_element_count - 1);
  select_properties.ForEach([&](absl::string_view key, float val) {
    if (KeyToCopyFromSubcomputation(key)) {
      current_properties_[key] += val * select_count;
    }
  });
  scatter_properties.ForEach([&](absl::string_view key, float val) {
    if (KeyToCopyFromSubcomputation(key)) {
      current_properties_[key] += val * source_element_count;
    }
  });
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleBitcast(const HloInstruction*) {
  current_properties_[kBytesAccessedKey] = 0;
  current_properties_.set_output_bytes_accessed(0);
  current_properties_.set_operand_bytes_accessed(0, 0);
  current_properties_[kOptimalSecondsKey] = 0;
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleBroadcast(const HloInstruction* broadcast) {
  if (options_.count_multiple_input_accesses) {
    current_properties_.set_operand_bytes_accessed(
        0, GetShapeSize(broadcast->shape()));
    current_properties_.set_operand_utilization(
        0, 1.0 * ShapeUtil::ElementsIn(broadcast->shape()) /
               ShapeUtil::ElementsIn(broadcast->operand(0)->shape()));
  }
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandlePad(const HloInstruction*) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleAsyncStart(
    const HloInstruction* async_start) {
  TF_ASSIGN_OR_RETURN(
      current_properties_,
      ProcessSubcomputation(async_start->called_computations()[0]));
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleAsyncUpdate(const HloInstruction*) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleAsyncDone(const HloInstruction*) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleCopyStart(const HloInstruction*) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleCopyDone(const HloInstruction*) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleSend(const HloInstruction*) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleSendDone(const HloInstruction*) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleRecv(const HloInstruction*) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleRecvDone(const HloInstruction*) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleReshape(const HloInstruction*) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleDynamicReshape(const HloInstruction*) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleBatchNormTraining(const HloInstruction*) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleBatchNormInference(const HloInstruction*) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleBatchNormGrad(const HloInstruction*) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleTranspose(const HloInstruction* transpose) {
  if (transpose->IsEffectiveBitcast()) {
    return HandleBitcast(transpose);
  }
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleAfterAll(const HloInstruction* token) {
  current_should_compute_bottleneck_time_ = false;
  current_properties_[kBytesAccessedKey] = 0;
  current_properties_.set_output_bytes_accessed(0);
  for (int i = 0; i < token->operand_count(); ++i) {
    current_properties_.set_operand_bytes_accessed(i, 0);
  }
  current_properties_[kOptimalSecondsKey] = 0;
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleAddDependency(
    const HloInstruction* add_dependency) {
  current_should_compute_bottleneck_time_ = false;
  current_properties_[kBytesAccessedKey] = 0;
  current_properties_.set_output_bytes_accessed(0);
  for (int i = 0; i < add_dependency->operand_count(); ++i) {
    current_properties_.set_operand_bytes_accessed(i, 0);
  }
  current_properties_[kOptimalSecondsKey] = 0;
  return absl::OkStatus();
}
int64_t HloCostAnalysis::GetConvolutionFlops(
    const HloInstruction* convolution) {
  auto lhs = convolution->operand(0);
  auto rhs = convolution->operand(1);
  const Shape& lhs_shape = lhs->shape();
  const Shape& rhs_shape = rhs->shape();
  const Shape& result_shape = convolution->shape();
  return GetConvolutionFlops(convolution, lhs_shape, rhs_shape, result_shape);
}
int64_t HloCostAnalysis::GetConvolutionFlops(const HloInstruction* convolution,
                                             const Shape& lhs_shape,
                                             const Shape& rhs_shape,
                                             const Shape& result_shape) {
  Window window = convolution->window();
  const auto& dnums = convolution->convolution_dimension_numbers();
  const int64_t input_batch_dim = dnums.input_batch_dimension();
  const int64_t input_feature_dim = dnums.input_feature_dimension();
  const int64_t output_feature_dim = dnums.output_feature_dimension();
  const int64_t input_feature =
      ShapeUtil::GetDimension(lhs_shape, input_feature_dim);
  const int64_t output_feature =
      ShapeUtil::GetDimension(result_shape, output_feature_dim);
  const int64_t batch = ShapeUtil::GetDimension(lhs_shape, input_batch_dim);
  DimensionVector kernel_limits;
  DimensionVector output_limits;
  DimensionVector input_limits;
  if (window.dimensions().empty()) {
    window = window_util::MakeWindow({1});
    kernel_limits.push_back(1);
    output_limits.push_back(1);
    input_limits.push_back(1);
  } else {
    for (int64_t spatial_dimension = 0;
         spatial_dimension < window.dimensions_size(); ++spatial_dimension) {
      const int64_t kernel_spatial_dim =
          dnums.kernel_spatial_dimensions(spatial_dimension);
      const int64_t kernel_limit = rhs_shape.dimensions(kernel_spatial_dim);
      kernel_limits.push_back(kernel_limit);
      const int64_t output_spatial_dim =
          dnums.output_spatial_dimensions(spatial_dimension);
      const int64_t output_limit = result_shape.dimensions(output_spatial_dim);
      output_limits.push_back(output_limit);
      const int64_t input_spatial_dim =
          dnums.input_spatial_dimensions(spatial_dimension);
      const int64_t input_limit = lhs_shape.dimensions(input_spatial_dim);
      input_limits.push_back(input_limit);
    }
  }
  DimensionVector valid_position_counts;
  for (int64_t spatial_dimension = 0;
       spatial_dimension < window.dimensions_size(); ++spatial_dimension) {
    const auto& window_dim = window.dimensions(spatial_dimension);
    if (input_limits[spatial_dimension] == output_limits[spatial_dimension] &&
        kernel_limits[spatial_dimension] == output_limits[spatial_dimension] &&
        input_limits[spatial_dimension] == window_dim.base_dilation() &&
        window_dim.window_dilation() == 1 &&
        std::max<int64_t>(1, input_limits[spatial_dimension] - 1) ==
            window_dim.stride() &&
        window_dim.padding_low() == 0 && window_dim.padding_high() == 0) {
      valid_position_counts.push_back(input_limits[spatial_dimension]);
      continue;
    }
    if (input_limits[spatial_dimension] == 1 &&
        kernel_limits[spatial_dimension] == output_limits[spatial_dimension] &&
        window_dim.window_dilation() == 1 && window_dim.base_dilation() == 1 &&
        window_dim.stride() == 1 &&
        window_dim.padding_high() == output_limits[spatial_dimension] - 1 &&
        window_dim.padding_low() == output_limits[spatial_dimension] - 1) {
      valid_position_counts.push_back(output_limits[spatial_dimension]);
      continue;
    }
    int64_t valid_position_count = 0;
    for (int64_t kernel_idx = 0; kernel_idx < kernel_limits[spatial_dimension];
         ++kernel_idx) {
      if (window_dim.stride() == 1 && window_dim.base_dilation() == 1) {
        const int64_t undilated_index_base =
            window_dim.padding_low() -
            kernel_idx * window_dim.window_dilation();
        valid_position_count += std::max<int64_t>(
            std::min<int64_t>(
                input_limits[spatial_dimension] + undilated_index_base,
                output_limits[spatial_dimension]) -
                std::max<int64_t>(undilated_index_base, int64_t{0}),
            int64_t{0});
        continue;
      }
      for (int64_t output_idx = 0;
           output_idx < output_limits[spatial_dimension]; ++output_idx) {
        const int64_t undilated_index =
            output_idx * window_dim.stride() - window_dim.padding_low() +
            kernel_idx * window_dim.window_dilation();
        const int64_t lhs_spatial_index =
            window_dim.base_dilation() > 1
                ? undilated_index / window_dim.base_dilation()
                : undilated_index;
        if (undilated_index != lhs_spatial_index * window_dim.base_dilation()) {
          continue;
        }
        if (lhs_spatial_index < 0 ||
            lhs_spatial_index >= input_limits[spatial_dimension]) {
          continue;
        }
        valid_position_count += 1;
      }
    }
    valid_position_counts.push_back(valid_position_count);
  }
  const int64_t fma_count =
      (input_feature / convolution->feature_group_count()) * output_feature *
      (batch / convolution->batch_group_count()) *
      Product(valid_position_counts);
  return fma_count * kFmaFlops;
}
absl::Status HloCostAnalysis::HandleConvolution(
    const HloInstruction* convolution) {
  current_properties_[kFlopsKey] = GetConvolutionFlops(convolution);
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleFft(const HloInstruction* fft) {
  auto real_shape =
      fft->operand(0)->shape().IsTuple()
          ? ShapeUtil::GetTupleElementShape(fft->operand(0)->shape(), 0)
          : fft->operand(0)->shape();
  constexpr int kFmaPerComplexMul = 4;
  int64_t log_factors = 1;
  for (int64_t dim : fft->fft_length()) {
    log_factors *= Log2Floor<uint64_t>(dim);
  }
  current_properties_[kFlopsKey] = kFmaFlops * kFmaPerComplexMul * log_factors *
                                   ShapeUtil::ElementsIn(real_shape);
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleTriangularSolve(const HloInstruction* hlo) {
  float bytes_accessed = GetShapeSize(hlo->shape());
  current_properties_.set_output_bytes_accessed(GetShapeSize(hlo->shape()));
  bytes_accessed += GetShapeSize(hlo->operand(0)->shape()) / 2.0f;
  current_properties_.set_operand_bytes_accessed(
      0, GetShapeSize(hlo->operand(0)->shape()) / 2.0f);
  bytes_accessed += GetShapeSize(hlo->operand(1)->shape());
  current_properties_.set_operand_bytes_accessed(
      0, GetShapeSize(hlo->operand(1)->shape()));
  current_properties_[kBytesAccessedKey] = bytes_accessed;
  const Shape& a_shape = hlo->operand(0)->shape();
  const Shape& b_shape = hlo->operand(1)->shape();
  int64_t elems = a_shape.dimensions(a_shape.dimensions_size() - 1);
  elems *= ShapeUtil::ElementsIn(b_shape);
  current_properties_[kFlopsKey] = kFmaFlops * elems;
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleCholesky(const HloInstruction* hlo) {
  float bytes_accessed = GetShapeSize(hlo->operand(0)->shape()) / 2.0f;
  current_properties_.set_output_bytes_accessed(
      GetShapeSize(hlo->operand(0)->shape()) / 2.0f);
  bytes_accessed += GetShapeSize(hlo->operand(0)->shape()) / 2.0f;
  current_properties_.set_operand_bytes_accessed(
      0, GetShapeSize(hlo->operand(0)->shape()) / 2.0f);
  current_properties_[kBytesAccessedKey] = bytes_accessed;
  const Shape& a_shape = hlo->operand(0)->shape();
  int64_t elems = a_shape.dimensions(a_shape.dimensions_size() - 1);
  elems *= ShapeUtil::ElementsIn(a_shape);
  current_properties_[kFlopsKey] = elems / 3;
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleOptimizationBarrier(
    const HloInstruction* ) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleAllGather(const HloInstruction* ) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleAllGatherStart(const HloInstruction* hlo) {
  return HandleAllGather(hlo);
}
absl::Status HloCostAnalysis::HandleAllGatherDone(
    const HloInstruction* ) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleAllReduce(const HloInstruction* crs) {
  double flops = 0.0;
  int64_t output_bytes_accessed = 0;
  ShapeUtil::ForEachSubshape(
      crs->shape(), [&](const Shape& subshape, const ShapeIndex&) {
        if (subshape.IsArray()) {
          flops += ShapeUtil::ElementsIn(subshape);
          output_bytes_accessed += GetShapeSize(subshape);
        }
      });
  int64_t bytes_accessed = output_bytes_accessed;
  for (const HloInstruction* operand : crs->operands()) {
    bytes_accessed += GetShapeSize(operand->shape());
  }
  current_properties_[kFlopsKey] = flops;
  current_properties_.set_output_bytes_accessed(output_bytes_accessed);
  current_properties_[kBytesAccessedKey] = bytes_accessed;
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleReduceScatter(const HloInstruction* hlo) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleAllReduceStart(const HloInstruction* hlo) {
  return HandleAllReduce(hlo);
}
absl::Status HloCostAnalysis::HandleAllReduceDone(
    const HloInstruction* ) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleAllToAll(const HloInstruction* hlo) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleCollectiveBroadcast(
    const HloInstruction* ) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleCollectivePermute(
    const HloInstruction* ) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleCollectivePermuteStart(
    const HloInstruction* ) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleCollectivePermuteDone(
    const HloInstruction* ) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandlePartitionId(const HloInstruction* ) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleReplicaId(const HloInstruction* ) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleRng(const HloInstruction* random) {
  current_properties_[kTranscendentalsKey] =
      ShapeUtil::ElementsIn(random->shape());
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleRngBitGenerator(
    const HloInstruction* random) {
  current_properties_[kTranscendentalsKey] =
      ShapeUtil::ElementsInRecursive(random->shape());
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleRngGetAndUpdateState(
    const HloInstruction* random) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::FusionProcessOutputBytesAccessed(
    const HloInstruction* fusion) {
  ShapeUtil::ForEachSubshape(
      fusion->shape(),
      [this, fusion](const Shape& subshape, const ShapeIndex& shape_index) {
        if (!subshape.IsArray()) {
          return;
        }
        const HloInstruction* root = fusion->fused_expression_root();
        auto further_examine_index =
            shape_index.size() == 1 && root->opcode() == HloOpcode::kTuple;
        if (further_examine_index &&
            ShouldFilterFusionOutputIndex(fusion, shape_index)) {
          current_properties_.set_output_bytes_accessed(shape_index, 0);
          hlo_properties_[root->operand(shape_index[0])]
                         [GetOperandUtilizationKey(0)] = 0;
          return;
        }
        if (further_examine_index) {
          root = root->operand(shape_index[0]);
        }
        if (root->opcode() == HloOpcode::kDynamicUpdateSlice) {
          int64_t size = GetShapeSize(root->operand(1)->shape());
          current_properties_[kBytesAccessedKey] += size;
          current_properties_.set_output_bytes_accessed(shape_index, size);
          hlo_properties_[root][GetOperandUtilizationKey(0)] = 0;
          return;
        }
        current_properties_[kBytesAccessedKey] += GetShapeSize(subshape);
        current_properties_.set_output_bytes_accessed(shape_index,
                                                      GetShapeSize(subshape));
      });
  if (fusion->shape().IsTuple()) {
    std::function<float(const Shape&, const ShapeIndex&)>
        propagate_output_size_to_parent;
    propagate_output_size_to_parent = [&](const Shape& shape,
                                          const ShapeIndex& shape_index) {
      float& bytes_accessed =
          current_properties_[GetOutputBytesAccessedKey(shape_index)];
      if (bytes_accessed != 0) {
        return bytes_accessed;
      }
      for (int i = 0; i < shape.tuple_shapes_size(); ++i) {
        const Shape& subshape = shape.tuple_shapes(i);
        if (!subshape.IsTuple() && ShouldFilterFusionOutputIndex(fusion, {i})) {
          continue;
        }
        ShapeIndex subshape_index(shape_index);
        subshape_index.push_back(i);
        bytes_accessed +=
            propagate_output_size_to_parent(subshape, subshape_index);
      }
      return bytes_accessed;
    };
    current_properties_[GetOutputBytesAccessedKey()] = 0;
    propagate_output_size_to_parent(fusion->shape(), {});
  }
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::FusionProcessOperandBytesRead(
    const HloInstruction* fusion) {
  for (int64_t i = 0; i < fusion->fused_parameters().size(); ++i) {
    const HloInstruction* operand = fusion->fused_parameter(i);
    int64_t operand_size = 0;
    if (ShouldFilterFusionInput(fusion, i)) {
      current_properties_.set_operand_bytes_accessed(i, operand_size);
      current_properties_.set_operand_utilization(
          i, hlo_properties_[operand][kUtilizationKey]);
      continue;
    }
    if (!operand->shape().IsTuple()) {
      operand_size = FusionParameterReadBytes(operand);
    } else {
      ShapeUtil::ForEachLeafShape(
          operand->shape(),
          [&](const Shape& , const ShapeIndex& index) {
            const HloInstruction* gte = operand;
            for (int64_t sub_index : index) {
              for (const HloInstruction* user : gte->users()) {
                if (user->opcode() == HloOpcode::kGetTupleElement &&
                    user->tuple_index() == sub_index) {
                  gte = user;
                  break;
                }
              }
            }
            int64_t size = FusionParameterReadBytes(gte);
            operand_size += size;
            current_properties_.set_operand_bytes_accessed(i, index, size);
          });
    }
    current_properties_[kBytesAccessedKey] += operand_size;
    current_properties_.set_operand_bytes_accessed(i, operand_size);
    current_properties_.set_operand_utilization(
        i, hlo_properties_[operand][kUtilizationKey]);
  }
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::FusionCountConstantsMemoryAccess(
    const HloInstruction* fusion) {
  for (const HloInstruction* instr :
       fusion->fused_instructions_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kConstant &&
        ShapeUtil::ElementsIn(instr->shape()) >
            immediate_constant_max_elements()) {
      float utilization = hlo_properties_[instr][kUtilizationKey];
      if (!options_.count_multiple_input_accesses) {
        utilization = fmin(utilization, 1.0);
      }
      current_properties_[kBytesAccessedKey] +=
          GetShapeSize(instr->shape()) * utilization;
    }
  }
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleFusion(const HloInstruction* fusion) {
  VLOG(8) << "Processing fusion " << fusion->ToString();
  if (fusion->IsCustomFusion()) {
    for (const HloInstruction* hlo :
         fusion->fused_instructions_computation()->instructions()) {
      if (hlo->opcode() == HloOpcode::kGather) {
        return HandleGather(hlo);
      }
      if (hlo->opcode() == HloOpcode::kScatter) {
        return HandleScatter(hlo);
      }
    }
  }
  TF_ASSIGN_OR_RETURN(
      current_properties_,
      ProcessSubcomputation(fusion->fused_instructions_computation()));
  current_properties_[kBytesAccessedKey] = 0;
  TF_RETURN_IF_ERROR(FusionProcessOutputBytesAccessed(fusion));
  TF_RETURN_IF_ERROR(FusionCalculateUtilizations(fusion));
  TF_RETURN_IF_ERROR(FusionCountConstantsMemoryAccess(fusion));
  TF_RETURN_IF_ERROR(FusionProcessOperandBytesRead(fusion));
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleCall(const HloInstruction* call) {
  TF_ASSIGN_OR_RETURN(current_properties_,
                      ProcessSubcomputation(call->to_apply()));
  current_should_compute_bottleneck_time_ = false;
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleCustomCall(
    const HloInstruction* custom_call) {
  current_properties_[kOptimalSecondsKey] = -1;
  current_properties_[kBytesAccessedKey] = -1;
  current_properties_.set_output_bytes_accessed(-1);
  for (int i = 0; i < custom_call->operand_count(); ++i) {
    current_properties_.set_operand_bytes_accessed(i, -1);
  }
  current_properties_[kFlopsKey] = -1;
  current_should_compute_bottleneck_time_ = false;
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleSort(const HloInstruction* sort) {
  int64_t elements = ShapeUtil::ElementsIn(sort->operand(0)->shape());
  current_properties_[kFlopsKey] = elements * Log2Ceiling<uint64_t>(elements);
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleTopK(const HloInstruction* topk) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleWhile(const HloInstruction* xla_while) {
  TF_ASSIGN_OR_RETURN(const Properties body_properties,
                      ProcessSubcomputation(xla_while->while_body()));
  TF_ASSIGN_OR_RETURN(const Properties condition_properties,
                      ProcessSubcomputation(xla_while->while_condition()));
  current_properties_ = Properties();
  body_properties.ForEach([&](absl::string_view key, float val) {
    current_properties_[key] += val;
  });
  condition_properties.ForEach([&](absl::string_view key, float val) {
    current_properties_[key] += val;
  });
  current_should_compute_bottleneck_time_ = false;
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleConditional(
    const HloInstruction* conditional) {
  TF_ASSIGN_OR_RETURN(
      const Properties branch0_computation_properties,
      ProcessSubcomputation(conditional->branch_computation(0)));
  current_properties_ = branch0_computation_properties;
  for (int j = 1; j < conditional->branch_count(); ++j) {
    TF_ASSIGN_OR_RETURN(
        const Properties branch_computation_properties,
        ProcessSubcomputation(conditional->branch_computation(j)));
    branch_computation_properties.ForEach(
        [&](absl::string_view key, float val) {
          auto& current_property = current_properties_[key];
          current_property = std::max(current_property, val);
        });
  }
  current_should_compute_bottleneck_time_ = false;
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleGather(const HloInstruction* gather) {
  int64_t output_size = GetShapeSize(gather->shape());
  current_properties_[kBytesAccessedKey] =
      output_size * 2 + GetShapeSize(gather->operand(1)->shape());
  current_properties_.set_operand_bytes_accessed(0, output_size);
  current_properties_.set_operand_bytes_accessed(
      1, GetShapeSize(gather->operand(1)->shape()));
  current_properties_.set_operand_utilization(
      0, 1.0 * ShapeUtil::ElementsIn(gather->shape()) /
             ShapeUtil::ElementsIn(gather->operand(0)->shape()));
  current_properties_.set_output_bytes_accessed(output_size);
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleScatter(const HloInstruction* hlo) {
  auto* scatter = Cast<HloScatterInstruction>(hlo);
  int64_t total_update_size = 0;
  for (int i = 0, n = scatter->scatter_operand_count(); i < n; ++i) {
    int64_t update_size = GetShapeSize(scatter->scatter_updates()[i]->shape());
    current_properties_.set_operand_bytes_accessed(i, update_size);
    current_properties_.set_operand_bytes_accessed(n + 1 + i, update_size);
    total_update_size += update_size;
  }
  int64_t scatter_indices_size =
      GetShapeSize(scatter->scatter_indices()->shape());
  current_properties_.set_operand_bytes_accessed(
      scatter->scatter_operand_count(), scatter_indices_size);
  current_properties_[kBytesAccessedKey] =
      total_update_size * 3 + scatter_indices_size;
  current_properties_.set_output_bytes_accessed(total_update_size);
  const int64_t element_count =
      ShapeUtil::ElementsIn(scatter->scatter_updates()[0]->shape());
  TF_ASSIGN_OR_RETURN(const Properties sub_properties,
                      ProcessSubcomputation(scatter->to_apply()));
  sub_properties.ForEach([&](absl::string_view key, float val) {
    if (KeyToCopyFromSubcomputation(key)) {
      current_properties_[key] = val * element_count;
    }
  });
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleGetDimensionSize(
    const HloInstruction* ) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::HandleSetDimensionSize(
    const HloInstruction* ) {
  return absl::OkStatus();
}
absl::Status HloCostAnalysis::FinishVisit(const HloInstruction*) {
  return absl::OkStatus();
}
float HloCostAnalysis::flop_count() const { return properties_sum_[kFlopsKey]; }
float HloCostAnalysis::transcendental_count() const {
  return properties_sum_[kTranscendentalsKey];
}
float HloCostAnalysis::bytes_accessed() const {
  return properties_sum_[kBytesAccessedKey];
}
float HloCostAnalysis::optimal_seconds() const {
  return properties_sum_[kOptimalSecondsKey];
}
HloCostAnalysis::Properties HloCostAnalysis::properties(
    const HloInstruction& hlo) const {
  auto it = hlo_properties_.find(&hlo);
  if (it == hlo_properties_.end()) {
    return Properties();
  }
  return it->second;
}
int64_t HloCostAnalysis::flop_count(const HloInstruction& hlo) const {
  return GetPropertyForHlo(hlo, kFlopsKey, hlo_properties_);
}
int64_t HloCostAnalysis::transcendental_count(const HloInstruction& hlo) const {
  return GetPropertyForHlo(hlo, kTranscendentalsKey, hlo_properties_);
}
int64_t HloCostAnalysis::bytes_accessed(const HloInstruction& hlo) const {
  return GetPropertyForHlo(hlo, kBytesAccessedKey, hlo_properties_);
}
int64_t HloCostAnalysis::operand_bytes_accessed(const HloInstruction& hlo,
                                                int64_t operand_num,
                                                ShapeIndex index) const {
  return GetPropertyForHlo(hlo, GetOperandBytesAccessedKey(operand_num, index),
                           hlo_properties_);
}
float HloCostAnalysis::operand_utilization(const HloInstruction& hlo,
                                           int64_t operand_num,
                                           ShapeIndex index) const {
  return GetPropertyForHlo(hlo, GetOperandUtilizationKey(operand_num, index),
                           hlo_properties_);
}
int64_t HloCostAnalysis::output_bytes_accessed(const HloInstruction& hlo,
                                               ShapeIndex index) const {
  return GetPropertyForHlo(hlo, GetOutputBytesAccessedKey(index),
                           hlo_properties_);
}
float HloCostAnalysis::optimal_seconds(const HloInstruction& hlo) const {
  return GetPropertyForHlo(hlo, kOptimalSecondsKey, hlo_properties_);
}
int64_t HloCostAnalysis::GetBytesRead(
    const HloInstruction& hlo, std::optional<int64_t> memory_space) const {
  int64_t bytes_read = 0;
  for (int operand_number = 0; operand_number < hlo.operand_count();
       ++operand_number) {
    const Shape& shape = hlo.operand(operand_number)->shape();
    ShapeUtil::ForEachSubshape(
        shape, [&](const Shape& sub_shape, const ShapeIndex& index) {
          if (ShapeUtil::IsLeafIndex(shape, index)) {
            std::optional<int64_t> index_memory_space;
            if (sub_shape.has_layout()) {
              index_memory_space = sub_shape.layout().memory_space();
            }
            if (!memory_space || memory_space == index_memory_space) {
              bytes_read += operand_bytes_accessed(hlo, operand_number, index);
            }
          }
        });
  }
  return bytes_read;
}
int64_t HloCostAnalysis::GetBytesWritten(
    const HloInstruction& hlo, std::optional<int64_t> memory_space) const {
  int64_t bytes_written = 0;
  ShapeUtil::ForEachLeafShape(
      hlo.shape(), [&](const Shape& sub_shape, const ShapeIndex& index) {
        std::optional<int64_t> index_memory_space;
        if (sub_shape.has_layout()) {
          index_memory_space = sub_shape.layout().memory_space();
        }
        if (!memory_space || memory_space == index_memory_space) {
          bytes_written += output_bytes_accessed(hlo, index);
        }
      });
  return bytes_written;
}
absl::StatusOr<HloCostAnalysis::Properties>
HloCostAnalysis::ProcessSubcomputation(HloComputation* computation) {
  auto visitor = CreateNestedCostAnalysis();
  visitor->ReserveVisitStates(computation->instruction_count());
  TF_RETURN_IF_ERROR(computation->Accept(visitor.get()));
  for (auto& entry : visitor->hlo_properties_) {
    hlo_properties_[entry.first] = std::move(entry.second);
  }
  return visitor->properties();
}
std::unique_ptr<HloCostAnalysis> HloCostAnalysis::CreateNestedCostAnalysis() {
  return std::make_unique<HloCostAnalysis>(options_);
}
 std::string HloCostAnalysis::GetOperandBytesAccessedKey(
    int64_t operand_num, const ShapeIndex& index) {
  return absl::StrCat(kBytesAccessedKey, operand_num, index.ToString());
}
 std::string HloCostAnalysis::GetOperandUtilizationKey(
    int64_t operand_num, const ShapeIndex& index) {
  return absl::StrCat(kUtilizationKey, operand_num, index.ToString());
}
 std::string HloCostAnalysis::GetOutputBytesAccessedKey(
    const ShapeIndex& index) {
  return absl::StrCat(kBytesAccessedKey, "out", index.ToString());
}
bool HloCostAnalysis::KeyToCopyFromSubcomputation(absl::string_view key) const {
  return !absl::StartsWith(key, kBytesAccessedKey) &&
         !absl::StartsWith(key, kUtilizationKey);
}
}  