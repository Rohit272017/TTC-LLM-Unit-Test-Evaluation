#include "xla/service/gpu/triton_tiling_propagation.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <list>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/layout.h"
#include "xla/permutation_util.h"
#include "xla/service/gpu/fusions/triton/triton_support.h"
#include "xla/service/gpu/fusions/triton/triton_support_legacy.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
namespace xla {
namespace gpu {
namespace {
absl::flat_hash_map<int, TensorIterationSpec::DimIterationSpec>
FilterTrivialDims(
    const absl::flat_hash_map<int, TensorIterationSpec::DimIterationSpec>&
        dim_iter_specs) {
  absl::flat_hash_map<int, TensorIterationSpec::DimIterationSpec>
      non_trivial_dim_iteration_specs;
  for (const auto& [dim, dim_spec] : dim_iter_specs) {
    if (dim_spec.size() == 1 && dim_spec[0].count == 1) {
      continue;
    }
    non_trivial_dim_iteration_specs[dim] = dim_spec;
  }
  return non_trivial_dim_iteration_specs;
}
}  
const TensorIterationSpec::DimIterationSpec* TensorIterationSpec::Find(
    const int dimension) const {
  if (auto it = dim_iteration_specs_.find(dimension);
      it != dim_iteration_specs_.end()) {
    return &it->second;
  }
  return nullptr;
}
std::vector<int> TensorIterationSpec::GetDimensions() const {
  std::vector<int> result;
  result.reserve(dim_iteration_specs_.size());
  for (const auto& [dim, _] : dim_iteration_specs_) {
    result.push_back(dim);
  }
  return result;
}
bool TensorIterationSpec::IsPhysicallyEquivalent(
    const TensorIterationSpec& other) const {
  const absl::flat_hash_map<int, DimIterationSpec>
      non_trivial_dim_iteration_specs = FilterTrivialDims(dim_iteration_specs_);
  const absl::flat_hash_map<int, DimIterationSpec>
      other_non_trivial_dim_iteration_specs =
          FilterTrivialDims(other.dim_iteration_specs_);
  if (non_trivial_dim_iteration_specs.size() !=
      other_non_trivial_dim_iteration_specs.size()) {
    return false;
  }
  for (const auto& pair : non_trivial_dim_iteration_specs) {
    int dimension = pair.first;
    const DimIterationSpec& dim_iter_spec = pair.second;
    auto other_it = other_non_trivial_dim_iteration_specs.find(dimension);
    if (other_it == other_non_trivial_dim_iteration_specs.end()) {
      return false;
    }
    const DimIterationSpec& other_dim_iter_spec = other_it->second;
    if (dim_iter_spec.size() != other_dim_iter_spec.size()) {
      return false;
    }
    for (size_t i = 0; i < dim_iter_spec.size(); i++) {
      if (!dim_iter_spec[i].IsPhysicallyEquivalent(other_dim_iter_spec[i])) {
        return false;
      }
    }
  }
  return true;
}
std::string TensorIterationSpec::IterationSpecFragment::ToString() const {
  return absl::StrCat("{stride=", stride, ", count=", count,
                      ", slice_start=", slice_start,
                      ", sliced_count=", sliced_count, ", subfragments=[",
                      absl::StrJoin(subfragments, ", "), "]}");
}
std::string TensorIterationSpec::ToString() const {
  return absl::StrCat(
      "{",
      absl::StrJoin(dim_iteration_specs_, ", ",
                    [&](std::string* s, const auto& kv) {
                      absl::StrAppend(
                          s, kv.first, ": ", "[",
                          absl::StrJoin(kv.second, ", ",
                                        [&](std::string* ss, const auto& v) {
                                          absl::StrAppend(ss, v.ToString());
                                        }),
                          "]");
                    }),
      "}");
}
namespace triton_fusion {
using Fragment = DimensionOrder::Fragment;
using Fragments = DimensionOrder::Fragments;
using FragmentOrders = DimensionOrder::FragmentOrders;
 DimensionOrder DimensionOrder::FromDotOperandOrOutput(
    const HloInstruction& hlo, const int split_k_dimension_index) {
  DimensionOrder dim_order;
  dim_order.tensor_fragments_order_.reserve(hlo.shape().rank());
  for (const int i : hlo.shape().layout().minor_to_major()) {
    int target_dim_number = i;
    if (i == split_k_dimension_index) {
      CHECK(!dim_order.tensor_fragments_order_.empty())
          << "The split-K batch dimension has be preceded by the contracting "
             "dimension it originates from by construction.";
      target_dim_number =
          dim_order.tensor_fragments_order_.back().dst_dim_number();
    }
    dim_order.dim_fragments_orders_[target_dim_number].push_back(
        dim_order.tensor_fragments_order_.size());
    dim_order.tensor_fragments_order_.push_back(
        Fragment{target_dim_number, hlo.shape().dimensions(i)});
  }
  return dim_order;
}
std::string DimensionOrder::Fragment::ToString() const {
  return absl::StrCat(dst_dim_number_, ":", count_, ":", slice_start_, "-",
                      sliced_count_);
}
std::string DimensionOrder::ToString() const {
  std::string ret = absl::StrJoin(tensor_fragments_order_, " - ",
                                  [](std::string* out, const Fragment& f) {
                                    absl::StrAppend(out, f.ToString(), " ");
                                  });
  absl::StrAppend(&ret, "|");
  for (const auto& [dim, fragments] : dim_fragments_orders_) {
    absl::StrAppend(&ret, dim, ":", absl::StrJoin(fragments, ","), " ");
  }
  return ret;
}
TensorIterationSpec DimensionOrder::ToTensorIterationSpec() const {
  const Fragments& dim_fragments = TensorFragmentsOrder();
  TensorIterationSpec tensor_spec;
  int64_t accumulated_stride = 1;
  int last_dim = -1;
  for (int dim_order_index = 0; dim_order_index < dim_fragments.size();
       ++dim_order_index) {
    const DimensionOrder::Fragment& fragment = dim_fragments[dim_order_index];
    VLOG(6) << fragment.ToString();
    TensorIterationSpec::DimIterationSpec& dim_spec =
        tensor_spec[fragment.dst_dim_number()];
    if (last_dim == fragment.dst_dim_number()) {
      if (!dim_spec.empty() && !dim_spec.back().subfragments.empty() &&
          dim_spec.back().subfragments.back() == 1) {
        dim_spec.back().subfragments.pop_back();
      }
      if (fragment.full_count() > 1) {
        CHECK(!dim_spec.empty());
        CHECK(!dim_spec.back().is_sliced())
            << "Only the major-most fragment can have an offset.";
        dim_spec.back().slice_start =
            fragment.slice_start() * dim_spec.back().count;
        dim_spec.back().sliced_count =
            fragment.sliced_count() * dim_spec.back().count;
        dim_spec.back().count *= fragment.full_count();
        dim_spec.back().subfragments.push_back(fragment.sliced_count());
      }
    } else {
      dim_spec.push_back(TensorIterationSpec::IterationSpecFragment{
          accumulated_stride,
          fragment.full_count(),
          fragment.slice_start(),
          fragment.sliced_count(),
          {fragment.sliced_count()}});
    }
    accumulated_stride *= fragment.full_count();
    last_dim = fragment.dst_dim_number();
  }
  for (int dim_idx : tensor_spec.GetDimensions()) {
    TensorIterationSpec::DimIterationSpec& dim_spec = tensor_spec[dim_idx];
    if (dim_spec.size() <= 1) continue;
    TensorIterationSpec::DimIterationSpec filtered_dim_spec;
    absl::c_copy_if(dim_spec, std::back_inserter(filtered_dim_spec),
                    [](const TensorIterationSpec::IterationSpecFragment& f) {
                      return f.count != 1;
                    });
    tensor_spec[dim_idx] = filtered_dim_spec;
  }
  tensor_spec.RemoveEmptyDimensions();
  return tensor_spec;
}
namespace {
std::optional<int> LogicalIndexOfLabeledDimension(
    const Shape& shape, const DimensionOrder& dim_order, const int label) {
  auto fragment_it = dim_order.TensorFragmentsOrder().cbegin();
  for (int dim : shape.layout().minor_to_major()) {
    const int64_t dim_size = shape.dimensions()[dim];
    int64_t fragments_size = 1;
    while (fragments_size < dim_size) {
      fragments_size *= fragment_it->full_count();
      if (fragment_it->dst_dim_number() == label) {
        return dim;
      }
      ++fragment_it;
    }
  }
  return std::nullopt;
}
using Int64OrError = std::variant<int64_t, FusionDecision>;
Int64OrError CombineSplitDimMajorPartSizeReqs(int64_t a, int64_t b) {
  if (a == b || b == kNoSplitRequirement) {
    return a;
  }
  if (a == kNoSplitRequirement) {
    return b;
  }
  return FusionDecision::Forbid("Conflicting splits of splittable dimension");
}
}  
DotRequirementsOrError CombineDotRequirements(
    DotRequirements a, DotRequirementsOrError b_or_error) {
  if (std::holds_alternative<FusionDecision>(b_or_error)) {
    return b_or_error;
  }
  const DotRequirements& b = std::get<DotRequirements>(b_or_error);
  Int64OrError combined_size_req =
      CombineSplitDimMajorPartSizeReqs(a.splittable_dimension_major_part_size,
                                       b.splittable_dimension_major_part_size);
  if (std::holds_alternative<FusionDecision>(combined_size_req)) {
    return std::get<FusionDecision>(combined_size_req);
  }
  return DotRequirements(std::get<int64_t>(combined_size_req));
}
namespace {
DotRequirementsOrError GetRequirementsIfSupportedOrder(
    const DimensionOrder& order, const DotProperties& properties) {
  VLOG(8) << order.ToString();
  int64_t split_dim_major_part = kNoSplitRequirement;
  const Fragments& tensor_dim_fragments = order.TensorFragmentsOrder();
  for (const auto& [dim_index, dim_fragments] : order.DimFragmentsOrders()) {
    CHECK(!dim_fragments.empty());
    for (int i = 0; i < dim_fragments.size() - 1; ++i) {
      if (tensor_dim_fragments[dim_fragments[i]].is_sliced()) {
        return FusionDecision::Forbid("Sliced non-major-most fragment.");
      }
    }
    int group_counter = 0;
    int last_seen_group_last_fragment_index = -1;
    auto fragment_it = dim_fragments.cbegin();
    while (true) {
      if (fragment_it == dim_fragments.cend()) {
        break;
      }
      int64_t grouped_size = tensor_dim_fragments[*fragment_it].full_count();
      while ((fragment_it + 1) != dim_fragments.cend() &&
             *(fragment_it + 1) == *fragment_it + 1) {
        ++fragment_it;
        grouped_size *= tensor_dim_fragments[*fragment_it].full_count();
      }
      if (grouped_size == 1) {
        ++fragment_it;
        continue;
      }
      if (last_seen_group_last_fragment_index > *fragment_it) {
        return FusionDecision::Forbid("Transpose within a dimension.");
      }
      ++group_counter;
      if (group_counter > 1) {
        const int splittable_dimension_index =
            properties.splittable_dimension_index;
        if (dim_index == splittable_dimension_index) {
          if (group_counter == 2) {
            if (split_dim_major_part != kNoSplitRequirement &&
                split_dim_major_part != grouped_size) {
              return FusionDecision::Forbid(
                  "Conflicting splits of splittable dimension");
            }
            split_dim_major_part = grouped_size;
          } else if (group_counter > 2) {
            return FusionDecision::Forbid(
                "2nd split of a splittable dimension.");
          }
        } else {
          return FusionDecision::Forbid("Unsupported split of a dimension.");
        }
      }
      last_seen_group_last_fragment_index = *fragment_it;
      ++fragment_it;
    }
  }
  return DotRequirements(split_dim_major_part);
}
DotRequirementsOrError GetRequirementsIfSupportedOrders(
    const HloInstruction& hlo, const DimOrderMap& dim_orders,
    const DotProperties& properties) {
  const DotRequirements empty_requirements(kNoSplitRequirement);
  auto get_requirements =
      [&](const HloInstruction& instr) -> DotRequirementsOrError {
    if (auto it = dim_orders.find(&instr); it != dim_orders.end()) {
      return GetRequirementsIfSupportedOrder(it->second, properties);
    }
    return empty_requirements;
  };
  DotRequirements requirements = empty_requirements;
  for (const HloInstruction* operand : hlo.operands()) {
    DotRequirementsOrError requirements_or_error =
        CombineDotRequirements(requirements, get_requirements(*operand));
    if (std::holds_alternative<FusionDecision>(requirements_or_error)) {
      return requirements_or_error;
    }
    requirements = std::get<DotRequirements>(requirements_or_error);
  }
  return CombineDotRequirements(requirements, get_requirements(hlo));
}
DimOrderMap GetPropagatedDimOrdersForElementwise(
    const HloInstruction& hlo, TransformDirection direction,
    const DimensionOrder& src_dim_order) {
  if (direction == TransformDirection::kOutputToInput) {
    DimOrderMap map;
    for (const HloInstruction* operand : hlo.operands()) {
      map.insert({operand, src_dim_order});
    }
    return map;
  }
  return {{&hlo, src_dim_order}};
}
const HloInstruction& GetSourceHlo(const HloInstruction& hlo,
                                   TransformDirection direction) {
  CHECK_GE(hlo.operand_count(), 1);
  if (direction == TransformDirection::kOutputToInput) {
    return hlo;
  }
  return *hlo.operand(0);
}
using ConstInstructionVector = absl::InlinedVector<const HloInstruction*, 2>;
ConstInstructionVector GetDestHlos(const HloInstruction& hlo,
                                   TransformDirection direction) {
  if (direction == TransformDirection::kInputToOutput) {
    return {&hlo};
  }
  ConstInstructionVector hlos;
  hlos.reserve(hlo.operands().size());
  for (const HloInstruction* operand : hlo.operands()) {
    hlos.push_back(operand);
  }
  return hlos;
}
const HloInstruction& GetDestHlo(const HloInstruction& hlo,
                                 TransformDirection direction) {
  CHECK_EQ(hlo.operand_count(), 1);
  if (direction == TransformDirection::kInputToOutput) {
    return hlo;
  }
  return *hlo.operand(0);
}
DimOrderMapOrError GetPropagatedDimOrdersForBitcast(
    const HloInstruction& hlo, const TransformDirection direction,
    const DimensionOrder& src_dim_order, const DotProperties& properties) {
  const HloInstruction& dst = GetDestHlo(hlo, direction);
  const Shape& dst_shape = dst.shape();
  const Fragments& src_fragments_order = src_dim_order.TensorFragmentsOrder();
  DimOrderMap dst_dim_orders;
  DimensionOrder& dst_dim_order =
      dst_dim_orders.insert({&dst, DimensionOrder()}).first->second;
  Fragments& dst_fragments_order = dst_dim_order.TensorFragmentsOrder();
  int64_t dst_remaining_size = 1;
  absl::flat_hash_map<const Fragment*, std::vector<int>> src_to_dst;
  auto dst_dim_it = dst_shape.layout().minor_to_major().cbegin();
  const auto dst_dim_end = dst_shape.layout().minor_to_major().cend();
  for (auto src_dim = src_fragments_order.cbegin();
       src_dim != src_fragments_order.cend(); ++src_dim) {
    auto add_new_fragment = [&](const Fragment& fragment) {
      dst_fragments_order.push_back(fragment);
      src_to_dst[&*src_dim].push_back(dst_fragments_order.size() - 1);
    };
    if (dst_remaining_size >= src_dim->full_count()) {
      if (dst_remaining_size % src_dim->full_count()) {
        return FusionDecision::Forbid("Unsupported bitcast");
      }
      add_new_fragment(*src_dim);
      dst_remaining_size /= src_dim->full_count();
    } else {
      int64_t src_remaining_size = src_dim->full_count();
      if (dst_remaining_size > 1) {
        if (src_remaining_size % dst_remaining_size || (src_dim->is_sliced())) {
          return FusionDecision::Forbid("Unsupported bitcast");
        }
        add_new_fragment(
            Fragment{src_dim->dst_dim_number(), dst_remaining_size});
        src_remaining_size /= dst_remaining_size;
        dst_remaining_size = 1;
      }
      while (src_remaining_size > 1) {
        CHECK(dst_dim_it != dst_dim_end);
        int64_t dst_dim_size = dst_shape.dimensions(*dst_dim_it);
        int64_t new_fragment_size = dst_dim_size;
        if (dst_dim_size > src_remaining_size) {
          if (dst_dim_size % src_remaining_size) {
            return FusionDecision::Forbid("Unsupported bitcast");
          }
          dst_remaining_size = dst_dim_size / src_remaining_size;
          new_fragment_size = src_remaining_size;
        }
        if (src_dim->is_sliced()) {
          return FusionDecision::Forbid("Unsupported bitcast");
        }
        add_new_fragment(
            Fragment{src_dim->dst_dim_number(), new_fragment_size});
        src_remaining_size /= new_fragment_size;
        ++dst_dim_it;
      }
    }
  }
  CHECK_EQ(dst_remaining_size, 1);
  while (dst_dim_it != dst_dim_end) {
    if (dst_shape.dimensions(*dst_dim_it) != 1) {
      return FusionDecision::Forbid("Unsupported bitcast");
    }
    if (!dst_fragments_order.empty()) {
      dst_fragments_order.push_back(
          Fragment{dst_fragments_order.back().dst_dim_number(), 1});
      src_to_dst[&src_fragments_order.back()].push_back(
          dst_fragments_order.size() - 1);
    }
    ++dst_dim_it;
  }
  FragmentOrders& dst_dim_fragment_orders = dst_dim_order.DimFragmentsOrders();
  for (const auto& [dim_index, dim_sequence] :
       src_dim_order.DimFragmentsOrders()) {
    std::vector<int>& dst = dst_dim_fragment_orders[dim_index];
    dst.reserve(dim_sequence.size());
    for (const int src : dim_sequence) {
      std::copy(src_to_dst[&src_fragments_order[src]].cbegin(),
                src_to_dst[&src_fragments_order[src]].cend(),
                std::back_inserter(dst));
    }
  }
  return dst_dim_orders;
}
DimOrderMapOrError GetPropagatedDimOrdersForDimAlteringOp(
    const HloInstruction& hlo, const TransformDirection direction,
    const DimensionOrder& src_dim_order, const DotProperties& properties) {
  std::list<Fragment> new_fragments;
  const HloInstruction& src = GetSourceHlo(hlo, direction);
  Fragments src_fragments_order = src_dim_order.TensorFragmentsOrder();
  if (hlo.opcode() == HloOpcode::kSlice &&
      ShapeUtil::IsEffectiveScalar(hlo.shape())) {
    return FusionDecision::Forbid("Slice to scalar is not implemented yet.");
  }
  std::vector<std::vector<Fragment*>> src_physical;
  src_physical.reserve(src.shape().rank());
  if (src_fragments_order.size() < src.shape().rank()) {
    return FusionDecision::Forbid(
        "Cannot propagate further from trivial sized tensor");
  }
  auto src_fragment_it = src_fragments_order.begin();
  for (int64_t dim_index : src.shape().layout().minor_to_major()) {
    const int64_t dim_size = src.shape().dimensions(dim_index);
    int64_t subdim_size_accumulator = 1;
    std::vector<Fragment*> subdim_group;
    do {
      CHECK(src_fragment_it != src_fragments_order.end());
      subdim_size_accumulator *= src_fragment_it->full_count();
      subdim_group.push_back(&*src_fragment_it);
      ++src_fragment_it;
    } while (subdim_size_accumulator < dim_size);
    CHECK_EQ(subdim_size_accumulator, dim_size);
    src_physical.push_back(subdim_group);
  }
  std::vector<std::vector<Fragment*>> src_logical;
  src_logical.resize(src_physical.size());
  for (int i = 0; i < src_physical.size(); ++i) {
    src_logical[src.shape().layout().minor_to_major(i)] = src_physical[i];
  }
  DimOrderMap dst_dim_orders;
  int64_t concat_accumulated_size = 0;
  for (const HloInstruction* dst : GetDestHlos(hlo, direction)) {
    DimensionOrder& dst_dim_order =
        dst_dim_orders.insert({dst, DimensionOrder()}).first->second;
    std::vector<std::vector<Fragment*>> dst_logical;
    if (hlo.opcode() == HloOpcode::kTranspose) {
      const auto* transpose = Cast<HloTransposeInstruction>(&hlo);
      std::vector<int64_t> permutation(transpose->dimensions().cbegin(),
                                       transpose->dimensions().cend());
      if (direction == TransformDirection::kInputToOutput) {
        permutation = InversePermutation(permutation);
      }
      dst_logical.resize(permutation.size());
      for (int i = 0; i < permutation.size(); ++i) {
        dst_logical[permutation[i]] = src_logical[i];
      }
    } else if (hlo.opcode() == HloOpcode::kBroadcast) {
      const auto* broadcast = Cast<HloBroadcastInstruction>(&hlo);
      dst_logical.resize(broadcast->dimensions().size());
      for (int i = 0; i < broadcast->dimensions().size(); ++i) {
        dst_logical[i] = src_logical[broadcast->dimensions()[i]];
      }
    } else if (hlo.opcode() == HloOpcode::kReduce) {
      if (dst != &hlo && hlo.operand_index(dst) == 1) {
        continue;
      }
      const auto* reduce = Cast<HloReduceInstruction>(&hlo);
      dst_logical.resize(src_logical.size() + reduce->dimensions().size());
      if (reduce->dimensions().size() != 1) {
        return FusionDecision::Forbid("Unsupported reduction.");
      } else if (reduce->dimensions().front() !=
                 reduce->operand(0)->shape().rank() - 1) {
        return FusionDecision::Forbid("Only row reductions are supported.");
      }
    } else if (hlo.opcode() == HloOpcode::kConcatenate) {
      dst_logical.resize(src_logical.size());
      for (int i = 0; i < src_logical.size(); ++i) {
        if (i == hlo.concatenate_dimension()) {
          if (src_logical[i].size() != 1 || src_logical[i][0]->is_sliced()) {
            return FusionDecision::Forbid("Unsupported concatenation.");
          }
          const Fragment& src_fragment = *src_logical[i][0];
          Fragment& dst_fragment = new_fragments.emplace_back(
              src_fragment.dst_dim_number(), dst->shape().dimensions(i));
          dst_fragment.set_slice(-concat_accumulated_size,
                                 dst->shape().dimensions(i));
          concat_accumulated_size += dst->shape().dimensions(i);
          dst_logical[i].push_back(&dst_fragment);
        } else {
          dst_logical[i] = src_logical[i];
        }
      }
    } else if (hlo.opcode() == HloOpcode::kCopy) {
      CHECK(ShapeUtil::SameDimensions(src.shape(), dst->shape()));
      dst_logical = src_logical;
    } else if (hlo.opcode() == HloOpcode::kPad) {
      if (dst != &hlo && hlo.operand_index(dst) == 1) {
        continue;
      }
      const auto* pad = Cast<HloPadInstruction>(&hlo);
      dst_logical.resize(src_logical.size());
      for (int i = 0; i < src_logical.size(); ++i) {
        const int padding =
            pad->padding_config().dimensions(i).edge_padding_high();
        CHECK_EQ(pad->padding_config().dimensions(i).edge_padding_low(), 0);
        CHECK_EQ(pad->padding_config().dimensions(i).interior_padding(), 0);
        if (padding == 0) {
          dst_logical[i] = src_logical[i];
        } else {
          const std::vector<Fragment*>& fragments = src_logical[i];
          CHECK_GE(fragments.size(), 2);
          CHECK(absl::c_all_of(fragments, [&](const Fragment* fragment) {
            return fragment->dst_dim_number() ==
                   fragments.front()->dst_dim_number();
          }));
          std::vector<Fragment*> non_trivial_fragments;
          absl::c_copy_if(fragments, std::back_inserter(non_trivial_fragments),
                          [](const Fragment* fragment) {
                            return fragment->full_count() > 1;
                          });
          CHECK_EQ(non_trivial_fragments.size(), 2);
          new_fragments.emplace_back(
              non_trivial_fragments[0]->dst_dim_number(),
              non_trivial_fragments[0]->full_count() *
                      non_trivial_fragments[1]->full_count() -
                  padding);
          dst_logical[i] = {&new_fragments.back()};
        }
      }
    } else if (hlo.opcode() == HloOpcode::kSlice) {
      const auto slice = Cast<HloSliceInstruction>(&hlo);
      dst_logical.resize(src_logical.size());
      for (int dim = 0; dim < src_logical.size(); ++dim) {
        dst_logical[dim] = src_logical[dim];
        if (slice->slice_limits(dim) - slice->slice_starts(dim) !=
            dst->shape().dimensions(dim)) {
          if (dst_logical[dim].size() > 1) {
            return FusionDecision::Forbid("Slicing of fragmented dimension.");
          }
          auto fragment = dst_logical[dim].front();
          fragment->set_count(dst->shape().dimensions(dim));
          fragment->set_slice(
              fragment->slice_start() + slice->slice_starts(dim),
              fragment->sliced_count());
        }
      }
    } else if (hlo.opcode() == HloOpcode::kDynamicSlice) {
      if (dst != &hlo && hlo.operand_index(dst) >= 1) {
        continue;
      }
      const auto dynamic_slice = Cast<HloDynamicSliceInstruction>(&hlo);
      dst_logical.resize(src_logical.size());
      for (int dim = 0; dim < src_logical.size(); ++dim) {
        dst_logical[dim] = src_logical[dim];
        if (dynamic_slice->slice_sizes(dim) != dst->shape().dimensions(dim)) {
          if (dst_logical[dim].size() > 1) {
            return FusionDecision::Forbid("Slicing of fragmented dimension.");
          }
          auto fragment = dst_logical[dim].front();
          fragment->set_count(dst->shape().dimensions(dim));
          fragment->set_slice(fragment->slice_start(),
                              dst->shape().dimensions(dim));
        }
      }
    } else {
      return FusionDecision::Forbid("Function called on a wrong instruction.");
    }
    absl::flat_hash_map<const Fragment*, int> src_to_dst;
    Fragments& dst_fragments_order = dst_dim_order.TensorFragmentsOrder();
    FragmentOrders& dst_dim_fragments_order =
        dst_dim_order.DimFragmentsOrders();
    absl::flat_hash_set<int> dim_numbers_present_in_dst;
    for (const int64_t dim_idx : dst->shape().layout().minor_to_major()) {
      for (const Fragment* subdim : dst_logical[dim_idx]) {
        dst_fragments_order.push_back(*subdim);
        src_to_dst[subdim] = dst_fragments_order.size() - 1;
        dim_numbers_present_in_dst.insert(subdim->dst_dim_number());
      }
    }
    for (const auto& [dim_index, dim_sequence] :
         src_dim_order.DimFragmentsOrders()) {
      for (const int fragment_number : dim_sequence) {
        const auto it = src_to_dst.find(&src_fragments_order[fragment_number]);
        if (it == src_to_dst.cend()) {
          if (hlo.opcode() == HloOpcode::kBroadcast &&
              src_fragments_order[fragment_number].full_count() > 1 &&
              dim_numbers_present_in_dst.contains(dim_index)) {
            return FusionDecision::Forbid("Unsupported broadcast");
          }
          continue;
        }
        dst_dim_fragments_order[dim_index].push_back(it->second);
      }
    }
  }
  return dst_dim_orders;
}
DimOrderMapOrError GetPropagatedDimOrders(const HloInstruction& hlo,
                                          const TransformDirection direction,
                                          const DimensionOrder& src_dim_order,
                                          const DotProperties& properties) {
  VLOG(7) << "Analyzing " << hlo.ToString();
  if (hlo.opcode() != HloOpcode::kParameter &&
      direction == TransformDirection::kOutputToInput &&
      absl::c_any_of(hlo.users(), [](const HloInstruction* user) {
        return (user->opcode() == HloOpcode::kConcatenate ||
                user->opcode() == HloOpcode::kDynamicSlice);
      })) {
    return FusionDecision::Forbid(
        "No fusion into concatenations or dynamic slice.");
  }
  if (hlo.opcode() == HloOpcode::kParameter ||
      hlo_query::IsScalarConstant(&hlo)) {
    CHECK(direction == TransformDirection::kOutputToInput);
    return DimOrderMap{};
  } else if (hlo.opcode() == HloOpcode::kTranspose ||
             hlo.opcode() == HloOpcode::kCopy) {
    return GetPropagatedDimOrdersForDimAlteringOp(hlo, direction, src_dim_order,
                                                  properties);
  } else if (hlo.opcode() == HloOpcode::kBroadcast) {
    if (direction != TransformDirection::kOutputToInput) {
      return FusionDecision::Forbid("Unsupported broadcast direction.");
    }
    return GetPropagatedDimOrdersForDimAlteringOp(hlo, direction, src_dim_order,
                                                  properties);
  } else if (hlo.opcode() == HloOpcode::kPad) {
    if (direction != TransformDirection::kOutputToInput) {
      return FusionDecision::Forbid("Unsupported pad direction.");
    }
    return GetPropagatedDimOrdersForDimAlteringOp(hlo, direction, src_dim_order,
                                                  properties);
  } else if (hlo.operand_count() > 0 &&
             legacy_triton::IsTritonSupportedElementwiseUpToFloatNormalization(
                 hlo.opcode(), hlo.operand(0)->shape().element_type())) {
    return GetPropagatedDimOrdersForElementwise(hlo, direction, src_dim_order);
  } else if (hlo.opcode() == HloOpcode::kBitcast) {
    return GetPropagatedDimOrdersForBitcast(hlo, direction, src_dim_order,
                                            properties);
  } else if (hlo.opcode() == HloOpcode::kSlice) {
    if (direction != TransformDirection::kOutputToInput) {
      return FusionDecision::Forbid("Unsupported slice direction.");
    }
    return GetPropagatedDimOrdersForDimAlteringOp(hlo, direction, src_dim_order,
                                                  properties);
  } else if (hlo.opcode() == HloOpcode::kDynamicSlice &&
             direction == TransformDirection::kOutputToInput) {
    if (CodegenDecision decision = legacy_triton::IsTritonSupportedDynamicSlice(
            *Cast<HloDynamicSliceInstruction>(&hlo));
        !decision.CanFuse()) {
      return decision;
    }
    return GetPropagatedDimOrdersForDimAlteringOp(hlo, direction, src_dim_order,
                                                  properties);
  } else if (hlo.opcode() == HloOpcode::kReshape) {
    if (!ShapeUtil::ReshapeIsBitcast(hlo.operand(0)->shape(), hlo.shape())) {
      return FusionDecision::Forbid("Non-bitcast reshape.");
    }
    return GetPropagatedDimOrdersForBitcast(hlo, direction, src_dim_order,
                                            properties);
  } else if (hlo.opcode() == HloOpcode::kConcatenate &&
             direction == TransformDirection::kOutputToInput) {
    int64_t noncontracting_dim_label = properties.noncontracting_dimension;
    const FragmentOrders& src_dim_fragments_orders =
        src_dim_order.DimFragmentsOrders();
    auto noncontracting_dim_fragment_order_it =
        src_dim_fragments_orders.find(noncontracting_dim_label);
    if (noncontracting_dim_fragment_order_it !=
        src_dim_fragments_orders.end()) {
      if (noncontracting_dim_fragment_order_it->second.size() > 1) {
        return FusionDecision::Forbid(
            "Concatenations on split non-contracting dimensions are "
            "unsupported.");
      }
    }
    auto dim = LogicalIndexOfLabeledDimension(hlo.shape(), src_dim_order,
                                              noncontracting_dim_label);
    if (!dim.has_value() || dim.value() != hlo.concatenate_dimension()) {
      return FusionDecision::Forbid("Unsupported concatenation.");
    }
    if (absl::c_any_of(hlo.operands(), [&hlo](const HloInstruction* operand) {
          constexpr int kMinConcatFragmentSize = 64;
          return operand->shape().dimensions(hlo.concatenate_dimension()) %
                     kMinConcatFragmentSize !=
                 0;
        })) {
      return FusionDecision::Forbid(
          "At least one operand of concatenation can not be perfectly tiled.");
    }
    return GetPropagatedDimOrdersForDimAlteringOp(hlo, direction, src_dim_order,
                                                  properties);
  }
  return FusionDecision::Forbid("Unimplemented instruction.");
}
int64_t InputMinusOutputBytes(const HloInstruction& hlo) {
  CHECK(!hlo.shape().IsTuple());
  int64_t input_size = 0;
  for (const HloInstruction* operand : hlo.operands()) {
    CHECK(!operand->shape().IsTuple());
    input_size += ShapeUtil::ByteSizeOf(operand->shape());
  }
  return input_size - ShapeUtil::ByteSizeOf(hlo.shape());
}
bool CanNotBeFusedIntoAUser(const HloInstruction& hlo) {
  return hlo.IsRoot() || (hlo.user_count() == 1 && hlo.users()[0]->IsRoot() &&
                          hlo.users()[0]->opcode() == HloOpcode::kTuple);
}
constexpr int kIoToleranceBytes = 1024;
bool IsInputWorthFusing(const HloInstruction& hlo) {
  if (InputMinusOutputBytes(hlo) <= kIoToleranceBytes) {
    return true;
  }
  if (hlo.user_count() > 1) {
    return false;
  }
  if (hlo.opcode() == HloOpcode::kSlice &&
      hlo_query::AllOperandsAreParametersOrConstants(hlo)) {
    return true;
  }
  return hlo_query::AllOperandsAreParametersOrConstantsWithSingleUser(hlo);
}
bool IsOutputWorthFusing(const HloInstruction& hlo) {
  return CanNotBeFusedIntoAUser(hlo) ||
         InputMinusOutputBytes(hlo) >= -kIoToleranceBytes;
}
FusionDecision IsConversionWorthFusing(const HloInstruction& input,
                                       se::GpuComputeCapability gpu_version) {
  if (ShapeUtil::ByteSizeOf(input.operand(0)->shape()) >
      ShapeUtil::ByteSizeOf(input.shape())) {
    return FusionDecision::Forbid("Narrowing conversion.");
  }
  return FusionDecision::Allow();
}
}  
DimOrdersAndReqsOrError GetPropagatedDimOrdersAndRequirements(
    const HloInstruction& hlo, const DimensionOrder& src_dim_order,
    TransformDirection direction, const DotProperties& properties) {
  DimOrderMapOrError propagated_dim_orders_or_error =
      GetPropagatedDimOrders(hlo, direction, src_dim_order, properties);
  if (std::holds_alternative<FusionDecision>(propagated_dim_orders_or_error)) {
    return std::get<FusionDecision>(propagated_dim_orders_or_error);
  }
  DimOrderMap propagated_dim_orders =
      std::move(std::get<DimOrderMap>(propagated_dim_orders_or_error));
  DotRequirementsOrError requirements_or_error =
      GetRequirementsIfSupportedOrders(hlo, propagated_dim_orders, properties);
  if (std::holds_alternative<FusionDecision>(requirements_or_error)) {
    return std::get<FusionDecision>(requirements_or_error);
  }
  return DimOrdersAndReqs{propagated_dim_orders,
                          std::get<DotRequirements>(requirements_or_error)};
}
DimOrdersAndReqsOrError
GetPropagatedDimOrdersAndRequirementsIfProfitablyFusible(
    const HloInstruction& hlo, TransformDirection transform_direction,
    const std::optional<int>& src_operand_index,
    const DimensionOrder& src_dim_order,
    const se::GpuComputeCapability& gpu_version,
    const DotProperties& properties) {
  CHECK_EQ(transform_direction == TransformDirection::kInputToOutput,
           src_operand_index.has_value());
  if (hlo.opcode() == HloOpcode::kTuple ||
      hlo.opcode() == HloOpcode::kGetTupleElement) {
    return FusionDecision::Forbid("Unsupported instruction.");
  }
  if (hlo.opcode() == HloOpcode::kReduce ||
      hlo.opcode() == HloOpcode::kAllReduce ||
      hlo.opcode() == HloOpcode::kAllReduceStart ||
      hlo.opcode() == HloOpcode::kAllReduceDone) {
    return FusionDecision::Forbid("Reductions are not fused yet.");
  }
  if (hlo.opcode() == HloOpcode::kPad) {
    return FusionDecision::Forbid("Pads are not fused yet.");
  }
  if (auto decision =
          legacy_triton::IsTritonSupportedInstruction(hlo, gpu_version);
      !decision.CanFuse()) {
    return decision;
  }
  DimOrdersAndReqsOrError result_or_error =
      GetPropagatedDimOrdersAndRequirements(hlo, src_dim_order,
                                            transform_direction, properties);
  if (std::holds_alternative<FusionDecision>(result_or_error)) {
    VLOG(5) << "Not fusing " << hlo.ToString()
            << " to the output due to the decision: "
            << std::get<FusionDecision>(result_or_error).Explain();
    return result_or_error;
  }
  DimOrdersAndReqs dim_orders_and_requirements =
      std::move(std::get<DimOrdersAndReqs>(result_or_error));
  int fusion_level =
      hlo.GetModule()->config().debug_options().xla_gpu_triton_fusion_level();
  if (transform_direction == TransformDirection::kOutputToInput) {
    if (fusion_level < 2) {
      if (hlo.opcode() == HloOpcode::kConvert) {
        if (FusionDecision decision = IsConversionWorthFusing(hlo, gpu_version);
            !decision) {
          return decision;
        }
      } else if (hlo.IsElementwise() && hlo.opcode() != HloOpcode::kCopy) {
        return FusionDecision::Forbid("Ignored elementwise operation");
      }
    } else {
      bool accepted = false;
      if (hlo.IsElementwise() && hlo.operand_count() == 2) {
        for (const HloInstruction* operand : hlo.operands()) {
          if (operand->opcode() == HloOpcode::kBroadcast &&
              (operand->operand(0)->opcode() == HloOpcode::kParameter ||
               operand->operand(0)->opcode() == HloOpcode::kConstant) &&
              std::holds_alternative<DimOrdersAndReqs>(
                  GetPropagatedDimOrdersAndRequirementsIfProfitablyFusible(
                      *operand, TransformDirection::kOutputToInput,
                      std::nullopt,
                      dim_orders_and_requirements.dim_orders.at(operand),
                      gpu_version, properties))) {
            accepted = true;
            break;
          }
        }
      }
      if (!accepted && !IsInputWorthFusing(hlo)) {
        return FusionDecision::Forbid(
            "Not obviously profitable to fuse as input.");
      }
    }
  } else {
    if (fusion_level < 2) {
      return FusionDecision::Forbid(
          "Skipping fusing outputs at low fusion levels.");
    }
    for (int i = 0; i < hlo.operand_count(); ++i) {
      const HloInstruction* operand = hlo.operand(i);
      if (i == *src_operand_index) {
        continue;
      }
      if ((operand->opcode() == HloOpcode::kBroadcast &&
           ShapeUtil::IsScalar(operand->operand(0)->shape())) ||
          operand->opcode() == HloOpcode::kParameter) {
        continue;
      }
      return FusionDecision::Forbid(
          "Has multiple inputs - not properly analyzed yet.");
    }
    if (!IsOutputWorthFusing(hlo)) {
      return FusionDecision::Forbid(
          "Not obviously profitable to fuse as output.");
    }
  }
  return dim_orders_and_requirements;
}
}  
}  
}  