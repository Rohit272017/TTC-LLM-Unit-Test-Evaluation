#include "xla/service/gpu/transforms/transpose_dimension_grouper.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/layout_util.h"
#include "xla/permutation_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace gpu {
namespace {
absl::InlinedVector<size_t, 3> ConsecutiveSegments(
    absl::Span<const int64_t> xs) {
  absl::InlinedVector<size_t, 3> is = {0};
  for (size_t i = 1; i < xs.size(); ++i) {
    if (1 != xs[i] - xs[i - 1]) {
      is.push_back(i);
    }
  }
  return is;
}
Shape MergeDimensions(absl::Span<const size_t> segs, const Shape &shape) {
  std::vector<int64_t> dimensions;
  const auto size = segs.size();
  dimensions.reserve(size);
  for (size_t i = 1; i <= size; ++i) {
    dimensions.push_back(std::accumulate(
        shape.dimensions().begin() + segs[i - 1],
        shape.dimensions().begin() +
            (segs.size() == i ? shape.dimensions().size() : segs[i]),
        int64_t{1}, std::multiplies<int64_t>()));
  }
  return ShapeUtil::MakeShapeWithDescendingLayout(shape.element_type(),
                                                  dimensions);
}
absl::InlinedVector<int64_t, 3> GetNormalizedTransposeShapeHelper(
    const Shape &output_shape, absl::Span<int64_t const> output_to_input,
    absl::InlinedVector<int64_t, 3> &permutation) {
  absl::InlinedVector<size_t, 3> segments =
      ConsecutiveSegments(output_to_input);
  Shape normalized_shape = MergeDimensions(segments, output_shape);
  absl::InlinedVector<int64_t, 3> normalized_dims(
      normalized_shape.dimensions().begin(),
      normalized_shape.dimensions().end());
  if (segments.size() == 1) {
    return normalized_dims;
  }
  std::vector<int64_t> segment_to_normalized_dim(output_shape.rank(), -1);
  for (size_t segment : segments) {
    segment_to_normalized_dim[output_to_input[segment]] = 0;
  }
  int64_t normalized_dim = 0;
  for (int64_t i = 0; i < segment_to_normalized_dim.size(); ++i) {
    if (segment_to_normalized_dim[i] >= 0) {
      segment_to_normalized_dim[i] = normalized_dim++;
    }
  }
  permutation.reserve(segments.size());
  for (int64_t i = 0; i < segments.size(); ++i) {
    permutation.push_back(
        segment_to_normalized_dim[output_to_input[segments[i]]]);
  }
  return normalized_dims;
}
absl::InlinedVector<int64_t, 3> GetNormalizedLogicalTransposeShape(
    const Shape &output_shape, absl::Span<int64_t const> dimensions,
    absl::InlinedVector<int64_t, 3> &permutation) {
  permutation.clear();
  absl::InlinedVector<int64_t, 3> delta(output_shape.rank() + 1, 0);
  auto input_dimensions = ComposePermutations(output_shape.dimensions(),
                                              InversePermutation(dimensions));
  for (int i = 0; i < output_shape.rank(); ++i) {
    delta[i + 1] = delta[i];
    if (input_dimensions[i] == static_cast<int64_t>(1)) {
      ++delta[i + 1];
    }
  }
  absl::InlinedVector<int64_t, 3> new_dimensions;
  for (int i = 0; i < dimensions.size(); i++) {
    if (output_shape.dimensions(i) != 1) {
      new_dimensions.push_back(dimensions[i] - delta[dimensions[i]]);
    }
  }
  return GetNormalizedTransposeShapeHelper(
      ShapeUtil::DropDegenerateDimensions(output_shape), new_dimensions,
      permutation);
}
class TransposeDimensionGroupVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleTranspose(HloInstruction *transpose) override {
    VLOG(4) << "Input: " << transpose->ToString();
    if (!LayoutUtil::IsMonotonicWithDim0Major(transpose->shape().layout()) ||
        !LayoutUtil::IsMonotonicWithDim0Major(
            transpose->operand(0)->shape().layout())) {
      return FailedPrecondition(
          "Layout normalization should have assigned the default layout to "
          "transpose and its operand");
    }
    absl::InlinedVector<int64_t, 3> permutation;
    auto normalized_dims = GetNormalizedLogicalTransposeShape(
        transpose->shape(), transpose->dimensions(), permutation);
    if (normalized_dims.size() == 1 ||
        normalized_dims == transpose->shape().dimensions()) {
      return absl::OkStatus();
    }
    auto normalized_operand_dims =
        ComposePermutations(normalized_dims, InversePermutation(permutation));
    Shape grouped_operand_shape = ShapeUtil::MakeShapeWithDescendingLayout(
        transpose->shape().element_type(), normalized_operand_dims);
    auto new_operand = transpose->AddInstruction(HloInstruction::CreateBitcast(
        grouped_operand_shape, transpose->mutable_operand(0)));
    Shape grouped_shape = ShapeUtil::MakeShapeWithDescendingLayout(
        transpose->shape().element_type(), normalized_dims);
    auto new_transpose =
        transpose->AddInstruction(HloInstruction::CreateTranspose(
            grouped_shape, new_operand, permutation));
    VLOG(5) << "Generated new transpose: " << new_transpose->ToString();
    return ReplaceWithNewInstruction(
        transpose,
        HloInstruction::CreateBitcast(transpose->shape(), new_transpose));
  }
};
}  
absl::StatusOr<bool> TransposeDimensionGrouper::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  TF_ASSIGN_OR_RETURN(
      bool changed,
      TransposeDimensionGroupVisitor().RunOnModule(module, execution_threads));
  return changed;
}
}  
}  