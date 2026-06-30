#include "tensorstore/index_space/dimension_permutation.h"
#include <algorithm>
#include <numeric>
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/dimension_set.h"
#include "tensorstore/util/span.h"
namespace tensorstore {
void SetPermutation(ContiguousLayoutOrder order,
                    span<DimensionIndex> permutation) {
  if (order == c_order) {
    for (DimensionIndex i = 0; i < permutation.size(); ++i) {
      permutation[i] = i;
    }
  } else {
    for (DimensionIndex i = 0; i < permutation.size(); ++i) {
      permutation[i] = permutation.size() - 1 - i;
    }
  }
}
bool IsValidPermutation(span<const DimensionIndex> permutation) {
  DimensionSet seen_dims;
  const DimensionIndex rank = permutation.size();
  if (rank > kMaxRank) return false;
  for (DimensionIndex i = 0; i < rank; ++i) {
    DimensionIndex dim = permutation[i];
    if (dim < 0 || dim >= rank || seen_dims[dim]) {
      return false;
    }
    seen_dims[dim] = true;
  }
  return true;
}
bool PermutationMatchesOrder(span<const DimensionIndex> permutation,
                             ContiguousLayoutOrder order) {
  if (order == c_order) {
    for (DimensionIndex i = 0; i < permutation.size(); ++i) {
      if (permutation[i] != i) return false;
    }
  } else {
    for (DimensionIndex i = 0; i < permutation.size(); ++i) {
      if (permutation[i] != permutation.size() - i - 1) return false;
    }
  }
  return true;
}
void InvertPermutation(DimensionIndex rank, const DimensionIndex* perm,
                       DimensionIndex* inverse_perm) {
  assert(IsValidPermutation(span(perm, rank)));
  for (DimensionIndex i = 0; i < rank; ++i) {
    inverse_perm[perm[i]] = i;
  }
}
void SetPermutationFromStridedLayout(StridedLayoutView<> layout,
                                     span<DimensionIndex> permutation) {
  assert(layout.rank() == permutation.size());
  std::iota(permutation.begin(), permutation.end(), DimensionIndex(0));
  const auto get_effective_byte_stride_nabs = [&](DimensionIndex i) -> Index {
    const Index byte_stride = layout.byte_strides()[i];
    if (byte_stride > 0) return -byte_stride;
    return byte_stride;
  };
  std::stable_sort(permutation.begin(), permutation.end(),
                   [&](DimensionIndex a, DimensionIndex b) {
                     return get_effective_byte_stride_nabs(a) <
                            get_effective_byte_stride_nabs(b);
                   });
}
void TransformOutputDimensionOrder(IndexTransformView<> transform,
                                   span<const DimensionIndex> output_perm,
                                   span<DimensionIndex> input_perm) {
  assert(transform.valid());
  assert(IsValidPermutation(output_perm));
  const DimensionIndex output_rank = transform.output_rank();
  const DimensionIndex input_rank = transform.input_rank();
  assert(input_rank == input_perm.size());
  assert(output_rank == output_perm.size());
  DimensionIndex min_output_dim[kMaxRank];
  std::fill_n(min_output_dim, input_rank, kMaxRank);
  for (DimensionIndex orig_perm_i = 0; orig_perm_i < output_rank;
       ++orig_perm_i) {
    const DimensionIndex output_dim = output_perm[orig_perm_i];
    const auto map = transform.output_index_maps()[output_dim];
    if (map.method() != OutputIndexMethod::single_input_dimension) continue;
    const DimensionIndex input_dim = map.input_dimension();
    min_output_dim[input_dim] =
        std::min(min_output_dim[input_dim], orig_perm_i);
  }
  std::iota(input_perm.begin(), input_perm.end(), DimensionIndex(0));
  std::sort(input_perm.begin(), input_perm.end(),
            [&](DimensionIndex a, DimensionIndex b) {
              DimensionIndex a_ordinal = min_output_dim[a];
              DimensionIndex b_ordinal = min_output_dim[b];
              if (a_ordinal != b_ordinal) return a_ordinal < b_ordinal;
              return a < b;
            });
  assert(IsValidPermutation(input_perm));
}
void TransformInputDimensionOrder(IndexTransformView<> transform,
                                  span<const DimensionIndex> input_perm,
                                  span<DimensionIndex> output_perm) {
  assert(transform.valid());
  assert(IsValidPermutation(input_perm));
  [[maybe_unused]] const DimensionIndex output_rank = transform.output_rank();
  const DimensionIndex input_rank = transform.input_rank();
  assert(input_rank == input_perm.size());
  assert(output_rank == output_perm.size());
  DimensionIndex inverse_input_perm[kMaxRank];
  InvertPermutation(input_rank, input_perm.data(), inverse_input_perm);
  std::iota(output_perm.begin(), output_perm.end(), DimensionIndex(0));
  const auto get_output_dim_ordinal = [&](DimensionIndex output_dim) {
    const auto map = transform.output_index_maps()[output_dim];
    if (map.method() != OutputIndexMethod::single_input_dimension) {
      return kMaxRank;
    }
    return inverse_input_perm[map.input_dimension()];
  };
  std::sort(output_perm.begin(), output_perm.end(),
            [&](DimensionIndex a, DimensionIndex b) {
              DimensionIndex a_ordinal = get_output_dim_ordinal(a);
              DimensionIndex b_ordinal = get_output_dim_ordinal(b);
              if (a_ordinal != b_ordinal) return a_ordinal < b_ordinal;
              return a < b;
            });
  assert(IsValidPermutation(output_perm));
}
}  