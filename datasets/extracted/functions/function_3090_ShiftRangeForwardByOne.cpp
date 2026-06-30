#include "tensorstore/index_space/internal/diagonal_op.h"
#include <algorithm>
namespace tensorstore {
namespace internal_index_space {
namespace {
template <typename R>
void ShiftRangeForwardByOne(R range) {
  for (DimensionIndex i = range.size() - 1; i > 0; --i) {
    range[i] = range[i - 1];
  }
}
void ExtractDiagonal(TransformRep* original, TransformRep* result,
                     DimensionIndexBuffer* dimensions, bool domain_only) {
  const DimensionIndex orig_input_rank = original->input_rank;
  const DimensionIndex output_rank = domain_only ? 0 : original->output_rank;
  const DimensionIndex new_input_rank =
      orig_input_rank - dimensions->size() + 1;
  assert(result->input_rank_capacity >= new_input_rank);
  const DimensionIndex diag_input_dim = 0;
  DimensionIndex orig_to_new_input_dim[kMaxRank];
  std::fill_n(&orig_to_new_input_dim[0], orig_input_rank,
              static_cast<DimensionIndex>(-1));
  bool lower_diagonal_bound_implicit = true,
       upper_diagonal_bound_implicit = true;
  IndexInterval diagonal_bounds;
  for (DimensionIndex orig_input_dim : *dimensions) {
    orig_to_new_input_dim[orig_input_dim] = diag_input_dim;
    const auto d = original->input_dimension(orig_input_dim);
    diagonal_bounds = Intersect(diagonal_bounds, d.domain());
    if (!d.implicit_lower_bound()) {
      lower_diagonal_bound_implicit = false;
    }
    if (!d.implicit_upper_bound()) {
      upper_diagonal_bound_implicit = false;
    }
  }
  for (DimensionIndex orig_input_dim = 0, new_input_dim = 1;
       orig_input_dim < orig_input_rank; ++orig_input_dim) {
    if (orig_to_new_input_dim[orig_input_dim] == -1) {
      orig_to_new_input_dim[orig_input_dim] = new_input_dim++;
    }
  }
  const bool domain_is_explicitly_empty = !lower_diagonal_bound_implicit &&
                                          !upper_diagonal_bound_implicit &&
                                          diagonal_bounds.empty();
  span<const OutputIndexMap> orig_maps =
      original->output_index_maps().first(output_rank);
  span<OutputIndexMap> result_maps =
      result->output_index_maps().first(output_rank);
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    const auto& orig_map = orig_maps[output_dim];
    auto& result_map = result_maps[output_dim];
    result_map.stride() = orig_map.stride();
    result_map.offset() = orig_map.offset();
    switch (orig_map.method()) {
      case OutputIndexMethod::constant:
        result_map.SetConstant();
        break;
      case OutputIndexMethod::single_input_dimension: {
        const DimensionIndex orig_input_dim = orig_map.input_dimension();
        assert(orig_input_dim >= 0 && orig_input_dim < orig_input_rank);
        const DimensionIndex new_input_dim =
            orig_to_new_input_dim[orig_input_dim];
        result_map.SetSingleInputDimension(new_input_dim);
        break;
      }
      case OutputIndexMethod::array: {
        if (domain_is_explicitly_empty) {
          result_map.SetConstant();
          result_map.stride() = 0;
          result_map.offset() = 0;
          break;
        }
        auto& result_index_array = result_map.SetArrayIndexing(new_input_rank);
        const auto& orig_index_array = orig_map.index_array_data();
        assert(orig_index_array.rank_capacity >= orig_input_rank);
        Index diag_byte_stride = 0;
        for (DimensionIndex orig_input_dim : *dimensions) {
          diag_byte_stride += orig_index_array.byte_strides[orig_input_dim];
        }
        for (DimensionIndex orig_input_dim = 0;
             orig_input_dim < orig_input_rank; ++orig_input_dim) {
          const DimensionIndex new_input_dim =
              orig_to_new_input_dim[orig_input_dim];
          if (new_input_dim == diag_input_dim) continue;
          assert(new_input_dim - 1 <= orig_input_dim);
          result_index_array.byte_strides[new_input_dim - 1] =
              orig_index_array.byte_strides[orig_input_dim];
        }
        ShiftRangeForwardByOne(
            span(result_index_array.byte_strides, new_input_rank));
        result_index_array.byte_strides[diag_input_dim] = diag_byte_stride;
        result_index_array.index_range = orig_index_array.index_range;
        result_index_array.element_pointer =
            orig_index_array.element_pointer.pointer();
        break;
      }
    }
  }
  for (DimensionIndex orig_input_dim = 0; orig_input_dim < orig_input_rank;
       ++orig_input_dim) {
    const DimensionIndex new_input_dim = orig_to_new_input_dim[orig_input_dim];
    if (new_input_dim == diag_input_dim) continue;
    assert(new_input_dim - 1 <= orig_input_dim);
    result->input_dimension(new_input_dim - 1) =
        original->input_dimension(orig_input_dim);
  }
  ShiftRangeForwardByOne(result->all_input_dimensions(new_input_rank));
  {
    const auto d = result->input_dimension(diag_input_dim);
    d.domain() = diagonal_bounds;
    d.implicit_lower_bound() = lower_diagonal_bound_implicit;
    d.implicit_upper_bound() = upper_diagonal_bound_implicit;
    d.SetEmptyLabel();
  }
  result->input_rank = new_input_rank;
  result->output_rank = output_rank;
  dimensions->clear();
  dimensions->push_back(diag_input_dim);
  NormalizeImplicitBounds(*result);
}
}  
Result<IndexTransform<>> ApplyDiagonal(IndexTransform<> transform,
                                       DimensionIndexBuffer* dimensions,
                                       bool domain_only) {
  TransformRep* rep = TransformAccess::rep(transform);
  const DimensionIndex new_input_rank =
      rep->input_rank - dimensions->size() + 1;
  TransformRep::Ptr<> new_rep =
      NewOrMutableRep(rep, new_input_rank, rep->output_rank, domain_only);
  ExtractDiagonal(rep, new_rep.get(), dimensions, domain_only);
  internal_index_space::DebugCheckInvariants(new_rep.get());
  return TransformAccess::Make<IndexTransform<>>(std::move(new_rep));
}
}  
}  