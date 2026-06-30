#include "tensorstore/index_space/internal/add_new_dims_op.h"
#include <cassert>
#include <utility>
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/index_space/output_index_method.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/dimension_set.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
namespace tensorstore {
namespace internal_index_space {
namespace {
void AddNewDims(TransformRep* original, TransformRep* result,
                DimensionIndexBuffer* dimensions, bool domain_only) {
  const DimensionIndex orig_input_rank = original->input_rank;
  const DimensionIndex new_input_rank = orig_input_rank + dimensions->size();
  assert(result->input_rank_capacity >= new_input_rank);
  const DimensionIndex output_rank = domain_only ? 0 : original->output_rank;
  assert(result->output_rank_capacity >= output_rank);
  DimensionSet newly_added_input_dims;
  for (DimensionIndex new_input_dim : *dimensions) {
    newly_added_input_dims[new_input_dim] = true;
  }
  DimensionIndex orig_to_new_input_dim[kMaxRank];
  for (DimensionIndex new_input_dim = 0, orig_input_dim = 0;
       new_input_dim < new_input_rank; ++new_input_dim) {
    if (newly_added_input_dims[new_input_dim]) continue;
    orig_to_new_input_dim[orig_input_dim] = new_input_dim;
    ++orig_input_dim;
  }
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
        auto& result_index_array = result_map.SetArrayIndexing(new_input_rank);
        const auto& orig_index_array = orig_map.index_array_data();
        for (DimensionIndex orig_input_dim = orig_input_rank - 1;
             orig_input_dim >= 0; --orig_input_dim) {
          const DimensionIndex new_input_dim =
              orig_to_new_input_dim[orig_input_dim];
          assert(new_input_dim >= orig_input_dim);
          result_index_array.byte_strides[new_input_dim] =
              orig_index_array.byte_strides[orig_input_dim];
        }
        for (const DimensionIndex new_input_dim : *dimensions) {
          result_index_array.byte_strides[new_input_dim] = 0;
        }
        result_index_array.index_range = orig_index_array.index_range;
        result_index_array.element_pointer = orig_index_array.element_pointer;
        break;
      }
    }
  }
  for (DimensionIndex orig_input_dim = orig_input_rank - 1; orig_input_dim >= 0;
       --orig_input_dim) {
    const DimensionIndex new_input_dim = orig_to_new_input_dim[orig_input_dim];
    result->input_dimension(new_input_dim) =
        original->input_dimension(orig_input_dim);
  }
  for (DimensionIndex new_input_dim : *dimensions) {
    const auto d = result->input_dimension(new_input_dim);
    d.domain() = IndexInterval::UncheckedSized(-kInfIndex, kInfSize);
    d.implicit_lower_bound() = true;
    d.implicit_upper_bound() = true;
    d.SetEmptyLabel();
  }
  result->input_rank = new_input_rank;
  result->output_rank = output_rank;
}
}  
Result<IndexTransform<>> ApplyAddNewDims(IndexTransform<> transform,
                                         DimensionIndexBuffer* dimensions,
                                         bool domain_only) {
  const DimensionIndex new_input_rank =
      transform.input_rank() + dimensions->size();
  TENSORSTORE_RETURN_IF_ERROR(ValidateRank(new_input_rank));
  auto new_rep =
      NewOrMutableRep(TransformAccess::rep(transform), new_input_rank,
                      transform.output_rank(), domain_only);
  AddNewDims(TransformAccess::rep(transform), new_rep.get(), dimensions,
             domain_only);
  internal_index_space::DebugCheckInvariants(new_rep.get());
  return TransformAccess::Make<IndexTransform<>>(std::move(new_rep));
}
}  
}  