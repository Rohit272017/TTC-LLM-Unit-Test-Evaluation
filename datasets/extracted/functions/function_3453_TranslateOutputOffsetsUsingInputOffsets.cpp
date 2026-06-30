#include "tensorstore/index_space/internal/translate_op.h"
#include <algorithm>
#include "absl/status/status.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/util/str_cat.h"
namespace tensorstore {
namespace internal_index_space {
namespace {
absl::Status TranslateOutputOffsetsUsingInputOffsets(
    TransformRep* transform, const Index* input_offsets) {
  const DimensionIndex output_rank = transform->output_rank;
  const DimensionIndex input_rank = transform->input_rank;
  span<OutputIndexMap> maps = transform->output_index_maps().first(output_rank);
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    auto& map = maps[output_dim];
    switch (map.method()) {
      case OutputIndexMethod::single_input_dimension: {
        const DimensionIndex input_dim = map.input_dimension();
        const Index offset_change = input_offsets[input_dim];
        Index new_offset;
        if (internal::MulOverflow(offset_change, map.stride(), &new_offset) ||
            internal::SubOverflow(map.offset(), new_offset, &map.offset())) {
          return absl::InvalidArgumentError(tensorstore::StrCat(
              "Integer overflow computing output offset for dimension ",
              output_dim, "."));
        }
        break;
      }
      case OutputIndexMethod::array: {
        auto& index_array_data = map.index_array_data();
        index_array_data.element_pointer = AddByteOffset(
            std::move(index_array_data.element_pointer),
            -IndexInnerProduct(input_rank, index_array_data.byte_strides,
                               input_offsets));
        break;
      }
      case OutputIndexMethod::constant:
        break;
    }
  }
  return absl::OkStatus();
}
}  
Result<IndexTransform<>> ApplyTranslate(IndexTransform<> transform,
                                        DimensionIndexBuffer* dimensions,
                                        IndexVectorOrScalarView offsets,
                                        TranslateOpKind kind,
                                        bool domain_only) {
  const DimensionIndex num_dims = dimensions->size();
  const DimensionIndex input_rank = transform.input_rank();
  TENSORSTORE_RETURN_IF_ERROR(CheckIndexVectorSize(offsets, num_dims));
  TransformRep::Ptr<> rep = MutableRep(
      TransformAccess::rep_ptr<container>(std::move(transform)), domain_only);
  const auto input_domain = rep->input_domain(input_rank);
  Index input_offsets[kMaxRank];
  std::fill_n(&input_offsets[0], input_rank, static_cast<Index>(0));
  for (DimensionIndex i = 0; i < num_dims; ++i) {
    const DimensionIndex input_dim = (*dimensions)[i];
    Index offset = offsets[i];
    if (offset == kImplicit) continue;
    const IndexInterval old_interval = input_domain[input_dim];
    IndexInterval new_interval;
    switch (kind) {
      case TranslateOpKind::kTranslateTo: {
        TENSORSTORE_ASSIGN_OR_RETURN(new_interval,
                                     ShiftIntervalTo(old_interval, offset));
        offset = new_interval.inclusive_min() - old_interval.inclusive_min();
        break;
      }
      case TranslateOpKind::kTranslateBackwardBy: {
        offset = -offset;
      }
        [[fallthrough]];
      case TranslateOpKind::kTranslateBy: {
        TENSORSTORE_ASSIGN_OR_RETURN(new_interval,
                                     ShiftInterval(old_interval, offset));
        break;
      }
    }
    input_domain[input_dim] = new_interval;
    input_offsets[input_dim] = offset;
  }
  TENSORSTORE_RETURN_IF_ERROR(
      TranslateOutputOffsetsUsingInputOffsets(rep.get(), &input_offsets[0]));
  internal_index_space::DebugCheckInvariants(rep.get());
  return TransformAccess::Make<IndexTransform<>>(std::move(rep));
}
}  
}  