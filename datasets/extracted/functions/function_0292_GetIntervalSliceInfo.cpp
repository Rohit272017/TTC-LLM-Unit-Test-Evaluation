#include "tensorstore/index_space/internal/interval_slice_op.h"
#include <algorithm>
#include "absl/status/status.h"
#include "tensorstore/index_space/internal/transform_rep_impl.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/str_cat.h"
namespace tensorstore {
namespace internal_index_space {
namespace {
struct InputDimensionIntervalSliceInfo {
  Index offset;
  Index stride;
};
absl::Status GetIntervalSliceInfo(
    span<InputDimensionIntervalSliceInfo> dimension_info,
    TransformRep* transform, span<const DimensionIndex> dimensions,
    IntervalForm interval_form, bool translate,
    IndexVectorOrScalarView start_vector,
    IndexVectorOrScalarView stop_or_size_vector,
    IndexVectorOrScalarView stride_vector) {
  const DimensionIndex input_rank = dimension_info.size();
  assert(input_rank == transform->input_rank);
  for (DimensionIndex input_dim = 0; input_dim < input_rank; ++input_dim) {
    dimension_info[input_dim] = InputDimensionIntervalSliceInfo{0, 1};
  }
  auto compute_input_domain_slice = [&](DimensionIndex i,
                                        DimensionIndex input_dim) {
    const Index stride = stride_vector[i];
    const InputDimensionRef d = transform->input_dimension(input_dim);
    auto& info = dimension_info[input_dim];
    info.stride = stride;
    OptionallyImplicitIndexInterval new_domain;
    TENSORSTORE_RETURN_IF_ERROR(ComputeStridedSliceMap(
        d.optionally_implicit_domain(), interval_form,
        translate ? 0 : kImplicit, start_vector[i], stop_or_size_vector[i],
        stride, &new_domain, &info.offset));
    d.domain() = new_domain.interval();
    d.implicit_lower_bound() = new_domain.implicit_lower();
    d.implicit_upper_bound() = new_domain.implicit_upper();
    return absl::OkStatus();
  };
  for (DimensionIndex i = 0; i < dimensions.size(); ++i) {
    const DimensionIndex input_dim = dimensions[i];
    TENSORSTORE_RETURN_IF_ERROR(
        compute_input_domain_slice(i, input_dim),
        MaybeAnnotateStatus(
            _,
            tensorstore::StrCat("Computing interval slice for input dimension ",
                                input_dim)));
  }
  return absl::OkStatus();
}
absl::Status ApplyOffsetsAndStridesToOutputIndexMaps(
    TransformRep* rep,
    span<const InputDimensionIntervalSliceInfo> input_dimension_info) {
  const DimensionIndex input_rank = input_dimension_info.size();
  const DimensionIndex output_rank = rep->output_rank;
  BoxView<> input_domain = rep->input_domain(input_rank);
  const bool domain_is_explicitly_empty = IsDomainExplicitlyEmpty(rep);
  span<OutputIndexMap> maps = rep->output_index_maps().first(output_rank);
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    auto& map = maps[output_dim];
    switch (map.method()) {
      case OutputIndexMethod::constant:
        break;
      case OutputIndexMethod::single_input_dimension: {
        const DimensionIndex input_dim = map.input_dimension();
        const auto& slice_info = input_dimension_info[input_dim];
        Index offset;
        if (internal::MulOverflow(slice_info.offset, map.stride(), &offset) ||
            internal::AddOverflow(offset, map.offset(), &map.offset())) {
          return absl::InvalidArgumentError(tensorstore::StrCat(
              "Integer overflow computing offset for output dimension ",
              output_dim));
        }
        if (internal::MulOverflow(slice_info.stride, map.stride(),
                                  &map.stride())) {
          return absl::InvalidArgumentError(tensorstore::StrCat(
              "Integer overflow computing stride for output dimension ",
              output_dim));
        }
        break;
      }
      case OutputIndexMethod::array: {
        if (domain_is_explicitly_empty) {
          map.SetConstant();
          map.offset() = 0;
          map.stride() = 0;
          break;
        }
        auto& index_array_data = map.index_array_data();
        Index element_pointer_byte_offset = 0;
        bool array_is_singleton = true;
        for (DimensionIndex input_dim = 0; input_dim < input_rank;
             ++input_dim) {
          const auto& slice_info = input_dimension_info[input_dim];
          Index& byte_stride = index_array_data.byte_strides[input_dim];
          element_pointer_byte_offset = internal::wrap_on_overflow::Add(
              element_pointer_byte_offset, internal::wrap_on_overflow::Multiply(
                                               byte_stride, slice_info.offset));
          byte_stride = internal::wrap_on_overflow::Multiply(byte_stride,
                                                             slice_info.stride);
          if (input_domain.shape()[input_dim] == 1) {
            element_pointer_byte_offset = internal::wrap_on_overflow::Add(
                element_pointer_byte_offset,
                internal::wrap_on_overflow::Multiply(
                    byte_stride, input_domain.origin()[input_dim]));
            byte_stride = 0;
          } else if (byte_stride != 0) {
            array_is_singleton = false;
          }
        }
        index_array_data.element_pointer =
            AddByteOffset(std::move(index_array_data.element_pointer),
                          element_pointer_byte_offset);
        if (array_is_singleton) {
          const Index index = *index_array_data.array_view(input_domain)
                                   .byte_strided_origin_pointer();
          const IndexInterval index_range = index_array_data.index_range;
          map.SetConstant();
          TENSORSTORE_RETURN_IF_ERROR(ReplaceZeroRankIndexArrayIndexMap(
              index, index_range, &map.offset(), &map.stride()));
        }
        break;
      }
    }
  }
  internal_index_space::DebugCheckInvariants(rep);
  return absl::OkStatus();
}
}  
Result<IndexTransform<>> ApplyIntervalSliceOp(
    IndexTransform<> transform, DimensionIndexBuffer* dimensions,
    IntervalForm interval_form, bool translate,
    IndexVectorOrScalarView start_vector,
    IndexVectorOrScalarView stop_or_size_vector,
    IndexVectorOrScalarView stride_vector, bool domain_only) {
  const DimensionIndex num_dims = dimensions->size();
  const DimensionIndex input_rank = transform.input_rank();
  TENSORSTORE_RETURN_IF_ERROR(CheckIndexVectorSize(start_vector, num_dims));
  TENSORSTORE_RETURN_IF_ERROR(
      CheckIndexVectorSize(stop_or_size_vector, num_dims));
  TENSORSTORE_RETURN_IF_ERROR(CheckIndexVectorSize(stride_vector, num_dims));
  TransformRep::Ptr<> rep = MutableRep(
      TransformAccess::rep_ptr<container>(std::move(transform)), domain_only);
  InputDimensionIntervalSliceInfo input_dimension_info[kMaxRank];
  TENSORSTORE_RETURN_IF_ERROR(
      GetIntervalSliceInfo(span(input_dimension_info).first(input_rank),
                           rep.get(), *dimensions, interval_form, translate,
                           start_vector, stop_or_size_vector, stride_vector));
  TENSORSTORE_RETURN_IF_ERROR(ApplyOffsetsAndStridesToOutputIndexMaps(
      rep.get(), span(input_dimension_info).first(input_rank)));
  return TransformAccess::Make<IndexTransform<>>(std::move(rep));
}
Result<IndexTransform<>> ApplyStrideOp(IndexTransform<> transform,
                                       DimensionIndexBuffer* dimensions,
                                       IndexVectorOrScalarView strides,
                                       bool domain_only) {
  const DimensionIndex num_dims = dimensions->size();
  const DimensionIndex input_rank = transform.input_rank();
  TENSORSTORE_RETURN_IF_ERROR(CheckIndexVectorSize(strides, num_dims));
  TransformRep::Ptr<> rep = MutableRep(
      TransformAccess::rep_ptr<container>(std::move(transform)), domain_only);
  InputDimensionIntervalSliceInfo input_dimension_info[kMaxRank];
  std::fill_n(&input_dimension_info[0], input_rank,
              InputDimensionIntervalSliceInfo{0, 1});
  const auto compute_input_domain = [&](DimensionIndex i,
                                        DimensionIndex input_dim) {
    const Index stride = strides[i];
    if (stride == 0) {
      return absl::InvalidArgumentError("Stride must be non-zero");
    }
    input_dimension_info[input_dim].stride = stride;
    const InputDimensionRef d = rep->input_dimension(input_dim);
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto new_domain,
        GetAffineTransformDomain(d.optionally_implicit_domain(), 0,
                                 stride));
    d.domain() = new_domain.interval();
    d.implicit_lower_bound() = new_domain.implicit_lower();
    d.implicit_upper_bound() = new_domain.implicit_upper();
    return absl::OkStatus();
  };
  for (DimensionIndex i = 0; i < num_dims; ++i) {
    const DimensionIndex input_dim = (*dimensions)[i];
    TENSORSTORE_RETURN_IF_ERROR(
        compute_input_domain(i, input_dim),
        MaybeAnnotateStatus(
            _, tensorstore::StrCat("Applying stride to input dimension ",
                                   input_dim)));
  }
  TENSORSTORE_RETURN_IF_ERROR(ApplyOffsetsAndStridesToOutputIndexMaps(
      rep.get(), span(input_dimension_info).first(input_rank)));
  return TransformAccess::Make<IndexTransform<>>(std::move(rep));
}
}  
}  