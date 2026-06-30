#include "tensorstore/internal/grid_partition_impl.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/index_space/output_index_map.h"
#include "tensorstore/index_space/output_index_method.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/rank.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/byte_strided_pointer.h"
#include "tensorstore/util/dimension_set.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/iterate_over_index_range.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"
namespace tensorstore {
namespace internal_grid_partition {
using ::tensorstore::internal_index_space::OutputIndexMap;
using ::tensorstore::internal_index_space::TransformRep;
using IndexArraySet = IndexTransformGridPartition::IndexArraySet;
using StridedSet = IndexTransformGridPartition::StridedSet;
using OutputToGridCellFn = absl::FunctionRef<Index(
    DimensionIndex grid_dim, Index output_index, IndexInterval* cell_bounds)>;
SharedArray<const Index, 2>
IndexTransformGridPartition::IndexArraySet::partition_input_indices(
    Index partition_i) const {
  assert(partition_i >= 0 && partition_i < num_partitions());
  SharedArray<const Index, 2> result;
  const Index start = grid_cell_partition_offsets[partition_i];
  const Index end =
      static_cast<size_t>(partition_i + 1) == grid_cell_partition_offsets.size()
          ? partitioned_input_indices.shape()[0]
          : grid_cell_partition_offsets[partition_i + 1];
  assert(start >= 0 && start < partitioned_input_indices.shape()[0]);
  assert(end > start && end <= partitioned_input_indices.shape()[0]);
  result.pointer() =
      std::shared_ptr<const Index>(partitioned_input_indices.pointer(),
                                   &partitioned_input_indices(start, 0));
  result.layout() = partitioned_input_indices.layout();
  result.shape()[0] = end - start;
  return result;
}
tensorstore::span<const Index>
IndexTransformGridPartition::IndexArraySet::partition_grid_cell_indices(
    Index partition_i) const {
  assert(partition_i >= 0 && partition_i < num_partitions());
  assert(grid_cell_indices.size() ==
         static_cast<size_t>(num_partitions() * grid_dimensions.count()));
  return tensorstore::span(
      &grid_cell_indices[partition_i * grid_dimensions.count()],
      grid_dimensions.count());
}
namespace {
struct GridCellIndicesIndirectPartialCompare {
  DimensionSet grid_dimensions;
  const Index* grid_cell_indices_for_partitions;
  Index operator()(Index partition_i, const Index* full_indices) const {
    const Index* other_grid_cell_indices =
        grid_cell_indices_for_partitions +
        partition_i * grid_dimensions.count();
    DimensionIndex j = 0;
    for (DimensionIndex grid_dim : grid_dimensions.index_view()) {
      Index diff = other_grid_cell_indices[j] - full_indices[grid_dim];
      if (diff != 0) {
        return diff;
      }
      ++j;
    }
    return 0;
  }
};
}  
Index IndexTransformGridPartition::IndexArraySet::FindPartition(
    tensorstore::span<const Index> grid_cell_indices) const {
  Index lower = 0, upper = num_partitions();
  GridCellIndicesIndirectPartialCompare compare{grid_dimensions,
                                                this->grid_cell_indices.data()};
  while (lower != upper) {
    Index mid = (lower + upper) / 2;
    Index c = compare(mid, grid_cell_indices.data());
    if (c == 0) return mid;
    if (c > 0) {
      upper = mid;
    } else {
      lower = mid + 1;
    }
  }
  return -1;
}
void UpdateCellTransformForIndexArraySetPartition(
    const IndexArraySet& index_array_set, DimensionIndex set_i,
    Index partition_i, internal_index_space::TransformRep* cell_transform) {
  const SharedArray<const Index, 2> partition_input_indices =
      index_array_set.partition_input_indices(partition_i);
  cell_transform->input_shape()[set_i] = partition_input_indices.shape()[0];
  ByteStridedPointer<const Index> partition_input_indices_ptr =
      partition_input_indices.byte_strided_pointer();
  const Index vector_dimension_byte_stride =
      partition_input_indices.byte_strides()[1];
  const tensorstore::span<OutputIndexMap> output_maps =
      cell_transform->output_index_maps();
  for (DimensionIndex full_input_dim :
       index_array_set.input_dimensions.index_view()) {
    internal_index_space::IndexArrayData& index_array_data =
        output_maps[full_input_dim].index_array_data();
    index_array_data.element_pointer = std::shared_ptr<const Index>(
        partition_input_indices.pointer(), partition_input_indices_ptr);
    partition_input_indices_ptr += vector_dimension_byte_stride;
  }
}
IndexTransform<> IndexTransformGridPartition::GetCellTransform(
    IndexTransformView<> full_transform,
    tensorstore::span<const Index> grid_cell_indices,
    tensorstore::span<const DimensionIndex> grid_output_dimensions,
    absl::FunctionRef<IndexInterval(DimensionIndex grid_dim,
                                    Index grid_cell_index)>
        get_grid_cell_output_interval) const {
  auto cell_transform = InitializeCellTransform(*this, full_transform);
  for (DimensionIndex set_i = 0, num_sets = index_array_sets().size();
       set_i < num_sets; ++set_i) {
    const IndexArraySet& index_array_set = index_array_sets()[set_i];
    const Index partition_i = index_array_set.FindPartition(grid_cell_indices);
    assert(partition_i != -1);
    UpdateCellTransformForIndexArraySetPartition(
        index_array_set, set_i, partition_i, cell_transform.get());
  }
  for (DimensionIndex set_i = 0, num_sets = strided_sets().size();
       set_i < num_sets; ++set_i) {
    const StridedSet& strided_set = strided_sets()[set_i];
    const DimensionIndex cell_input_dim = set_i + index_array_sets().size();
    IndexInterval restricted_domain =
        full_transform.input_domain()[strided_set.input_dimension];
    for (const DimensionIndex grid_dim :
         strided_set.grid_dimensions.index_view()) {
      const DimensionIndex output_dim = grid_output_dimensions[grid_dim];
      IndexInterval cell_range =
          get_grid_cell_output_interval(grid_dim, grid_cell_indices[grid_dim]);
      const OutputIndexMapRef<> map =
          full_transform.output_index_map(output_dim);
      const IndexInterval cell_domain =
          GetAffineTransformDomain(cell_range, map.offset(), map.stride())
              .value();
      restricted_domain = Intersect(restricted_domain, cell_domain);
    }
    assert(!restricted_domain.empty());
    cell_transform->input_origin()[cell_input_dim] =
        restricted_domain.inclusive_min();
    cell_transform->input_shape()[cell_input_dim] = restricted_domain.size();
  }
  return internal_index_space::TransformAccess::Make<IndexTransform<>>(
      std::move(cell_transform));
}
namespace {
template <typename SetCallbackFn>
void ForEachConnectedSet(
    tensorstore::span<const DimensionIndex> grid_output_dimensions,
    IndexTransformView<> transform, SetCallbackFn set_callback) {
  DimensionSet input_dims_for_grid_dims[kMaxRank];
  DimensionSet grid_dims_with_array_dependence;
  for (DimensionIndex grid_dim = 0; grid_dim < grid_output_dimensions.size();
       ++grid_dim) {
    auto [input_dims, array_dependence] =
        internal::GetInputDimensionsForOutputDimension(
            transform, grid_output_dimensions[grid_dim]);
    input_dims_for_grid_dims[grid_dim] = input_dims;
    grid_dims_with_array_dependence[grid_dim] = array_dependence;
  }
  DimensionSet current_input_dims, current_grid_dims;
  DimensionSet remaining_grid_dims{
      DimensionSet::UpTo(grid_output_dimensions.size())};
  bool current_set_has_array;
  const auto add_grid_dim_to_current_set =
      [&](DimensionIndex grid_dim) -> DimensionSet {
    assert(remaining_grid_dims.test(grid_dim));
    assert(grid_dim >= 0 && grid_dim < grid_output_dimensions.size());
    remaining_grid_dims.reset(grid_dim);
    current_grid_dims.set(grid_dim);
    auto input_dims = input_dims_for_grid_dims[grid_dim];
    current_set_has_array |= grid_dims_with_array_dependence[grid_dim];
    current_input_dims |= input_dims;
    return input_dims;
  };
  const auto is_grid_dim_in_set =
      [&](DimensionIndex grid_dim) -> DimensionIndex {
    assert(remaining_grid_dims.test(grid_dim));
    assert(grid_dim >= 0 && grid_dim < grid_output_dimensions.size());
    return !(input_dims_for_grid_dims[grid_dim] & current_input_dims).none();
  };
  while (!remaining_grid_dims.none()) {
    current_input_dims = {};
    current_grid_dims = {};
    current_set_has_array = false;
    if (add_grid_dim_to_current_set(remaining_grid_dims.index_view().front())
            .none()) {
      continue;
    }
    for (DimensionIndex grid_dim : remaining_grid_dims.index_view()) {
      if (is_grid_dim_in_set(grid_dim)) {
        add_grid_dim_to_current_set(grid_dim);
      }
    }
    set_callback(current_input_dims, current_grid_dims, current_set_has_array);
  }
}
template <typename T, typename Stride, typename OutputIt, typename OutputStride>
OutputIt FillWithTiledStridedRange(T start, T size, Stride stride,
                                   Index outer_count, Index inner_count,
                                   OutputIt output,
                                   OutputStride output_stride) {
  const T end = start + size * stride;
  for (Index outer_i = 0; outer_i < outer_count; ++outer_i) {
    for (Index i = start; i != end; i += stride) {
      for (Index inner_i = 0; inner_i < inner_count; ++inner_i) {
        *output = i;
        output += output_stride;
      }
    }
  }
  return output;
}
absl::Status GenerateSingleInputDimensionOutputIndices(
    OutputIndexMapRef<> map, DimensionSet input_dims,
    IndexTransformView<> index_transform, Index* output_indices,
    Index output_stride) {
  assert(map.method() == OutputIndexMethod::single_input_dimension);
  const DimensionIndex single_input_dim = map.input_dimension();
  const IndexInterval domain = index_transform.input_domain()[single_input_dim];
  const Index stride = map.stride();
  TENSORSTORE_RETURN_IF_ERROR(
      GetAffineTransformRange(domain, map.offset(), stride));
  const Index start = map.offset() + stride * domain.inclusive_min();
  tensorstore::span<const Index> input_shape = index_transform.input_shape();
  Index inner_count = 1;
  Index outer_count = 1;
  for (DimensionIndex input_dim : input_dims.index_view()) {
    if (input_dim == single_input_dim) {
      outer_count = inner_count;
      inner_count = 1;
    } else {
      inner_count *= input_shape[input_dim];
    }
  }
  FillWithTiledStridedRange(start, domain.size(), stride, outer_count,
                            inner_count, output_indices, output_stride);
  return absl::OkStatus();
}
absl::Status GenerateIndexArrayOutputIndices(
    OutputIndexMapRef<> map, DimensionSet input_dims,
    IndexTransformView<> index_transform, Index* output_indices,
    Index output_stride) {
  assert(map.method() == OutputIndexMethod::array);
  const DimensionIndex input_rank = index_transform.input_rank();
  Index output_byte_strides[kMaxRank];
  std::fill_n(&output_byte_strides[0], input_rank, static_cast<Index>(0));
  DimensionIndex byte_stride = sizeof(Index) * output_stride;
  Index input_dims_copy[kMaxRank];
  DimensionIndex num_input_dims = 0;
  for (DimensionIndex input_dim : input_dims.index_view()) {
    input_dims_copy[num_input_dims++] = input_dim;
  }
  for (DimensionIndex i = num_input_dims - 1; i >= 0; --i) {
    const DimensionIndex input_dim = input_dims_copy[i];
    output_byte_strides[input_dim] = byte_stride;
    byte_stride *= index_transform.input_shape()[input_dim];
  }
  const OutputIndexMapRef<>::IndexArrayView index_array = map.index_array();
  TENSORSTORE_RETURN_IF_ERROR(ValidateIndexArrayBounds(
      index_array.index_range(), index_array.array_ref()));
  const Index stride = map.stride();
  const Index offset = map.offset();
  IterateOverArrays(
      [stride, offset](const Index* source_ptr, Index* output_ptr) {
        const Index source_index = *source_ptr;
        *output_ptr = source_index * stride + offset;
        return true;
      },
      skip_repeated_elements,
      map.index_array().array_ref(),
      ArrayView<Index>(
          output_indices,
          StridedLayoutView<>(index_transform.input_shape(),
                              tensorstore::span<const Index>(
                                  &output_byte_strides[0], input_rank))));
  return absl::OkStatus();
}
Result<Index> ProductOfIndirectExtents(
    tensorstore::span<const Index> input_shape, DimensionSet dims) {
  Index num_positions = 1;
  for (const DimensionIndex dim : dims.index_view()) {
    if (internal::MulOverflow(num_positions, input_shape[dim],
                              &num_positions)) {
      return absl::InvalidArgumentError(
          "Overflow computing number of positions in domain.");
    }
  }
  return num_positions;
}
Result<std::vector<Index>> GenerateIndexArraySetGridCellIndices(
    DimensionSet grid_dims, DimensionSet input_dims,
    tensorstore::span<const DimensionIndex> grid_output_dimensions,
    OutputToGridCellFn output_to_grid_cell,
    IndexTransformView<> index_transform, Index num_positions) {
  const DimensionIndex num_grid_dims = grid_dims.count();
  std::vector<Index> temp_cell_indices(num_grid_dims * num_positions);
  DimensionIndex grid_i = 0;
  for (DimensionIndex grid_dim : grid_dims.index_view()) {
    const DimensionIndex output_dim = grid_output_dimensions[grid_dim];
    const OutputIndexMapRef<> map =
        index_transform.output_index_map(output_dim);
    Index* cur_cell_indices = temp_cell_indices.data() + grid_i;
    if (map.method() == OutputIndexMethod::single_input_dimension) {
      TENSORSTORE_RETURN_IF_ERROR(GenerateSingleInputDimensionOutputIndices(
          map, input_dims, index_transform, cur_cell_indices, num_grid_dims));
    } else {
      assert(map.method() == OutputIndexMethod::array);
      TENSORSTORE_RETURN_IF_ERROR(GenerateIndexArrayOutputIndices(
          map, input_dims, index_transform, cur_cell_indices, num_grid_dims));
    }
    for (Index* end = cur_cell_indices + num_positions * num_grid_dims;
         cur_cell_indices != end; cur_cell_indices += num_grid_dims) {
      *cur_cell_indices =
          output_to_grid_cell(grid_dim, *cur_cell_indices, nullptr);
    }
    ++grid_i;
  }
  return temp_cell_indices;
}
struct IndirectIndicesEqual {
  const Index* index_vectors;
  DimensionIndex num_dims;
  bool operator()(Index a, Index b) const {
    return std::equal(index_vectors + a * num_dims,
                      index_vectors + a * num_dims + num_dims,
                      index_vectors + b * num_dims);
  }
};
struct IndirectIndicesLess {
  const Index* index_vectors;
  DimensionIndex num_dims;
  bool operator()(Index a, Index b) const {
    return std::lexicographical_compare(
        index_vectors + a * num_dims, index_vectors + a * num_dims + num_dims,
        index_vectors + b * num_dims, index_vectors + b * num_dims + num_dims);
  }
};
struct IndirectHashIndices {
  const Index* index_vectors;
  DimensionIndex num_dims;
  size_t operator()(Index x) const {
    return absl::Hash<HashHelper>()(HashHelper{index_vectors, num_dims, x});
  }
 private:
  struct HashHelper {
    const Index* index_vectors;
    DimensionIndex num_dims;
    Index index;
    template <typename H>
    friend H AbslHashValue(H h, HashHelper x) {
      return H::combine_contiguous(
          std::move(h), x.index_vectors + x.index * x.num_dims, x.num_dims);
    }
  };
};
using IndirectVectorMap = absl::flat_hash_map<Index, Index, IndirectHashIndices,
                                              IndirectIndicesEqual>;
IndirectVectorMap PartitionIndexArraySetGridCellIndexVectors(
    const Index* temp_cell_indices, Index num_positions, Index num_grid_dims,
    std::vector<Index>* grid_cell_indices,
    std::vector<Index>* grid_cell_partition_offsets) {
  IndirectVectorMap cells(
      1, IndirectHashIndices{temp_cell_indices, num_grid_dims},
      IndirectIndicesEqual{temp_cell_indices, num_grid_dims});
  for (DimensionIndex i = 0; i < num_positions; ++i) {
    ++cells[i];
  }
  grid_cell_indices->resize(num_grid_dims * cells.size());
  grid_cell_partition_offsets->resize(cells.size());
  std::transform(cells.begin(), cells.end(),
                 grid_cell_partition_offsets->begin(),
                 [](IndirectVectorMap::const_reference x) { return x.first; });
  std::sort(grid_cell_partition_offsets->begin(),
            grid_cell_partition_offsets->end(),
            IndirectIndicesLess{temp_cell_indices, num_grid_dims});
  {
    Index offset = 0;
    Index* grid_cell_indices_ptr = grid_cell_indices->data();
    for (Index& position_i_or_offset : *grid_cell_partition_offsets) {
      const Index position_i = position_i_or_offset;
      auto it = cells.find(position_i);
      assert(it != cells.end());
      auto& count_or_offset = it->second;
      const Index count = count_or_offset;
      position_i_or_offset = count_or_offset = offset;
      offset += count;
      grid_cell_indices_ptr =
          std::copy_n(temp_cell_indices + position_i * num_grid_dims,
                      num_grid_dims, grid_cell_indices_ptr);
    }
  }
  return cells;
}
SharedArray<Index, 2> GenerateIndexArraySetPartitionedInputIndices(
    DimensionSet input_dims, BoxView<> full_input_domain,
    IndirectVectorMap cells, Index num_positions) {
  const DimensionIndex num_input_dims = input_dims.count();
  Box<dynamic_rank(internal::kNumInlinedDims)> partial_input_domain(
      num_input_dims);
  {
    DimensionIndex i = 0;
    for (DimensionIndex input_dim : input_dims.index_view()) {
      partial_input_domain[i] = full_input_domain[input_dim];
      ++i;
    }
  }
  SharedArray<Index, 2> partitioned_input_indices =
      AllocateArray<Index>({num_positions, num_input_dims});
  Index position_i = 0;
  IterateOverIndexRange(
      partial_input_domain, [&](tensorstore::span<const Index> indices) {
        auto it = cells.find(position_i);
        assert(it != cells.end());
        auto& offset = it->second;
        std::copy(indices.begin(), indices.end(),
                  partitioned_input_indices.data() + offset * num_input_dims);
        ++offset;
        ++position_i;
      });
  return partitioned_input_indices;
}
absl::Status FillIndexArraySetData(
    IndexTransformGridPartition::IndexArraySet& index_array_set,
    tensorstore::span<const DimensionIndex> grid_output_dimensions,
    OutputToGridCellFn output_to_grid_cell,
    IndexTransformView<> index_transform) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      Index num_positions,
      ProductOfIndirectExtents(index_transform.input_shape(),
                               index_array_set.input_dimensions));
  if (num_positions == 0) {
    return absl::OkStatus();
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      std::vector<Index> temp_cell_indices,
      GenerateIndexArraySetGridCellIndices(
          index_array_set.grid_dimensions, index_array_set.input_dimensions,
          grid_output_dimensions, output_to_grid_cell, index_transform,
          num_positions));
  IndirectVectorMap cells = PartitionIndexArraySetGridCellIndexVectors(
      temp_cell_indices.data(), num_positions,
      index_array_set.grid_dimensions.count(),
      &index_array_set.grid_cell_indices,
      &index_array_set.grid_cell_partition_offsets);
  index_array_set.partitioned_input_indices =
      GenerateIndexArraySetPartitionedInputIndices(
          index_array_set.input_dimensions, index_transform.domain().box(),
          std::move(cells), num_positions);
  return absl::OkStatus();
}
absl::Status GenerateIndexTransformGridPartitionData(
    tensorstore::span<const DimensionIndex> grid_output_dimensions,
    OutputToGridCellFn output_to_grid_cell,
    IndexTransformView<> index_transform,
    IndexTransformGridPartition& grid_partition) {
  IndexTransformGridPartition::StridedSet strided_sets[kMaxRank];
  DimensionIndex num_strided_sets = 0;
  std::pair<DimensionSet, DimensionSet> index_array_sets[kMaxRank];
  DimensionIndex num_index_array_sets = 0;
  ForEachConnectedSet(
      grid_output_dimensions, index_transform,
      [&](DimensionSet input_dims, DimensionSet grid_dims, bool has_array) {
        if (!has_array) {
          assert(input_dims.count() == 1);
          strided_sets[num_strided_sets++] = {grid_dims,
                                              input_dims.index_view().front()};
        } else {
          index_array_sets[num_index_array_sets++] = {grid_dims, input_dims};
        }
      });
  grid_partition.strided_sets_.assign(&strided_sets[0],
                                      &strided_sets[num_strided_sets]);
  grid_partition.index_array_sets_.resize(num_index_array_sets);
  for (DimensionIndex i = 0; i < num_index_array_sets; ++i) {
    auto& set = grid_partition.index_array_sets_[i];
    auto [grid_dims, input_dims] = index_array_sets[i];
    set.input_dimensions = input_dims;
    set.grid_dimensions = grid_dims;
    TENSORSTORE_RETURN_IF_ERROR(FillIndexArraySetData(
        set, grid_output_dimensions, output_to_grid_cell, index_transform));
  }
  return absl::OkStatus();
}
}  
internal_index_space::TransformRep::Ptr<> InitializeCellTransform(
    const IndexTransformGridPartition& info,
    IndexTransformView<> full_transform) {
  const DimensionIndex full_input_rank = full_transform.input_rank();
  DimensionIndex num_index_array_dims = 0;
  for (const IndexArraySet& index_array_set : info.index_array_sets()) {
    num_index_array_dims += index_array_set.input_dimensions.count();
  }
  const DimensionIndex cell_input_rank =
      full_input_rank - num_index_array_dims + info.index_array_sets().size();
  internal_index_space::TransformRep::Ptr<> cell_transform =
      TransformRep::Allocate(cell_input_rank, full_input_rank);
  cell_transform->input_rank = cell_input_rank;
  cell_transform->output_rank = full_input_rank;
  cell_transform->implicit_lower_bounds = false;
  cell_transform->implicit_upper_bounds = false;
  const tensorstore::span<Index> input_origin =
      cell_transform->input_origin().first(cell_input_rank);
  const tensorstore::span<OutputIndexMap> output_maps =
      cell_transform->output_index_maps().first(full_input_rank);
  {
    DimensionIndex cell_input_dim = 0;
    for (const IndexArraySet& index_array_set : info.index_array_sets()) {
      input_origin[cell_input_dim] = 0;
      for (const DimensionIndex full_input_dim :
           index_array_set.input_dimensions.index_view()) {
        auto& map = output_maps[full_input_dim];
        map.offset() = 0;
        map.stride() = 1;
        auto& index_array_data = map.SetArrayIndexing(cell_input_rank);
        std::fill_n(index_array_data.byte_strides, cell_input_rank, 0);
        index_array_data.byte_strides[cell_input_dim] =
            index_array_set.partitioned_input_indices.byte_strides()[0];
      }
      ++cell_input_dim;
    }
    for (const auto& strided_set : info.strided_sets()) {
      auto& map = output_maps[strided_set.input_dimension];
      map.SetSingleInputDimension(cell_input_dim);
      map.offset() = 0;
      map.stride() = 1;
      ++cell_input_dim;
    }
  }
  for (DimensionIndex cell_input_dim = info.index_array_sets().size() +
                                       info.strided_sets().size(),
                      full_input_dim = 0;
       full_input_dim < full_input_rank; ++full_input_dim) {
    auto& map = output_maps[full_input_dim];
    if (map.method() != OutputIndexMethod::constant) continue;
    map.SetSingleInputDimension(cell_input_dim);
    map.offset() = 0;
    map.stride() = 1;
    cell_transform->input_dimension(cell_input_dim) =
        internal_index_space::TransformAccess::rep(full_transform)
            ->input_dimension(full_input_dim);
    ++cell_input_dim;
  }
  return cell_transform;
}
absl::Status PrePartitionIndexTransformOverGrid(
    IndexTransformView<> index_transform,
    tensorstore::span<const DimensionIndex> grid_output_dimensions,
    OutputToGridCellFn output_to_grid_cell,
    IndexTransformGridPartition& grid_partition) {
  const DimensionIndex input_rank = index_transform.input_rank();
  for (DimensionIndex input_dim = 0; input_dim < input_rank; ++input_dim) {
    const IndexInterval domain = index_transform.input_domain()[input_dim];
    if (!IsFinite(domain)) {
      return absl::InvalidArgumentError(
          tensorstore::StrCat("Input dimension ", input_dim,
                              " has unbounded domain ", domain, "."));
    }
  }
  for (const DimensionIndex output_dim : grid_output_dimensions) {
    const OutputIndexMapRef<> map =
        index_transform.output_index_map(output_dim);
    if (map.method() != OutputIndexMethod::single_input_dimension) continue;
    auto status = GetAffineTransformRange(
                      index_transform.input_domain()[map.input_dimension()],
                      map.offset(), map.stride())
                      .status();
    if (!status.ok()) {
      return MaybeAnnotateStatus(
          status, tensorstore::StrCat("Computing range of output dimension ",
                                      output_dim));
    }
  }
  return internal_grid_partition::GenerateIndexTransformGridPartitionData(
      grid_output_dimensions, output_to_grid_cell, index_transform,
      grid_partition);
}
}  
}  