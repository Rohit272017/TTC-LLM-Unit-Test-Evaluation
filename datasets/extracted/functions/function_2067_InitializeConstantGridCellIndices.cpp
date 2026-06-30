#include "tensorstore/internal/grid_partition.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>
#include "absl/container/fixed_array.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/index_space/output_index_map.h"
#include "tensorstore/index_space/output_index_method.h"
#include "tensorstore/internal/grid_partition_impl.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/dimension_set.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
using ::tensorstore::internal::OutputToGridCellFn;
using ::tensorstore::internal_index_space::TransformAccess;
namespace tensorstore {
namespace internal_grid_partition {
namespace {
using IndexArraySet = IndexTransformGridPartition::IndexArraySet;
using StridedSet = IndexTransformGridPartition::StridedSet;
struct ConnectedSetIterateParameters {
  const IndexTransformGridPartition& info;
  tensorstore::span<const DimensionIndex> grid_output_dimensions;
  OutputToGridCellFn output_to_grid_cell;
  IndexTransformView<> transform;
  absl::FunctionRef<absl::Status(
      tensorstore::span<const Index> grid_cell_indices,
      IndexTransformView<> cell_transform)>
      func;
};
void InitializeConstantGridCellIndices(
    IndexTransformView<> transform,
    tensorstore::span<const DimensionIndex> grid_output_dimensions,
    OutputToGridCellFn output_to_grid_cell,
    tensorstore::span<Index> grid_cell_indices) {
  for (DimensionIndex grid_dim = 0; grid_dim < grid_output_dimensions.size();
       ++grid_dim) {
    const DimensionIndex output_dim = grid_output_dimensions[grid_dim];
    const OutputIndexMapRef<> map = transform.output_index_map(output_dim);
    if (map.method() != OutputIndexMethod::constant) continue;
    grid_cell_indices[grid_dim] =
        output_to_grid_cell(grid_dim, map.offset(), nullptr);
  }
}
class StridedSetGridCellIterator {
 public:
  explicit StridedSetGridCellIterator(
      IndexTransformView<> transform,
      tensorstore::span<const DimensionIndex> grid_output_dimensions,
      OutputToGridCellFn output_to_grid_cell, StridedSet strided_set)
      : transform_(transform),
        grid_output_dimensions_(grid_output_dimensions),
        output_to_grid_cell_(output_to_grid_cell),
        strided_set_(strided_set) {
    Reset();
  }
  void Reset() {
    const IndexInterval domain =
        transform_.input_domain()[strided_set_.input_dimension];
    input_end_index_ = domain.exclusive_max();
    input_index_ = domain.inclusive_min();
  }
  bool AtEnd() const { return input_index_ == input_end_index_; }
  IndexInterval Next(tensorstore::span<Index> output_grid_cell_indices) {
    assert(!AtEnd());
    IndexInterval restricted_domain =
        IndexInterval::UncheckedHalfOpen(input_index_, input_end_index_);
    for (const DimensionIndex grid_dim :
         strided_set_.grid_dimensions.index_view()) {
      const DimensionIndex output_dim = grid_output_dimensions_[grid_dim];
      const OutputIndexMapRef<> map = transform_.output_index_map(output_dim);
      IndexInterval cell_range;
      output_grid_cell_indices[grid_dim] = output_to_grid_cell_(
          grid_dim, input_index_ * map.stride() + map.offset(), &cell_range);
      const IndexInterval cell_domain =
          GetAffineTransformDomain(cell_range, map.offset(), map.stride())
              .value();
      restricted_domain = Intersect(restricted_domain, cell_domain);
    }
    assert(!restricted_domain.empty());
    input_index_ = restricted_domain.exclusive_max();
    return restricted_domain;
  }
 private:
  IndexTransformView<> transform_;
  tensorstore::span<const DimensionIndex> grid_output_dimensions_;
  OutputToGridCellFn output_to_grid_cell_;
  StridedSet strided_set_;
  Index input_end_index_;
  Index input_index_;
};
class IndexArraySetIterator {
 public:
  IndexArraySetIterator(const IndexArraySet& index_array_set)
      : grid_dimensions_(index_array_set.grid_dimensions),
        grid_cell_indices_(index_array_set.grid_cell_indices),
        partition_end_index_(index_array_set.num_partitions()),
        partition_index_(0) {}
  void Reset() { partition_index_ = 0; }
  bool AtEnd() const { return partition_index_ == partition_end_index_; }
  Index Next(tensorstore::span<Index> output_grid_cell_indices) {
    assert(!AtEnd());
    const Index grid_cell_indices_offset =
        partition_index_ * grid_dimensions_.count();
    DimensionIndex grid_i = 0;
    for (DimensionIndex grid_dim : grid_dimensions_.index_view()) {
      output_grid_cell_indices[grid_dim] =
          grid_cell_indices_[grid_cell_indices_offset + grid_i++];
    }
    return partition_index_++;
  }
 private:
  DimensionSet grid_dimensions_;
  tensorstore::span<const Index> grid_cell_indices_;
  Index partition_end_index_;
  Index partition_index_;
};
class ConnectedSetIterateHelper {
 public:
  explicit ConnectedSetIterateHelper(ConnectedSetIterateParameters params)
      : params_(std::move(params)),
        grid_cell_indices_(params_.grid_output_dimensions.size()),
        cell_transform_(internal_grid_partition::InitializeCellTransform(
            params_.info, params_.transform)) {
    InitializeConstantGridCellIndices(
        params_.transform, params_.grid_output_dimensions,
        params_.output_to_grid_cell, grid_cell_indices_);
  }
  absl::Status Iterate() { return IterateOverIndexArraySets(0); }
 private:
  absl::Status IterateOverIndexArraySets(DimensionIndex set_i) {
    if (set_i == params_.info.index_array_sets().size()) {
      return IterateOverStridedSets(0);
    }
    const IndexArraySet& index_array_set =
        params_.info.index_array_sets()[set_i];
    IndexArraySetIterator iterator(index_array_set);
    while (!iterator.AtEnd()) {
      Index partition_i = iterator.Next(grid_cell_indices_);
      UpdateCellTransformForIndexArraySetPartition(
          index_array_set, set_i, partition_i, cell_transform_.get());
      TENSORSTORE_RETURN_IF_ERROR(IterateOverIndexArraySets(set_i + 1));
    }
    return absl::OkStatus();
  }
  absl::Status IterateOverStridedSets(DimensionIndex set_i) {
    if (set_i == params_.info.strided_sets().size()) return InvokeCallback();
    StridedSetGridCellIterator iterator(
        params_.transform, params_.grid_output_dimensions,
        params_.output_to_grid_cell, params_.info.strided_sets()[set_i]);
    const DimensionIndex cell_input_dim =
        set_i + params_.info.index_array_sets().size();
    while (!iterator.AtEnd()) {
      auto restricted_domain = iterator.Next(grid_cell_indices_);
      cell_transform_->input_origin()[cell_input_dim] =
          restricted_domain.inclusive_min();
      cell_transform_->input_shape()[cell_input_dim] = restricted_domain.size();
      TENSORSTORE_RETURN_IF_ERROR(IterateOverStridedSets(set_i + 1));
    }
    return absl::OkStatus();
  }
  absl::Status InvokeCallback() {
    internal_index_space::DebugCheckInvariants(cell_transform_.get());
    auto status = params_.func(
        grid_cell_indices_,
        TransformAccess::Make<IndexTransformView<>>(cell_transform_.get()));
    cell_transform_ = MutableRep(std::move(cell_transform_));
    return status;
  }
  ConnectedSetIterateParameters params_;
  absl::FixedArray<Index, internal::kNumInlinedDims> grid_cell_indices_;
  internal_index_space::TransformRep::Ptr<> cell_transform_;
};
bool GetStridedGridCellRanges(
    IndexTransformView<> transform, OutputToGridCellFn output_to_grid_cell,
    DimensionIndex grid_dim, DimensionIndex output_dim,
    absl::FunctionRef<bool(IndexInterval grid_cell_range)> callback) {
  const auto output_map = transform.output_index_maps()[output_dim];
  assert(output_map.method() == OutputIndexMethod::single_input_dimension);
  const Index output_offset = output_map.offset();
  const Index output_stride = output_map.stride();
  const DimensionIndex input_dim = output_map.input_dimension();
  const IndexInterval input_domain = transform.domain().box()[input_dim];
  if (output_map.stride() == 1 || output_map.stride() == -1) {
    auto output_range = tensorstore::GetAffineTransformRange(
                            input_domain, output_offset, output_stride)
                            .value();
    Index min_cell_index =
        output_to_grid_cell(grid_dim, output_range.inclusive_min(), nullptr);
    Index max_cell_index =
        output_to_grid_cell(grid_dim, output_range.inclusive_max(), nullptr);
    return callback(
        IndexInterval::UncheckedClosed(min_cell_index, max_cell_index));
  }
  IndexInterval prev_interval;
  for (Index input_index = input_domain.inclusive_min();
       input_index < input_domain.exclusive_max();) {
    IndexInterval output_range;
    Index grid_cell = output_to_grid_cell(
        grid_dim, input_index * output_stride + output_offset, &output_range);
    const IndexInterval cell_domain =
        GetAffineTransformDomain(output_range, output_offset, output_stride)
            .value();
    assert(!cell_domain.empty());
    if (grid_cell == prev_interval.exclusive_min() ||
        grid_cell == prev_interval.exclusive_max()) {
      prev_interval = IndexInterval::UncheckedClosed(
          std::min(prev_interval.inclusive_min(), grid_cell),
          std::max(prev_interval.inclusive_max(), grid_cell));
    } else {
      if (IsFinite(prev_interval)) {
        if (!callback(prev_interval)) return false;
      }
      prev_interval = IndexInterval::UncheckedClosed(grid_cell, grid_cell);
    }
    input_index = cell_domain.exclusive_max();
  }
  return callback(prev_interval);
}
struct GetGridCellRangesIterateParameters {
  const IndexTransformGridPartition& info;
  tensorstore::span<const DimensionIndex> grid_output_dimensions;
  OutputToGridCellFn output_to_grid_cell;
  IndexTransformView<> transform;
  absl::FunctionRef<absl::Status(BoxView<> bounds)> func;
  DimensionIndex outer_prefix_rank;
  BoxView<> grid_bounds;
  tensorstore::span<const IndexInterval> inner_intervals;
  tensorstore::span<const StridedSet*> strided_sets_in_prefix;
  tensorstore::span<const IndexArraySet*> index_array_sets_in_prefix;
};
class GetGridCellRangesIterateHelper {
 public:
  explicit GetGridCellRangesIterateHelper(
      GetGridCellRangesIterateParameters params)
      : params_(params) {
    InitializeConstantGridCellIndices(
        params_.transform, params_.grid_output_dimensions,
        params_.output_to_grid_cell,
        tensorstore::span<Index>(&grid_bounds_origin_[0],
                                 params_.transform.output_rank()));
    for (DimensionIndex i = 0; i < params.outer_prefix_rank; ++i) {
      grid_bounds_shape_[i] = 1;
    }
    for (DimensionIndex i = params.outer_prefix_rank + 1,
                        rank = params.grid_bounds.rank();
         i < rank; ++i) {
      grid_bounds_origin_[i] = params.grid_bounds.origin()[i];
      grid_bounds_shape_[i] = params.grid_bounds.shape()[i];
    }
    if (params.inner_intervals.size() == 1) {
      const auto& inner_interval = params.inner_intervals[0];
      grid_bounds_origin_[params.outer_prefix_rank] =
          inner_interval.inclusive_min();
      grid_bounds_shape_[params.outer_prefix_rank] = inner_interval.size();
    }
  }
  absl::Status Iterate() { return IterateOverIndexArraySets(0); }
 private:
  GetGridCellRangesIterateParameters params_;
  Index grid_bounds_origin_[kMaxRank];
  Index grid_bounds_shape_[kMaxRank];
  absl::Status IterateOverIndexArraySets(DimensionIndex set_i) {
    if (set_i == params_.index_array_sets_in_prefix.size()) {
      return IterateOverStridedSets(0);
    }
    const IndexArraySet& index_array_set =
        *params_.index_array_sets_in_prefix[set_i];
    const auto grid_dimensions = index_array_set.grid_dimensions;
    const DimensionIndex num_grid_dimensions = grid_dimensions.count();
    for (Index partition_i = 0,
               num_partitions = index_array_set.num_partitions();
         partition_i < num_partitions; ++partition_i) {
      const Index grid_cell_indices_offset = partition_i * num_grid_dimensions;
      DimensionIndex grid_i = 0;
      for (DimensionIndex grid_dim : grid_dimensions.index_view()) {
        grid_bounds_origin_[grid_dim] =
            index_array_set
                .grid_cell_indices[grid_cell_indices_offset + grid_i++];
      }
      TENSORSTORE_RETURN_IF_ERROR(IterateOverIndexArraySets(set_i + 1));
    }
    return absl::OkStatus();
  }
  absl::Status IterateOverStridedSets(DimensionIndex set_i) {
    if (set_i == params_.strided_sets_in_prefix.size()) return InvokeCallback();
    StridedSetGridCellIterator iterator(
        params_.transform, params_.grid_output_dimensions,
        params_.output_to_grid_cell, *params_.strided_sets_in_prefix[set_i]);
    while (!iterator.AtEnd()) {
      iterator.Next(grid_bounds_origin_);
      TENSORSTORE_RETURN_IF_ERROR(IterateOverStridedSets(set_i + 1));
    }
    return absl::OkStatus();
  }
  absl::Status InvokeCallback() {
    MutableBoxView<> bounds(params_.grid_bounds.rank(), grid_bounds_origin_,
                            grid_bounds_shape_);
    if (params_.inner_intervals.size() == 1) {
      return params_.func(bounds);
    }
    DimensionIndex outer_prefix_rank = params_.outer_prefix_rank;
    for (const auto& inner_interval : params_.inner_intervals) {
      bounds[outer_prefix_rank] = inner_interval;
      TENSORSTORE_RETURN_IF_ERROR(params_.func(bounds));
    }
    return absl::OkStatus();
  }
};
}  
}  
namespace internal {
absl::Status PartitionIndexTransformOverGrid(
    tensorstore::span<const DimensionIndex> grid_output_dimensions,
    OutputToGridCellFn output_to_grid_cell, IndexTransformView<> transform,
    absl::FunctionRef<
        absl::Status(tensorstore::span<const Index> grid_cell_indices,
                     IndexTransformView<> cell_transform)>
        func) {
  internal_grid_partition::IndexTransformGridPartition partition_info;
  auto status = internal_grid_partition::PrePartitionIndexTransformOverGrid(
      transform, grid_output_dimensions, output_to_grid_cell, partition_info);
  if (!status.ok()) return status;
  return internal_grid_partition::ConnectedSetIterateHelper(
             {partition_info,
              grid_output_dimensions,
              output_to_grid_cell,
              transform,
              std::move(func)})
      .Iterate();
}
}  
namespace internal_grid_partition {
absl::Status GetGridCellRanges(
    const IndexTransformGridPartition& grid_partition,
    tensorstore::span<const DimensionIndex> grid_output_dimensions,
    BoxView<> grid_bounds, OutputToGridCellFn output_to_grid_cell,
    IndexTransformView<> transform,
    absl::FunctionRef<absl::Status(BoxView<> bounds)> callback) {
  assert(grid_output_dimensions.size() == grid_bounds.rank());
  if (transform.domain().box().is_empty()) {
    return absl::OkStatus();
  }
  if (grid_output_dimensions.empty()) {
    return callback({});
  }
  std::array<DimensionIndex, kMaxRank> dim_to_indexed_set;
  dim_to_indexed_set.fill(-1);
  DimensionSet one_to_one_grid_dims;
  for (const auto& strided_set : grid_partition.strided_sets()) {
    if (strided_set.grid_dimensions.count() != 1) {
      continue;
    }
    const DimensionIndex grid_dim =
        strided_set.grid_dimensions.index_view().front();
    one_to_one_grid_dims[grid_dim] = true;
  }
  for (size_t i = 0; i < grid_partition.index_array_sets().size(); ++i) {
    const auto& set = grid_partition.index_array_sets()[i];
    if (set.grid_dimensions.count() != 1) {
      continue;
    }
    const DimensionIndex grid_dim = set.grid_dimensions.index_view().front();
    one_to_one_grid_dims[grid_dim] = true;
    dim_to_indexed_set[grid_dim] = i;
  }
  absl::InlinedVector<IndexInterval, 1> inner_intervals;
  DimensionSet grid_dimensions_outside_prefix;
  DimensionIndex range_queryable_grid_dim = grid_output_dimensions.size() - 1;
  for (; range_queryable_grid_dim >= 0; --range_queryable_grid_dim) {
    const DimensionIndex grid_dim = range_queryable_grid_dim;
    const IndexInterval grid_interval = grid_bounds[grid_dim];
    if (grid_interval.size() == 1) {
      inner_intervals.clear();
      inner_intervals.push_back(grid_interval);
      continue;
    }
    if (!one_to_one_grid_dims[grid_dim]) {
      break;
    }
    grid_dimensions_outside_prefix[grid_dim] = true;
    const DimensionIndex output_dim = grid_output_dimensions[grid_dim];
    inner_intervals.clear();
    DimensionIndex indexed_set_i = dim_to_indexed_set[grid_dim];
    if (indexed_set_i == -1) {
      internal_grid_partition::GetStridedGridCellRanges(
          transform, output_to_grid_cell, grid_dim, output_dim,
          [&](IndexInterval grid_cell_range) {
            inner_intervals.push_back(grid_cell_range);
            return true;
          });
    } else {
      const auto& set = grid_partition.index_array_sets()[indexed_set_i];
      const auto& grid_cell_indices = set.grid_cell_indices;
      size_t i = 0;
      while (i < grid_cell_indices.size()) {
        size_t last_i = i;
        while (last_i + 1 < grid_cell_indices.size() &&
               grid_cell_indices[last_i] + 1 == grid_cell_indices[last_i + 1]) {
          ++last_i;
        }
        inner_intervals.push_back(IndexInterval::UncheckedClosed(
            grid_cell_indices[i], grid_cell_indices[last_i]));
        i = last_i + 1;
      }
    }
    if (inner_intervals.size() == 1 &&
        tensorstore::Contains(inner_intervals[0], grid_interval)) {
      inner_intervals.clear();
      inner_intervals.push_back(grid_interval);
      continue;
    }
    --range_queryable_grid_dim;
    break;
  }
  const StridedSet* strided_sets_in_prefix_storage[kMaxRank];
  const IndexArraySet* index_array_sets_in_prefix_storage[kMaxRank];
  const auto get_sets_in_prefix = [&](auto sets, auto* buffer) {
    ptrdiff_t i = 0;
    for (const auto& set : sets) {
      if (grid_dimensions_outside_prefix[set.grid_dimensions.index_view()
                                             .front()]) {
        continue;
      }
      buffer[i++] = &set;
    }
    return tensorstore::span(buffer, i);
  };
  auto strided_sets_in_prefix = get_sets_in_prefix(
      grid_partition.strided_sets(), strided_sets_in_prefix_storage);
  auto index_array_sets_in_prefix = get_sets_in_prefix(
      grid_partition.index_array_sets(), index_array_sets_in_prefix_storage);
  if (range_queryable_grid_dim == grid_output_dimensions.size() - 1) {
    inner_intervals.push_back(grid_bounds[range_queryable_grid_dim]);
  }
  internal_grid_partition::GetGridCellRangesIterateHelper iterate_helper(
      internal_grid_partition::GetGridCellRangesIterateParameters{
          grid_partition, grid_output_dimensions, output_to_grid_cell,
          transform, callback, range_queryable_grid_dim + 1, grid_bounds,
          inner_intervals, strided_sets_in_prefix, index_array_sets_in_prefix});
  return iterate_helper.Iterate();
}
}  
namespace internal {
absl::Status GetGridCellRanges(
    tensorstore::span<const DimensionIndex> grid_output_dimensions,
    BoxView<> grid_bounds, OutputToGridCellFn output_to_grid_cell,
    IndexTransformView<> transform,
    absl::FunctionRef<absl::Status(BoxView<> bounds)> callback) {
  using internal_grid_partition::StridedSet;
  assert(grid_output_dimensions.size() == grid_bounds.rank());
  if (transform.domain().box().is_empty()) {
    return absl::OkStatus();
  }
  if (grid_output_dimensions.empty()) {
    return callback({});
  }
  internal_grid_partition::IndexTransformGridPartition grid_partition;
  TENSORSTORE_RETURN_IF_ERROR(
      internal_grid_partition::PrePartitionIndexTransformOverGrid(
          transform, grid_output_dimensions, output_to_grid_cell,
          grid_partition));
  return internal_grid_partition::GetGridCellRanges(
      grid_partition, grid_output_dimensions, grid_bounds, output_to_grid_cell,
      transform, callback);
}
}  
}  