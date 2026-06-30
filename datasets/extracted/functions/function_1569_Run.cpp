#include "xla/service/cpu/shape_partition.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>
namespace xla {
namespace cpu {
std::vector<int64_t> ShapePartitionAssigner::Run(
    int64_t target_partition_count) {
  std::vector<int64_t> outer_dims;
  int64_t outer_dim_size = 1;
  for (int i = shape_.layout().minor_to_major_size() - 1; i >= 0; --i) {
    const int64_t dimension = shape_.layout().minor_to_major(i);
    outer_dims.push_back(dimension);
    outer_dim_size *= shape_.dimensions(dimension);
    if (outer_dim_size >= target_partition_count) {
      break;
    }
  }
  target_partition_count = std::min(outer_dim_size, target_partition_count);
  const int64_t target_dim_partition_count = std::pow(
      static_cast<double>(target_partition_count), 1.0 / outer_dims.size());
  std::vector<int64_t> dimension_partition_counts(outer_dims.size());
  for (int64_t i = 0; i < outer_dims.size(); ++i) {
    dimension_partition_counts[i] =
        std::min(static_cast<int64_t>(shape_.dimensions(outer_dims[i])),
                 target_dim_partition_count);
  }
  if (GetTotalPartitionCount(dimension_partition_counts) <
      target_partition_count) {
    for (int64_t i = 0; i < dimension_partition_counts.size(); ++i) {
      const int64_t current_dim_partition_count = dimension_partition_counts[i];
      const int64_t other_dims_partition_count =
          GetTotalPartitionCount(dimension_partition_counts) /
          current_dim_partition_count;
      int64_t additional_partition_count =
          target_partition_count / other_dims_partition_count -
          current_dim_partition_count;
      additional_partition_count = std::min(
          shape_.dimensions(outer_dims[i]) - dimension_partition_counts[i],
          additional_partition_count);
      if (additional_partition_count > 0) {
        dimension_partition_counts[i] += additional_partition_count;
      }
    }
  }
  return dimension_partition_counts;
}
int64_t ShapePartitionAssigner::GetTotalPartitionCount(
    const std::vector<int64_t>& dimension_partition_counts) {
  int64_t total_partition_count = 1;
  for (int64_t dim_partition_count : dimension_partition_counts) {
    total_partition_count *= dim_partition_count;
  }
  return total_partition_count;
}
ShapePartitionIterator::ShapePartitionIterator(
    const Shape& shape, absl::Span<const int64_t> dimension_partition_counts)
    : shape_(shape),
      dimension_partition_counts_(dimension_partition_counts.begin(),
                                  dimension_partition_counts.end()),
      dimensions_(dimension_partition_counts_.size()),
      dimension_partition_sizes_(dimension_partition_counts_.size()),
      dimension_partition_strides_(dimension_partition_counts_.size()) {
  for (int i = 0; i < dimensions_.size(); ++i) {
    dimensions_[i] = shape_.layout().minor_to_major(
        shape_.layout().minor_to_major_size() - 1 - i);
  }
  for (int i = 0; i < dimension_partition_sizes_.size(); ++i) {
    const int64_t dim_size = shape_.dimensions(dimensions_[i]);
    dimension_partition_sizes_[i] =
        std::max(int64_t{1}, dim_size / dimension_partition_counts_[i]);
  }
  dimension_partition_strides_[dimension_partition_strides_.size() - 1] = 1;
  for (int i = dimension_partition_strides_.size() - 2; i >= 0; --i) {
    dimension_partition_strides_[i] = dimension_partition_strides_[i + 1] *
                                      dimension_partition_counts_[i + 1];
  }
}
std::vector<std::pair<int64_t, int64_t>> ShapePartitionIterator::GetPartition(
    int64_t index) const {
  std::vector<std::pair<int64_t, int64_t>> partition(dimensions_.size());
  for (int64_t i = 0; i < partition.size(); ++i) {
    const int64_t partition_index = index / dimension_partition_strides_[i];
    partition[i].first = partition_index * dimension_partition_sizes_[i];
    if (partition_index == dimension_partition_counts_[i] - 1) {
      partition[i].second =
          shape_.dimensions(dimensions_[i]) - partition[i].first;
    } else {
      partition[i].second = dimension_partition_sizes_[i];
    }
    CHECK_GT(partition[i].second, 0);
    index -= partition_index * dimension_partition_strides_[i];
  }
  return partition;
}
int64_t ShapePartitionIterator::GetTotalPartitionCount() const {
  return ShapePartitionAssigner::GetTotalPartitionCount(
      dimension_partition_counts_);
}
}  
}  