#include "tensorstore/driver/downsample/grid_occupancy_map.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/span.h"
namespace tensorstore {
namespace internal_downsample {
GridOccupancyMap::GridOccupancyMap(GridOccupancyTracker&& tracker,
                                   BoxView<> domain)
    : partition_points(domain.rank()) {
  const DimensionIndex rank = domain.rank();
  span<Index> occupied_chunks = tracker.occupied_chunks;
  {
    absl::flat_hash_map<Index, Index> partition_map;
    for (DimensionIndex dim = 0; dim < rank; ++dim) {
      partition_map.clear();
      IndexInterval bounds = domain[dim];
      partition_map.emplace(bounds.inclusive_min(), 0);
      partition_map.emplace(bounds.exclusive_max(), 0);
      for (ptrdiff_t i = dim; i < occupied_chunks.size(); i += 2 * rank) {
        Index begin = occupied_chunks[i];
        Index end = begin + occupied_chunks[i + rank];
        partition_map.emplace(begin, 0);
        partition_map.emplace(end, 0);
      }
      auto& dim_partition_points = partition_points[dim];
      dim_partition_points.reserve(partition_map.size());
      for (const auto& p : partition_map) {
        dim_partition_points.push_back(p.first);
      }
      std::sort(dim_partition_points.begin(), dim_partition_points.end());
      for (size_t i = 0, size = dim_partition_points.size(); i < size; ++i) {
        partition_map.at(dim_partition_points[i]) = i;
      }
      for (ptrdiff_t i = dim; i < occupied_chunks.size(); i += 2 * rank) {
        Index& begin = occupied_chunks[i];
        Index& end = occupied_chunks[i + rank];
        end = partition_map.at(begin + end);
        begin = partition_map.at(begin);
      }
    }
  }
  Index grid_cell[kMaxRank];
  span<Index> grid_cell_span(&grid_cell[0], rank);
  {
    for (DimensionIndex dim = 0; dim < rank; ++dim) {
      grid_cell[dim] = partition_points[dim].size() - 1;
    }
    occupied_chunk_mask =
        AllocateArray<bool>(grid_cell_span, c_order, value_init);
  }
  for (ptrdiff_t i = 0; i < occupied_chunks.size(); i += 2 * rank) {
    std::copy_n(&occupied_chunks[i], rank, &grid_cell[0]);
    do {
      occupied_chunk_mask(grid_cell_span) = true;
    } while (internal::AdvanceIndices(rank, &grid_cell[0], &occupied_chunks[i],
                                      &occupied_chunks[i + rank]));
  }
}
bool GridOccupancyMap::GetGridCellDomain(
    span<const Index> grid_cell, MutableBoxView<> grid_cell_domain) const {
  assert(grid_cell.size() == grid_cell_domain.rank());
  assert(grid_cell.size() == rank());
  if (occupied_chunk_mask(grid_cell)) return false;
  for (DimensionIndex dim = 0; dim < grid_cell.size(); ++dim) {
    const Index partition_index = grid_cell[dim];
    grid_cell_domain[dim] = IndexInterval::UncheckedHalfOpen(
        partition_points[dim][partition_index],
        partition_points[dim][partition_index + 1]);
  }
  return true;
}
void GridOccupancyMap::InitializeCellIterator(span<Index> grid_cell) const {
  std::fill(grid_cell.begin(), grid_cell.end(), 0);
}
bool GridOccupancyMap::AdvanceCellIterator(span<Index> grid_cell) const {
  assert(grid_cell.size() == occupied_chunk_mask.rank());
  return internal::AdvanceIndices(grid_cell.size(), grid_cell.data(),
                                  occupied_chunk_mask.shape().data());
}
}  
}  