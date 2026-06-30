#ifndef TENSORSTORE_INTERNAL_REGULAR_GRID_H_
#define TENSORSTORE_INTERNAL_REGULAR_GRID_H_
#include <cassert>
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/span.h"
namespace tensorstore {
namespace internal_grid_partition {
struct RegularGridRef {
  tensorstore::span<const Index> grid_cell_shape;
  DimensionIndex rank() const { return grid_cell_shape.size(); }
  IndexInterval GetCellOutputInterval(DimensionIndex dim,
                                      Index cell_index) const {
    assert(dim >= 0 && dim < rank());
    return IndexInterval::UncheckedSized(cell_index * grid_cell_shape[dim],
                                         grid_cell_shape[dim]);
  }
  Index operator()(DimensionIndex dim, Index output_index,
                   IndexInterval* cell_bounds) const {
    assert(dim >= 0 && dim < rank());
    Index cell_index = FloorOfRatio(output_index, grid_cell_shape[dim]);
    if (cell_bounds) {
      *cell_bounds = GetCellOutputInterval(dim, cell_index);
    }
    return cell_index;
  }
};
}  
}  
#endif  