#include "xla/shape_tree.h"
#include <cstddef>
#include <cstdint>
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tsl/platform/logging.h"  
namespace xla {
namespace internal {
IndexTable::IndexTable(const Shape& shape) : entries_(1) {
  size_t next_node_id = 0;
  CreateEntry(entries_[0], shape, next_node_id);
}
void IndexTable::CreateEntry(Entry& entry, const Shape& shape,
                             size_t& next_node_id) {
  entry.node_id = next_node_id++;
  if (!shape.IsTuple()) return;
  size_t children_start_id = entries_.size();
  entry.children_start_id = children_start_id;
  entries_.resize(entries_.size() + shape.tuple_shapes_size());
  for (size_t i = 0; i < shape.tuple_shapes_size(); ++i) {
    CreateEntry(entries_[children_start_id + i], shape.tuple_shapes(i),
                next_node_id);
  }
}
const IndexTable::Entry& IndexTable::operator[](ShapeIndexView index) const {
  const Entry* result = &entries_.front();
  for (int64_t i : index) {
    CHECK_GE(result->children_start_id, 0);
    result = &entries_[result->children_start_id + i];
  }
  return *result;
}
}  
}  