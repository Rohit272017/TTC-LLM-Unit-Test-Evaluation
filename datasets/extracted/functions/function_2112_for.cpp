#include "arolla/jagged_shape/dense_array/jagged_shape.h"
#include <sstream>
#include <utility>
#include "arolla/jagged_shape/util/repr.h"
#include "arolla/util/repr.h"
#include "arolla/util/string.h"
namespace arolla {
ReprToken ReprTraits<JaggedDenseArrayShape>::operator()(
    const JaggedDenseArrayShape& value) const {
  std::ostringstream result;
  result << "JaggedShape(";
  bool first = true;
  for (const auto& edge : value.edges()) {
    result << NonFirstComma(first)
           << CompactSplitPointsAsSizesRepr(edge.edge_values().values.span(),
                                            3);
  }
  result << ")";
  return ReprToken{std::move(result).str()};
}
}  