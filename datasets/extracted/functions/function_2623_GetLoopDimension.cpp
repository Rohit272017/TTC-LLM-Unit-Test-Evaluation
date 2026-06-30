#ifndef TENSORSTORE_UTIL_ITERATE_OVER_INDEX_RANGE_H_
#define TENSORSTORE_UTIL_ITERATE_OVER_INDEX_RANGE_H_
#include <cassert>
#include <type_traits>
#include "tensorstore/box.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/void_wrapper.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/constant_vector.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/span.h"
namespace tensorstore {
namespace internal_iterate {
inline constexpr DimensionIndex GetLoopDimension(ContiguousLayoutOrder order,
                                                 DimensionIndex outer_dims,
                                                 DimensionIndex total_dims) {
  return order == ContiguousLayoutOrder::c ? outer_dims
                                           : total_dims - 1 - outer_dims;
}
template <typename Func, typename IndexType, DimensionIndex Rank>
using IterateOverIndexRangeResult = std::decay_t<
    std::invoke_result_t<Func, tensorstore::span<const IndexType, Rank>>>;
template <ContiguousLayoutOrder Order, typename Func, typename IndexType,
          DimensionIndex Rank>
struct IterateOverIndexRangeHelper {
  using IndicesSpan = tensorstore::span<const IndexType, Rank>;
  using ResultType = IterateOverIndexRangeResult<Func, IndexType, Rank>;
  using WrappedResultType = internal::Void::WrappedType<ResultType>;
  static WrappedResultType LoopImpl(
      Func func, DimensionIndex outer_dims, const IndexType* origin,
      const IndexType* shape, tensorstore::span<IndexType, Rank> indices) {
    WrappedResultType result =
        internal::DefaultIterationResult<WrappedResultType>::value();
    const DimensionIndex cur_dim =
        GetLoopDimension(Order, outer_dims, indices.size());
    const IndexType start = origin[cur_dim];
    const IndexType stop = shape[cur_dim] + start;
    if (outer_dims + 1 == indices.size()) {
      for (IndexType i = start; i < stop; ++i) {
        indices[cur_dim] = i;
        result = internal::Void::CallAndWrap(func, IndicesSpan(indices));
        if (!result) break;
      }
    } else {
      for (IndexType i = start; i < stop; ++i) {
        indices[cur_dim] = i;
        result = LoopImpl(func, outer_dims + 1, origin, shape, indices);
        if (!result) break;
      }
    }
    return result;
  }
  static ResultType Start(Func func, const IndexType* origin,
                          IndicesSpan shape) {
    if (shape.size() == 0) {
      return func(tensorstore::span<const IndexType, Rank>());
    }
    assert(shape.size() <= kMaxRank);
    IndexType indices[kMaxRank];
    return internal::Void::Unwrap(LoopImpl(
        func, 0, &origin[0], &shape[0],
        tensorstore::span<IndexType, Rank>(&indices[0], shape.size())));
  }
};
}  
template <ContiguousLayoutOrder Order = ContiguousLayoutOrder::c,
          typename IndexType, DimensionIndex Rank, typename Func>
internal_iterate::IterateOverIndexRangeResult<
    Func, std::remove_const_t<IndexType>, Rank>
IterateOverIndexRange(tensorstore::span<IndexType, Rank> origin,
                      tensorstore::span<IndexType, Rank> shape, Func&& func) {
  assert(origin.size() == shape.size());
  return internal_iterate::IterateOverIndexRangeHelper<
      Order, Func, std::remove_const_t<IndexType>, Rank>::Start(func,
                                                                origin.data(),
                                                                shape);
}
template <ContiguousLayoutOrder Order = ContiguousLayoutOrder::c,
          typename BoxType, typename Func>
std::enable_if_t<IsBoxLike<BoxType>,
                 internal_iterate::IterateOverIndexRangeResult<
                     Func, Index, BoxType::static_rank>>
IterateOverIndexRange(const BoxType& box, Func&& func,
                      ContiguousLayoutOrder order = ContiguousLayoutOrder::c) {
  return internal_iterate::IterateOverIndexRangeHelper<
      Order, Func, Index, BoxType::static_rank>::Start(func,
                                                       box.origin().data(),
                                                       box.shape());
}
template <ContiguousLayoutOrder Order = ContiguousLayoutOrder::c,
          typename IndexType, DimensionIndex Rank, typename Func>
internal_iterate::IterateOverIndexRangeResult<
    Func, std::remove_const_t<IndexType>, Rank>
IterateOverIndexRange(tensorstore::span<IndexType, Rank> shape, Func&& func) {
  using NonConstIndex = std::remove_const_t<IndexType>;
  return internal_iterate::
      IterateOverIndexRangeHelper<Order, Func, NonConstIndex, Rank>::Start(
          func,
          GetConstantVector<NonConstIndex, 0>(GetStaticOrDynamicExtent(shape))
              .data(),
          shape);
}
}  
#endif  