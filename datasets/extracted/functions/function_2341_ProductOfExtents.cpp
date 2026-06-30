#ifndef TENSORSTORE_UTIL_EXTENTS_H_
#define TENSORSTORE_UTIL_EXTENTS_H_
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>
#include "absl/base/optimization.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/span.h"
namespace tensorstore {
template <typename T, ptrdiff_t Extent>
T ProductOfExtents(tensorstore::span<T, Extent> s) {
  using value_type = std::remove_const_t<T>;
  value_type result = 1;
  for (const auto& x : s) {
    assert(x >= 0);
    if (ABSL_PREDICT_FALSE(internal::MulOverflow(result, x, &result))) {
      result = std::numeric_limits<value_type>::max();
    }
  }
  return result;
}
template <DimensionIndex Rank, typename Indices, typename = void>
constexpr inline bool IsCompatibleFullIndexVector = false;
template <DimensionIndex Rank, typename Indices>
constexpr inline bool IsCompatibleFullIndexVector<
    Rank, Indices, std::void_t<internal::ConstSpanType<Indices>>> =
    RankConstraint::EqualOrUnspecified(
        Rank, internal::ConstSpanType<Indices>::extent) &&
    internal::IsIndexPack<
        typename internal::ConstSpanType<Indices>::value_type>;
template <DimensionIndex Rank, typename Indices, typename = void>
constexpr inline bool IsImplicitlyCompatibleFullIndexVector = false;
template <DimensionIndex Rank, typename Indices>
constexpr inline bool IsImplicitlyCompatibleFullIndexVector<
    Rank, Indices, std::void_t<internal::ConstSpanType<Indices>>> =
    RankConstraint::Implies(internal::ConstSpanType<Indices>::extent, Rank) &&
    internal::IsIndexPack<
        typename internal::ConstSpanType<Indices>::value_type>;
template <DimensionIndex Rank, typename Indices, typename = void>
constexpr inline bool IsCompatiblePartialIndexVector = false;
template <DimensionIndex Rank, typename Indices>
constexpr inline bool IsCompatiblePartialIndexVector<
    Rank, Indices, std::void_t<internal::ConstSpanType<Indices>>> =
    RankConstraint::GreaterEqualOrUnspecified(
        Rank, internal::ConstSpanType<Indices>::extent) &&
    internal::IsIndexPack<
        typename internal::ConstSpanType<Indices>::value_type>;
template <DimensionIndex Rank, typename... IndexType>
constexpr inline bool IsCompatibleFullIndexPack =
    RankConstraint::EqualOrUnspecified(Rank, sizeof...(IndexType)) &&
    internal::IsIndexPack<IndexType...>;
template <typename Indices, typename = void>
constexpr inline bool IsIndexConvertibleVector = false;
template <typename Indices>
constexpr inline bool IsIndexConvertibleVector<
    Indices, std::void_t<internal::ConstSpanType<Indices>>> =
    internal::IsIndexPack<
        typename internal::ConstSpanType<Indices>::value_type>;
template <typename Indices, typename = Index>
constexpr inline bool IsIndexVector = false;
template <typename Indices>
constexpr inline bool IsIndexVector<
    Indices, typename internal::ConstSpanType<Indices>::value_type> = true;
template <typename Indices, typename = Index>
constexpr inline bool IsMutableIndexVector = false;
template <typename Indices>
constexpr inline bool IsMutableIndexVector<
    Indices, typename internal::SpanType<Indices>::element_type> = true;
namespace internal_extents {
template <typename... Xs>
struct SpanStaticExtentHelper {};
template <typename... Ts, ptrdiff_t Extent>
struct SpanStaticExtentHelper<tensorstore::span<Ts, Extent>...>
    : public std::integral_constant<ptrdiff_t, Extent> {};
}  
template <typename X0, typename... Xs>
using SpanStaticExtent =
    internal_extents::SpanStaticExtentHelper<internal::ConstSpanType<X0>,
                                             internal::ConstSpanType<Xs>...>;
}  
#endif  