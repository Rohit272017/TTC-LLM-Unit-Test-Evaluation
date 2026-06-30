#ifndef XLA_ITERATOR_UTIL_H_
#define XLA_ITERATOR_UTIL_H_
#include <cstddef>
#include <iterator>
#include <utility>
#include "xla/tsl/lib/gtl/iterator_range.h"
namespace xla {
template <typename NestedIter>
class UnwrappingIterator {
 public:
  using iterator_category = std::input_iterator_tag;
  using value_type = decltype(std::declval<NestedIter>()->get());
  using difference_type = ptrdiff_t;
  using pointer = value_type*;
  using reference = value_type&;
  explicit UnwrappingIterator(NestedIter iter) : iter_(std::move(iter)) {}
  auto operator*() -> value_type { return iter_->get(); }
  UnwrappingIterator& operator++() {
    ++iter_;
    return *this;
  }
  UnwrappingIterator operator++(int) {
    UnwrappingIterator temp(iter_);
    operator++();
    return temp;
  }
  friend bool operator==(const UnwrappingIterator& a,
                         const UnwrappingIterator& b) {
    return a.iter_ == b.iter_;
  }
  friend bool operator!=(const UnwrappingIterator& a,
                         const UnwrappingIterator& b) {
    return !(a == b);
  }
 private:
  NestedIter iter_;
};
template <typename NestedIter>
UnwrappingIterator<NestedIter> MakeUnwrappingIterator(NestedIter iter) {
  return UnwrappingIterator<NestedIter>(std::move(iter));
}
template <typename NestedIter, typename UnaryPredicate>
class FilteringIterator {
 public:
  using iterator_category = std::input_iterator_tag;
  using value_type = decltype(*std::declval<NestedIter>());
  using difference_type = ptrdiff_t;
  using pointer = value_type*;
  using reference = value_type&;
  FilteringIterator(NestedIter iter, NestedIter end_iter, UnaryPredicate pred)
      : iter_(std::move(iter)),
        end_iter_(std::move(end_iter)),
        pred_(std::move(pred)) {
    if (iter_ != end_iter_ && !pred_(**this)) {
      ++*this;
    }
  }
  auto operator*() -> value_type { return *iter_; }
  FilteringIterator& operator++() {
    do {
      ++iter_;
    } while (iter_ != end_iter_ && !pred_(**this));
    return *this;
  }
  FilteringIterator operator++(int) {
    FilteringIterator temp(iter_, end_iter_, pred_);
    operator++();
    return temp;
  }
  friend bool operator==(const FilteringIterator& a,
                         const FilteringIterator& b) {
    return a.iter_ == b.iter_;
  }
  friend bool operator!=(const FilteringIterator& a,
                         const FilteringIterator& b) {
    return !(a == b);
  }
 private:
  NestedIter iter_;
  NestedIter end_iter_;
  UnaryPredicate pred_;
};
template <typename NestedIter, typename UnaryPredicate>
using FilteringUnwrappingIterator =
    FilteringIterator<UnwrappingIterator<NestedIter>, UnaryPredicate>;
template <typename NestedIter, typename UnaryPredicate>
FilteringUnwrappingIterator<NestedIter, UnaryPredicate>
MakeFilteringUnwrappingIterator(NestedIter iter, NestedIter end_iter,
                                UnaryPredicate pred) {
  return FilteringUnwrappingIterator<NestedIter, UnaryPredicate>(
      MakeUnwrappingIterator(iter), MakeUnwrappingIterator(end_iter),
      std::move(pred));
}
template <typename NestedIter, typename UnaryPredicate>
tsl::gtl::iterator_range<
    FilteringUnwrappingIterator<NestedIter, UnaryPredicate>>
MakeFilteringUnwrappingIteratorRange(NestedIter begin_iter, NestedIter end_iter,
                                     UnaryPredicate pred) {
  return {MakeFilteringUnwrappingIterator(begin_iter, end_iter, pred),
          MakeFilteringUnwrappingIterator(end_iter, end_iter, pred)};
}
}  
#endif  