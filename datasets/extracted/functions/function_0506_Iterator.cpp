#ifndef QUICHE_QUIC_CORE_QUIC_INTERVAL_DEQUE_H_
#define QUICHE_QUIC_CORE_QUIC_INTERVAL_DEQUE_H_
#include <algorithm>
#include <optional>
#include "quiche/quic/core/quic_interval.h"
#include "quiche/quic/core/quic_types.h"
#include "quiche/quic/platform/api/quic_bug_tracker.h"
#include "quiche/quic/platform/api/quic_export.h"
#include "quiche/quic/platform/api/quic_logging.h"
#include "quiche/common/quiche_circular_deque.h"
namespace quic {
namespace test {
class QuicIntervalDequePeer;
}  
template <class T, class C = quiche::QuicheCircularDeque<T>>
class QUICHE_NO_EXPORT QuicIntervalDeque {
 public:
  class QUICHE_NO_EXPORT Iterator {
   public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;
    Iterator(std::size_t index, QuicIntervalDeque* deque)
        : index_(index), deque_(deque) {}
    Iterator& operator++() {
      const std::size_t container_size = deque_->container_.size();
      if (index_ >= container_size) {
        QUIC_BUG(QuicIntervalDeque_operator_plus_plus_iterator_out_of_bounds)
            << "Iterator out of bounds.";
        return *this;
      }
      index_++;
      if (deque_->cached_index_.has_value()) {
        const std::size_t cached_index = *deque_->cached_index_;
        if (index_ == container_size) {
          deque_->cached_index_.reset();
        } else {
          if (cached_index < index_) {
            deque_->cached_index_ = index_;
          }
        }
      }
      return *this;
    }
    Iterator operator++(int) {
      Iterator copy = *this;
      ++(*this);
      return copy;
    }
    Iterator& operator--() {
      if (index_ == 0) {
        QUIC_BUG(QuicIntervalDeque_operator_minus_minus_iterator_out_of_bounds)
            << "Iterator out of bounds.";
        return *this;
      }
      index_--;
      return *this;
    }
    Iterator operator--(int) {
      Iterator copy = *this;
      --(*this);
      return copy;
    }
    reference operator*() { return deque_->container_[index_]; }
    reference operator*() const { return deque_->container_[index_]; }
    pointer operator->() { return &deque_->container_[index_]; }
    bool operator==(const Iterator& rhs) const {
      return index_ == rhs.index_ && deque_ == rhs.deque_;
    }
    bool operator!=(const Iterator& rhs) const { return !(*this == rhs); }
    Iterator& operator+=(difference_type amount) {
      QUICHE_DCHECK_GE(static_cast<difference_type>(index_), -amount);
      index_ += amount;
      QUICHE_DCHECK_LT(index_, deque_->Size());
      return *this;
    }
    Iterator& operator-=(difference_type amount) { return operator+=(-amount); }
    difference_type operator-(const Iterator& rhs) const {
      return static_cast<difference_type>(index_) -
             static_cast<difference_type>(rhs.index_);
    }
   private:
    std::size_t index_;
    QuicIntervalDeque* deque_;
    friend class QuicIntervalDeque;
  };
  QuicIntervalDeque();
  void PushBack(T&& item);
  void PushBack(const T& item);
  void PopFront();
  Iterator DataBegin();
  Iterator DataEnd();
  Iterator DataAt(const std::size_t interval_begin);
  std::size_t Size() const;
  bool Empty() const;
 private:
  struct QUICHE_NO_EXPORT IntervalCompare {
    bool operator()(const T& item, std::size_t interval_begin) const {
      return item.interval().max() <= interval_begin;
    }
  };
  template <class U>
  void PushBackUniversal(U&& item);
  Iterator Search(const std::size_t interval_begin,
                  const std::size_t begin_index, const std::size_t end_index);
  friend class test::QuicIntervalDequePeer;
  C container_;
  std::optional<std::size_t> cached_index_;
};
template <class T, class C>
QuicIntervalDeque<T, C>::QuicIntervalDeque() {}
template <class T, class C>
void QuicIntervalDeque<T, C>::PushBack(T&& item) {
  PushBackUniversal(std::move(item));
}
template <class T, class C>
void QuicIntervalDeque<T, C>::PushBack(const T& item) {
  PushBackUniversal(item);
}
template <class T, class C>
void QuicIntervalDeque<T, C>::PopFront() {
  if (container_.size() == 0) {
    QUIC_BUG(QuicIntervalDeque_PopFront_empty)
        << "Trying to pop from an empty container.";
    return;
  }
  container_.pop_front();
  if (container_.size() == 0) {
    cached_index_.reset();
  }
  if (cached_index_.value_or(0) > 0) {
    cached_index_ = *cached_index_ - 1;
  }
}
template <class T, class C>
typename QuicIntervalDeque<T, C>::Iterator
QuicIntervalDeque<T, C>::DataBegin() {
  return Iterator(0, this);
}
template <class T, class C>
typename QuicIntervalDeque<T, C>::Iterator QuicIntervalDeque<T, C>::DataEnd() {
  return Iterator(container_.size(), this);
}
template <class T, class C>
typename QuicIntervalDeque<T, C>::Iterator QuicIntervalDeque<T, C>::DataAt(
    const std::size_t interval_begin) {
  if (!cached_index_.has_value()) {
    return Search(interval_begin, 0, container_.size());
  }
  const std::size_t cached_index = *cached_index_;
  QUICHE_DCHECK(cached_index < container_.size());
  const QuicInterval<size_t> cached_interval =
      container_[cached_index].interval();
  if (cached_interval.Contains(interval_begin)) {
    return Iterator(cached_index, this);
  }
  const std::size_t next_index = cached_index + 1;
  if (next_index < container_.size()) {
    if (container_[next_index].interval().Contains(interval_begin)) {
      cached_index_ = next_index;
      return Iterator(next_index, this);
    }
  }
  const std::size_t cached_begin = cached_interval.min();
  bool looking_below = interval_begin < cached_begin;
  const std::size_t lower = looking_below ? 0 : cached_index + 1;
  const std::size_t upper = looking_below ? cached_index : container_.size();
  Iterator ret = Search(interval_begin, lower, upper);
  if (ret == DataEnd()) {
    return ret;
  }
  if (!looking_below) {
    cached_index_ = ret.index_;
  }
  return ret;
}
template <class T, class C>
std::size_t QuicIntervalDeque<T, C>::Size() const {
  return container_.size();
}
template <class T, class C>
bool QuicIntervalDeque<T, C>::Empty() const {
  return container_.size() == 0;
}
template <class T, class C>
template <class U>
void QuicIntervalDeque<T, C>::PushBackUniversal(U&& item) {
  QuicInterval<std::size_t> interval = item.interval();
  if (interval.Empty()) {
    QUIC_BUG(QuicIntervalDeque_PushBackUniversal_empty)
        << "Trying to save empty interval to quiche::QuicheCircularDeque.";
    return;
  }
  container_.push_back(std::forward<U>(item));
  if (!cached_index_.has_value()) {
    cached_index_ = container_.size() - 1;
  }
}
template <class T, class C>
typename QuicIntervalDeque<T, C>::Iterator QuicIntervalDeque<T, C>::Search(
    const std::size_t interval_begin, const std::size_t begin_index,
    const std::size_t end_index) {
  auto begin = container_.begin() + begin_index;
  auto end = container_.begin() + end_index;
  auto res = std::lower_bound(begin, end, interval_begin, IntervalCompare());
  if (res != end && res->interval().Contains(interval_begin)) {
    return Iterator(std::distance(begin, res) + begin_index, this);
  }
  return DataEnd();
}
}  
#endif  