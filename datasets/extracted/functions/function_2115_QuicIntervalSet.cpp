#ifndef QUICHE_QUIC_CORE_QUIC_INTERVAL_SET_H_
#define QUICHE_QUIC_CORE_QUIC_INTERVAL_SET_H_
#include <stddef.h>
#include <algorithm>
#include <initializer_list>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "quiche/quic/core/quic_interval.h"
#include "quiche/quic/platform/api/quic_flags.h"
#include "quiche/common/platform/api/quiche_containers.h"
#include "quiche/common/platform/api/quiche_logging.h"
namespace quic {
template <typename T>
class QUICHE_NO_EXPORT QuicIntervalSet {
 public:
  using value_type = QuicInterval<T>;
 private:
  struct QUICHE_NO_EXPORT IntervalLess {
    using is_transparent = void;
    bool operator()(const value_type& a, const value_type& b) const;
    bool operator()(const value_type& a, const T& point) const;
    bool operator()(const value_type& a, T&& point) const;
    bool operator()(const T& point, const value_type& a) const;
    bool operator()(T&& point, const value_type& a) const;
  };
  using Set = quiche::QuicheSmallOrderedSet<value_type, IntervalLess>;
 public:
  using const_iterator = typename Set::const_iterator;
  using const_reverse_iterator = typename Set::const_reverse_iterator;
  QuicIntervalSet() = default;
  explicit QuicIntervalSet(const value_type& interval) { Add(interval); }
  QuicIntervalSet(const T& min, const T& max) { Add(min, max); }
  QuicIntervalSet(std::initializer_list<value_type> il) { assign(il); }
  void Clear() { intervals_.clear(); }
  size_t Size() const { return intervals_.size(); }
  value_type SpanningInterval() const;
  void Add(const value_type& interval);
  void Add(const T& min, const T& max) { Add(value_type(min, max)); }
  void AddOptimizedForAppend(const value_type& interval) {
    if (Empty() || !GetQuicFlag(quic_interval_set_enable_add_optimization)) {
      Add(interval);
      return;
    }
    const_reverse_iterator last_interval = intervals_.rbegin();
    if (interval.min() < last_interval->min() ||
        interval.min() > last_interval->max()) {
      Add(interval);
      return;
    }
    if (interval.max() <= last_interval->max()) {
      return;
    }
    const_cast<value_type*>(&(*last_interval))->SetMax(interval.max());
  }
  void AddOptimizedForAppend(const T& min, const T& max) {
    AddOptimizedForAppend(value_type(min, max));
  }
  void PopFront() {
    QUICHE_DCHECK(!Empty());
    intervals_.erase(intervals_.begin());
  }
  bool TrimLessThan(const T& value) {
    size_t num_intervals_trimmed = 0;
    while (!intervals_.empty()) {
      const_iterator first_interval = intervals_.begin();
      if (first_interval->min() >= value) {
        break;
      }
      ++num_intervals_trimmed;
      if (first_interval->max() <= value) {
        intervals_.erase(first_interval);
        continue;
      }
      const_cast<value_type*>(&(*first_interval))->SetMin(value);
      break;
    }
    return num_intervals_trimmed != 0;
  }
  bool Empty() const { return intervals_.empty(); }
  bool Contains(const T& value) const;
  bool Contains(const value_type& interval) const;
  bool Contains(const QuicIntervalSet<T>& other) const;
  bool Contains(const T& min, const T& max) const {
    return Contains(value_type(min, max));
  }
  bool Intersects(const QuicIntervalSet& other) const;
  const_iterator Find(const T& value) const;
  const_iterator Find(const value_type& interval) const;
  const_iterator Find(const T& min, const T& max) const {
    return Find(value_type(min, max));
  }
  const_iterator LowerBound(const T& value) const;
  const_iterator UpperBound(const T& value) const;
  bool IsDisjoint(const value_type& interval) const;
  void Union(const QuicIntervalSet& other);
  void Intersection(const QuicIntervalSet& other);
  void Difference(const value_type& interval);
  void Difference(const T& min, const T& max);
  void Difference(const QuicIntervalSet& other);
  void Complement(const T& min, const T& max);
  const_iterator begin() const { return intervals_.begin(); }
  const_iterator end() const { return intervals_.end(); }
  const_reverse_iterator rbegin() const { return intervals_.rbegin(); }
  const_reverse_iterator rend() const { return intervals_.rend(); }
  template <typename Iter>
  void assign(Iter first, Iter last) {
    Clear();
    for (; first != last; ++first) Add(*first);
  }
  void assign(std::initializer_list<value_type> il) {
    assign(il.begin(), il.end());
  }
  std::string ToString() const;
  QuicIntervalSet& operator=(std::initializer_list<value_type> il) {
    assign(il.begin(), il.end());
    return *this;
  }
  friend bool operator==(const QuicIntervalSet& a, const QuicIntervalSet& b) {
    return a.Size() == b.Size() &&
           std::equal(a.begin(), a.end(), b.begin(), NonemptyIntervalEq());
  }
  friend bool operator!=(const QuicIntervalSet& a, const QuicIntervalSet& b) {
    return !(a == b);
  }
 private:
  struct QUICHE_NO_EXPORT NonemptyIntervalEq {
    bool operator()(const value_type& a, const value_type& b) const {
      return a.min() == b.min() && a.max() == b.max();
    }
  };
  bool Valid() const;
  const_iterator FindIntersectionCandidate(const QuicIntervalSet& other) const;
  const_iterator FindIntersectionCandidate(const value_type& interval) const;
  template <typename X, typename Func>
  static bool FindNextIntersectingPairImpl(X* x, const QuicIntervalSet& y,
                                           const_iterator* mine,
                                           const_iterator* theirs,
                                           Func on_hole);
  bool FindNextIntersectingPair(const QuicIntervalSet& other,
                                const_iterator* mine,
                                const_iterator* theirs) const {
    return FindNextIntersectingPairImpl(
        this, other, mine, theirs,
        [](const QuicIntervalSet*, const_iterator, const_iterator end) {
          return end;
        });
  }
  bool FindNextIntersectingPairAndEraseHoles(const QuicIntervalSet& other,
                                             const_iterator* mine,
                                             const_iterator* theirs) {
    return FindNextIntersectingPairImpl(
        this, other, mine, theirs,
        [](QuicIntervalSet* x, const_iterator from, const_iterator to) {
          return x->intervals_.erase(from, to);
        });
  }
  Set intervals_;
};
template <typename T>
auto operator<<(std::ostream& out, const QuicIntervalSet<T>& seq)
    -> decltype(out << *seq.begin()) {
  out << "{";
  for (const auto& interval : seq) {
    out << " " << interval;
  }
  out << " }";
  return out;
}
template <typename T>
typename QuicIntervalSet<T>::value_type QuicIntervalSet<T>::SpanningInterval()
    const {
  value_type result;
  if (!intervals_.empty()) {
    result.SetMin(intervals_.begin()->min());
    result.SetMax(intervals_.rbegin()->max());
  }
  return result;
}
template <typename T>
void QuicIntervalSet<T>::Add(const value_type& interval) {
  if (interval.Empty()) return;
  const_iterator it = intervals_.lower_bound(interval.min());
  value_type the_union = interval;
  if (it != intervals_.begin()) {
    --it;
    if (it->Separated(the_union)) {
      ++it;
    }
  }
  const_iterator start = it;
  while (it != intervals_.end() && !it->Separated(the_union)) {
    the_union.SpanningUnion(*it);
    ++it;
  }
  intervals_.erase(start, it);
  intervals_.insert(the_union);
}
template <typename T>
bool QuicIntervalSet<T>::Contains(const T& value) const {
  const_iterator it = intervals_.upper_bound(value);
  if (it == intervals_.begin()) return false;
  --it;
  return it->Contains(value);
}
template <typename T>
bool QuicIntervalSet<T>::Contains(const value_type& interval) const {
  const_iterator it = intervals_.upper_bound(interval.min());
  if (it == intervals_.begin()) return false;
  --it;
  return it->Contains(interval);
}
template <typename T>
bool QuicIntervalSet<T>::Contains(const QuicIntervalSet<T>& other) const {
  if (!SpanningInterval().Contains(other.SpanningInterval())) {
    return false;
  }
  for (const_iterator i = other.begin(); i != other.end(); ++i) {
    if (!Contains(*i)) {
      return false;
    }
  }
  return true;
}
template <typename T>
typename QuicIntervalSet<T>::const_iterator QuicIntervalSet<T>::Find(
    const T& value) const {
  const_iterator it = intervals_.upper_bound(value);
  if (it == intervals_.begin()) return intervals_.end();
  --it;
  if (it->Contains(value))
    return it;
  else
    return intervals_.end();
}
template <typename T>
typename QuicIntervalSet<T>::const_iterator QuicIntervalSet<T>::Find(
    const value_type& probe) const {
  const_iterator it = intervals_.upper_bound(probe.min());
  if (it == intervals_.begin()) return intervals_.end();
  --it;
  if (it->Contains(probe))
    return it;
  else
    return intervals_.end();
}
template <typename T>
typename QuicIntervalSet<T>::const_iterator QuicIntervalSet<T>::LowerBound(
    const T& value) const {
  const_iterator it = intervals_.lower_bound(value);
  if (it == intervals_.begin()) {
    return it;
  }
  --it;
  if (it->Contains(value)) {
    return it;
  } else {
    return ++it;
  }
}
template <typename T>
typename QuicIntervalSet<T>::const_iterator QuicIntervalSet<T>::UpperBound(
    const T& value) const {
  return intervals_.upper_bound(value);
}
template <typename T>
bool QuicIntervalSet<T>::IsDisjoint(const value_type& interval) const {
  if (interval.Empty()) return true;
  const_iterator it = intervals_.upper_bound(interval.min());
  if (it != intervals_.end() && interval.max() > it->min()) return false;
  if (it == intervals_.begin()) return true;
  --it;
  return it->max() <= interval.min();
}
template <typename T>
void QuicIntervalSet<T>::Union(const QuicIntervalSet& other) {
  for (const value_type& interval : other.intervals_) {
    Add(interval);
  }
}
template <typename T>
typename QuicIntervalSet<T>::const_iterator
QuicIntervalSet<T>::FindIntersectionCandidate(
    const QuicIntervalSet& other) const {
  return FindIntersectionCandidate(*other.intervals_.begin());
}
template <typename T>
typename QuicIntervalSet<T>::const_iterator
QuicIntervalSet<T>::FindIntersectionCandidate(
    const value_type& interval) const {
  const_iterator mine = intervals_.upper_bound(interval.min());
  if (mine != intervals_.begin()) {
    --mine;
  }
  return mine;
}
template <typename T>
template <typename X, typename Func>
bool QuicIntervalSet<T>::FindNextIntersectingPairImpl(X* x,
                                                      const QuicIntervalSet& y,
                                                      const_iterator* mine,
                                                      const_iterator* theirs,
                                                      Func on_hole) {
  QUICHE_CHECK(x != nullptr);
  if ((*mine == x->intervals_.end()) || (*theirs == y.intervals_.end())) {
    return false;
  }
  while (!(**mine).Intersects(**theirs)) {
    const_iterator erase_first = *mine;
    while (*mine != x->intervals_.end() && (**mine).max() <= (**theirs).min()) {
      ++(*mine);
    }
    *mine = on_hole(x, erase_first, *mine);
    if (*mine == x->intervals_.end()) {
      return false;
    }
    while (*theirs != y.intervals_.end() &&
           (**theirs).max() <= (**mine).min()) {
      ++(*theirs);
    }
    if (*theirs == y.intervals_.end()) {
      on_hole(x, *mine, x->intervals_.end());
      return false;
    }
  }
  return true;
}
template <typename T>
void QuicIntervalSet<T>::Intersection(const QuicIntervalSet& other) {
  if (!SpanningInterval().Intersects(other.SpanningInterval())) {
    intervals_.clear();
    return;
  }
  const_iterator mine = FindIntersectionCandidate(other);
  mine = intervals_.erase(intervals_.begin(), mine);
  const_iterator theirs = other.FindIntersectionCandidate(*this);
  while (FindNextIntersectingPairAndEraseHoles(other, &mine, &theirs)) {
    value_type i(*mine);
    intervals_.erase(mine);
    mine = intervals_.end();
    value_type intersection;
    while (theirs != other.intervals_.end() &&
           i.Intersects(*theirs, &intersection)) {
      std::pair<const_iterator, bool> ins = intervals_.insert(intersection);
      QUICHE_DCHECK(ins.second);
      mine = ins.first;
      ++theirs;
    }
    QUICHE_DCHECK(mine != intervals_.end());
    --theirs;
    ++mine;
  }
  QUICHE_DCHECK(Valid());
}
template <typename T>
bool QuicIntervalSet<T>::Intersects(const QuicIntervalSet& other) const {
  auto mine = intervals_.begin();
  auto theirs = other.intervals_.begin();
  while (mine != intervals_.end() && theirs != other.intervals_.end()) {
    if (mine->Intersects(*theirs))
      return true;
    else if (*mine < *theirs)
      ++mine;
    else
      ++theirs;
  }
  return false;
}
template <typename T>
void QuicIntervalSet<T>::Difference(const value_type& interval) {
  if (!SpanningInterval().Intersects(interval)) {
    return;
  }
  Difference(QuicIntervalSet<T>(interval));
}
template <typename T>
void QuicIntervalSet<T>::Difference(const T& min, const T& max) {
  Difference(value_type(min, max));
}
template <typename T>
void QuicIntervalSet<T>::Difference(const QuicIntervalSet& other) {
  if (Empty()) return;
  Set result;
  const_iterator mine = intervals_.begin();
  value_type myinterval = *mine;
  const_iterator theirs = other.intervals_.begin();
  while (mine != intervals_.end()) {
    QUICHE_DCHECK(!myinterval.Empty());
    QUICHE_DCHECK(myinterval.max() == mine->max());
    if (theirs == other.intervals_.end() || myinterval.max() <= theirs->min()) {
      result.insert(result.end(), myinterval);
      myinterval.Clear();
    } else if (theirs->max() <= myinterval.min()) {
      ++theirs;
    } else if (myinterval.min() < theirs->min()) {
      result.insert(result.end(), value_type(myinterval.min(), theirs->min()));
      myinterval.SetMin(theirs->max());
    } else {
      myinterval.SetMin(theirs->max());
    }
    if (myinterval.Empty()) {
      ++mine;
      if (mine != intervals_.end()) {
        myinterval = *mine;
      }
    }
  }
  std::swap(result, intervals_);
  QUICHE_DCHECK(Valid());
}
template <typename T>
void QuicIntervalSet<T>::Complement(const T& min, const T& max) {
  QuicIntervalSet<T> span(min, max);
  span.Difference(*this);
  intervals_.swap(span.intervals_);
}
template <typename T>
std::string QuicIntervalSet<T>::ToString() const {
  std::ostringstream os;
  os << *this;
  return os.str();
}
template <typename T>
bool QuicIntervalSet<T>::Valid() const {
  const_iterator prev = end();
  for (const_iterator it = begin(); it != end(); ++it) {
    if (it->min() >= it->max()) return false;
    if (prev != end() && prev->max() >= it->min()) return false;
    prev = it;
  }
  return true;
}
template <typename T>
bool QuicIntervalSet<T>::IntervalLess::operator()(const value_type& a,
                                                  const value_type& b) const {
  return a.min() < b.min();
}
template <typename T>
bool QuicIntervalSet<T>::IntervalLess::operator()(const value_type& a,
                                                  const T& point) const {
  return a.min() < point;
}
template <typename T>
bool QuicIntervalSet<T>::IntervalLess::operator()(const value_type& a,
                                                  T&& point) const {
  return a.min() < point;
}
template <typename T>
bool QuicIntervalSet<T>::IntervalLess::operator()(const T& point,
                                                  const value_type& a) const {
  return point < a.min();
}
template <typename T>
bool QuicIntervalSet<T>::IntervalLess::operator()(T&& point,
                                                  const value_type& a) const {
  return point < a.min();
}
}  
#endif  