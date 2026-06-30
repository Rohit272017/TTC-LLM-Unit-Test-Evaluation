#ifndef QUICHE_QUIC_CORE_QUIC_INTERVAL_H_
#define QUICHE_QUIC_CORE_QUIC_INTERVAL_H_
#include <stddef.h>
#include <algorithm>
#include <ostream>
#include <type_traits>
#include <utility>
#include <vector>
#include "quiche/quic/platform/api/quic_export.h"
namespace quic {
template <typename T>
class QUICHE_NO_EXPORT QuicInterval {
 private:
  template <typename U>
  class QUICHE_NO_EXPORT DiffTypeOrVoid {
   private:
    template <typename V>
    static auto f(const V* v) -> decltype(*v - *v);
    template <typename V>
    static void f(...);
   public:
    using type = typename std::decay<decltype(f<U>(nullptr))>::type;
  };
 public:
  QuicInterval() : min_(), max_() {}
  QuicInterval(const T& min, const T& max) : min_(min), max_(max) {}
  template <typename U1, typename U2,
            typename = typename std::enable_if<
                std::is_convertible<U1, T>::value &&
                std::is_convertible<U2, T>::value>::type>
  QuicInterval(U1&& min, U2&& max)
      : min_(std::forward<U1>(min)), max_(std::forward<U2>(max)) {}
  const T& min() const { return min_; }
  const T& max() const { return max_; }
  void SetMin(const T& t) { min_ = t; }
  void SetMax(const T& t) { max_ = t; }
  void Set(const T& min, const T& max) {
    SetMin(min);
    SetMax(max);
  }
  void Clear() { *this = {}; }
  bool Empty() const { return min() >= max(); }
  typename DiffTypeOrVoid<T>::type Length() const {
    return (Empty() ? min() : max()) - min();
  }
  bool Contains(const T& t) const { return min() <= t && max() > t; }
  bool Contains(const QuicInterval& i) const {
    return !Empty() && !i.Empty() && min() <= i.min() && max() >= i.max();
  }
  bool Intersects(const QuicInterval& i) const {
    return !Empty() && !i.Empty() && min() < i.max() && max() > i.min();
  }
  bool Intersects(const QuicInterval& i, QuicInterval* out) const;
  bool IntersectWith(const QuicInterval& i);
  bool Separated(const QuicInterval& other) const {
    if (Empty() || other.Empty()) return true;
    return other.max() < min() || max() < other.min();
  }
  bool SpanningUnion(const QuicInterval& i);
  bool Difference(const QuicInterval& i,
                  std::vector<QuicInterval*>* difference) const;
  bool Difference(const QuicInterval& i, QuicInterval* lo,
                  QuicInterval* hi) const;
  friend bool operator==(const QuicInterval& a, const QuicInterval& b) {
    bool ae = a.Empty();
    bool be = b.Empty();
    if (ae && be) return true;   
    if (ae != be) return false;  
    return a.min() == b.min() && a.max() == b.max();
  }
  friend bool operator!=(const QuicInterval& a, const QuicInterval& b) {
    return !(a == b);
  }
  friend bool operator<(const QuicInterval& a, const QuicInterval& b) {
    return a.min() < b.min() || (!(b.min() < a.min()) && b.max() < a.max());
  }
 private:
  T min_;  
  T max_;  
};
template <typename T>
QuicInterval<T> MakeQuicInterval(T&& lhs, T&& rhs) {
  return QuicInterval<T>(std::forward<T>(lhs), std::forward<T>(rhs));
}
template <typename T>
auto operator<<(std::ostream& out, const QuicInterval<T>& i)
    -> decltype(out << i.min()) {
  return out << "[" << i.min() << ", " << i.max() << ")";
}
template <typename T>
bool QuicInterval<T>::Intersects(const QuicInterval& i,
                                 QuicInterval* out) const {
  if (!Intersects(i)) return false;
  if (out != nullptr) {
    *out = QuicInterval(std::max(min(), i.min()), std::min(max(), i.max()));
  }
  return true;
}
template <typename T>
bool QuicInterval<T>::IntersectWith(const QuicInterval& i) {
  if (Empty()) return false;
  bool modified = false;
  if (i.min() > min()) {
    SetMin(i.min());
    modified = true;
  }
  if (i.max() < max()) {
    SetMax(i.max());
    modified = true;
  }
  return modified;
}
template <typename T>
bool QuicInterval<T>::SpanningUnion(const QuicInterval& i) {
  if (i.Empty()) return false;
  if (Empty()) {
    *this = i;
    return true;
  }
  bool modified = false;
  if (i.min() < min()) {
    SetMin(i.min());
    modified = true;
  }
  if (i.max() > max()) {
    SetMax(i.max());
    modified = true;
  }
  return modified;
}
template <typename T>
bool QuicInterval<T>::Difference(const QuicInterval& i,
                                 std::vector<QuicInterval*>* difference) const {
  if (Empty()) {
    return false;
  }
  if (i.Empty()) {
    difference->push_back(new QuicInterval(*this));
    return false;
  }
  if (min() < i.max() && min() >= i.min() && max() > i.max()) {
    difference->push_back(new QuicInterval(i.max(), max()));
    return true;
  }
  if (max() > i.min() && max() <= i.max() && min() < i.min()) {
    difference->push_back(new QuicInterval(min(), i.min()));
    return true;
  }
  if (min() < i.min() && max() > i.max()) {
    difference->push_back(new QuicInterval(min(), i.min()));
    difference->push_back(new QuicInterval(i.max(), max()));
    return true;
  }
  if (min() >= i.min() && max() <= i.max()) {
    return true;
  }
  difference->push_back(new QuicInterval(*this));
  return false;
}
template <typename T>
bool QuicInterval<T>::Difference(const QuicInterval& i, QuicInterval* lo,
                                 QuicInterval* hi) const {
  *lo = {};
  *hi = {};
  if (Empty()) return false;
  if (i.Empty()) {
    *lo = *this;
    return false;
  }
  if (min() < i.max() && min() >= i.min() && max() > i.max()) {
    *hi = QuicInterval(i.max(), max());
    return true;
  }
  if (max() > i.min() && max() <= i.max() && min() < i.min()) {
    *lo = QuicInterval(min(), i.min());
    return true;
  }
  if (min() < i.min() && max() > i.max()) {
    *lo = QuicInterval(min(), i.min());
    *hi = QuicInterval(i.max(), max());
    return true;
  }
  if (min() >= i.min() && max() <= i.max()) {
    return true;
  }
  *lo = *this;  
  return false;
}
}  
#endif  