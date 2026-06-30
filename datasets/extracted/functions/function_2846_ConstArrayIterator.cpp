#ifndef AROLLA_UTIL_ITERATOR_H_
#define AROLLA_UTIL_ITERATOR_H_
#include <iterator>
namespace arolla {
template <typename Array>
class ConstArrayIterator {
 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = typename Array::value_type;
  using pointer = const value_type*;
  using reference = decltype(std::declval<const Array>()[0]);
  using size_type = typename Array::size_type;
  using difference_type = typename Array::difference_type;
  ConstArrayIterator() : arr_(nullptr), pos_(0) {}
  ConstArrayIterator(const Array* arr, size_type pos) : arr_(arr), pos_(pos) {}
  reference operator*() const { return (*arr_)[pos_]; }
  pointer operator->() const { return &**this; }
  reference operator[](size_type n) const { return *(*this + n); }
  ConstArrayIterator& operator+=(difference_type n) {
    pos_ += n;
    return *this;
  }
  ConstArrayIterator& operator-=(difference_type n) { return *this += -n; }
  ConstArrayIterator& operator++() { return *this += 1; }
  ConstArrayIterator& operator--() { return *this -= 1; }
  ConstArrayIterator operator++(int) {
    ConstArrayIterator t = *this;
    ++*this;
    return t;
  }
  ConstArrayIterator operator--(int) {
    ConstArrayIterator t = *this;
    --*this;
    return t;
  }
  friend ConstArrayIterator operator+(ConstArrayIterator v, difference_type n) {
    return v += n;
  }
  friend ConstArrayIterator operator+(difference_type n, ConstArrayIterator v) {
    return v += n;
  }
  friend ConstArrayIterator operator-(ConstArrayIterator v, difference_type n) {
    return v -= n;
  }
  friend difference_type operator-(const ConstArrayIterator& a,
                                   const ConstArrayIterator& b) {
    return static_cast<difference_type>(a.pos_) -
           static_cast<difference_type>(b.pos_);
  }
  friend bool operator==(ConstArrayIterator a, ConstArrayIterator b) {
    return a.pos_ == b.pos_;
  }
  friend bool operator!=(ConstArrayIterator a, ConstArrayIterator b) {
    return !(a == b);
  }
  friend bool operator<(ConstArrayIterator a, ConstArrayIterator b) {
    return b - a > 0;
  }
  friend bool operator>(ConstArrayIterator a, ConstArrayIterator b) {
    return b < a;
  }
  friend bool operator<=(ConstArrayIterator a, ConstArrayIterator b) {
    return !(a > b);
  }
  friend bool operator>=(ConstArrayIterator a, ConstArrayIterator b) {
    return !(a < b);
  }
 private:
  const Array* arr_;  
  size_type pos_;     
};
}  
#endif  