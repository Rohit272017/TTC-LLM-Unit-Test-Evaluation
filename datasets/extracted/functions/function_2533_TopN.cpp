#ifndef TENSORFLOW_LITE_KERNELS_CTC_TOP_N_H_
#define TENSORFLOW_LITE_KERNELS_CTC_TOP_N_H_
#include <stddef.h>
#include <algorithm>
#include <functional>
#include <string>
#include <vector>
#include "tensorflow/lite/kernels/internal/compatibility.h"
namespace tflite {
namespace gtl {
template <class T, class Cmp = std::greater<T> >
class TopN {
 public:
  enum State { UNORDERED, BOTTOM_KNOWN, HEAP_SORTED };
  using UnsortedIterator = typename std::vector<T>::const_iterator;
  explicit TopN(size_t limit) : TopN(limit, Cmp()) {}
  TopN(size_t limit, const Cmp &cmp) : limit_(limit), cmp_(cmp) {}
  size_t limit() const { return limit_; }
  size_t size() const { return std::min(elements_.size(), limit_); }
  bool empty() const { return size() == 0; }
  void reserve(size_t n) { elements_.reserve(std::min(n, limit_ + 1)); }
  void push(const T &v) { push(v, nullptr); }
  void push(const T &v, T *dropped) { PushInternal(v, dropped); }
  void push(T &&v) {  
    push(std::move(v), nullptr);
  }
  void push(T &&v, T *dropped) {  
    PushInternal(std::move(v), dropped);
  }
  const T &peek_bottom();
  std::vector<T> *Extract();
  std::vector<T> *ExtractUnsorted();
  std::vector<T> *ExtractNondestructive() const;
  void ExtractNondestructive(std::vector<T> *output) const;
  std::vector<T> *ExtractUnsortedNondestructive() const;
  void ExtractUnsortedNondestructive(std::vector<T> *output) const;
  UnsortedIterator unsorted_begin() const { return elements_.begin(); }
  UnsortedIterator unsorted_end() const { return elements_.begin() + size(); }
  Cmp *comparator() { return &cmp_; }
  void Reset();
 private:
  template <typename U>
  void PushInternal(U &&v, T *dropped);  
  std::vector<T> elements_;
  size_t limit_;  
  Cmp cmp_;       
  State state_ = UNORDERED;
};
template <class T, class Cmp>
template <typename U>
void TopN<T, Cmp>::PushInternal(U &&v, T *dropped) {  
  if (limit_ == 0) {
    if (dropped) *dropped = std::forward<U>(v);  
    return;
  }
  if (state_ != HEAP_SORTED) {
    elements_.push_back(std::forward<U>(v));  
    if (state_ == UNORDERED || cmp_(elements_.back(), elements_.front())) {
    } else {
      using std::swap;
      swap(elements_.front(), elements_.back());
    }
    if (elements_.size() == limit_ + 1) {
      std::make_heap(elements_.begin(), elements_.end(), cmp_);
      if (dropped) *dropped = std::move(elements_.front());
      std::pop_heap(elements_.begin(), elements_.end(), cmp_);
      state_ = HEAP_SORTED;
    }
  } else {
    if (cmp_(v, elements_.front())) {
      elements_.back() = std::forward<U>(v);  
      std::push_heap(elements_.begin(), elements_.end(), cmp_);
      if (dropped) *dropped = std::move(elements_.front());
      std::pop_heap(elements_.begin(), elements_.end(), cmp_);
    } else {
      if (dropped) *dropped = std::forward<U>(v);  
    }
  }
}
template <class T, class Cmp>
const T &TopN<T, Cmp>::peek_bottom() {
  TFLITE_DCHECK(!empty());
  if (state_ == UNORDERED) {
    int min_candidate = 0;
    for (size_t i = 1; i < elements_.size(); ++i) {
      if (cmp_(elements_[min_candidate], elements_[i])) {
        min_candidate = i;
      }
    }
    if (min_candidate != 0) {
      using std::swap;
      swap(elements_[0], elements_[min_candidate]);
    }
    state_ = BOTTOM_KNOWN;
  }
  return elements_.front();
}
template <class T, class Cmp>
std::vector<T> *TopN<T, Cmp>::Extract() {
  auto out = new std::vector<T>;
  out->swap(elements_);
  if (state_ != HEAP_SORTED) {
    std::sort(out->begin(), out->end(), cmp_);
  } else {
    out->pop_back();
    std::sort_heap(out->begin(), out->end(), cmp_);
  }
  return out;
}
template <class T, class Cmp>
std::vector<T> *TopN<T, Cmp>::ExtractUnsorted() {
  auto out = new std::vector<T>;
  out->swap(elements_);
  if (state_ == HEAP_SORTED) {
    out->pop_back();
  }
  return out;
}
template <class T, class Cmp>
std::vector<T> *TopN<T, Cmp>::ExtractNondestructive() const {
  auto out = new std::vector<T>;
  ExtractNondestructive(out);
  return out;
}
template <class T, class Cmp>
void TopN<T, Cmp>::ExtractNondestructive(std::vector<T> *output) const {
  TFLITE_DCHECK(output);
  *output = elements_;
  if (state_ != HEAP_SORTED) {
    std::sort(output->begin(), output->end(), cmp_);
  } else {
    output->pop_back();
    std::sort_heap(output->begin(), output->end(), cmp_);
  }
}
template <class T, class Cmp>
std::vector<T> *TopN<T, Cmp>::ExtractUnsortedNondestructive() const {
  auto elements = new std::vector<T>;
  ExtractUnsortedNondestructive(elements);
  return elements;
}
template <class T, class Cmp>
void TopN<T, Cmp>::ExtractUnsortedNondestructive(std::vector<T> *output) const {
  TFLITE_DCHECK(output);
  *output = elements_;
  if (state_ == HEAP_SORTED) {
    output->pop_back();
  }
}
template <class T, class Cmp>
void TopN<T, Cmp>::Reset() {
  elements_.clear();
  state_ = UNORDERED;
}
}  
}  
#endif  