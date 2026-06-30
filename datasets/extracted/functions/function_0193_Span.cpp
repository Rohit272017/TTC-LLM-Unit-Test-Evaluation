#ifndef LANGSVR_SPAN_H_
#define LANGSVR_SPAN_H_
#include <array>
#include <cstddef>
#include <vector>
namespace langsvr {
template <typename T>
class Span {
  public:
    Span(T* first, size_t count) : elements_(first), count_(count) {}
    Span(const std::vector<T>& vec) : elements_(vec.data()), count_(vec.size()) {}
    template <size_t N>
    Span(const std::array<T, N>& arr) : elements_(arr.data()), count_(arr.size()) {}
    const T& front() const { return elements_[0]; }
    const T& back() const { return elements_[count_ - 1]; }
    const T& operator[](size_t i) const { return elements_[i]; }
    const T* begin() const { return &elements_[0]; }
    const T* end() const { return &elements_[count_]; }
    size_t size() const { return count_; }
  private:
    const T* const elements_;
    const size_t count_;
};
}  
#endif  