#ifndef GLOG_STL_LOGGING_H
#define GLOG_STL_LOGGING_H
#include <deque>
#include <list>
#include <map>
#include <ostream>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
template <class First, class Second>
std::ostream& operator<<(std::ostream& out, const std::pair<First, Second>& p);
namespace google {
template <class Iter>
void PrintSequence(std::ostream& out, Iter begin, Iter end);
}
#define OUTPUT_TWO_ARG_CONTAINER(Sequence)                       \
  template <class T1, class T2>                                  \
  inline std::ostream& operator<<(std::ostream& out,             \
                                  const Sequence<T1, T2>& seq) { \
    google::PrintSequence(out, seq.begin(), seq.end());          \
    return out;                                                  \
  }
OUTPUT_TWO_ARG_CONTAINER(std::vector)
OUTPUT_TWO_ARG_CONTAINER(std::deque)
OUTPUT_TWO_ARG_CONTAINER(std::list)
#undef OUTPUT_TWO_ARG_CONTAINER
#define OUTPUT_THREE_ARG_CONTAINER(Sequence)                         \
  template <class T1, class T2, class T3>                            \
  inline std::ostream& operator<<(std::ostream& out,                 \
                                  const Sequence<T1, T2, T3>& seq) { \
    google::PrintSequence(out, seq.begin(), seq.end());              \
    return out;                                                      \
  }
OUTPUT_THREE_ARG_CONTAINER(std::set)
OUTPUT_THREE_ARG_CONTAINER(std::multiset)
#undef OUTPUT_THREE_ARG_CONTAINER
#define OUTPUT_FOUR_ARG_CONTAINER(Sequence)                              \
  template <class T1, class T2, class T3, class T4>                      \
  inline std::ostream& operator<<(std::ostream& out,                     \
                                  const Sequence<T1, T2, T3, T4>& seq) { \
    google::PrintSequence(out, seq.begin(), seq.end());                  \
    return out;                                                          \
  }
OUTPUT_FOUR_ARG_CONTAINER(std::map)
OUTPUT_FOUR_ARG_CONTAINER(std::multimap)
OUTPUT_FOUR_ARG_CONTAINER(std::unordered_set)
OUTPUT_FOUR_ARG_CONTAINER(std::unordered_multiset)
#undef OUTPUT_FOUR_ARG_CONTAINER
#define OUTPUT_FIVE_ARG_CONTAINER(Sequence)                                  \
  template <class T1, class T2, class T3, class T4, class T5>                \
  inline std::ostream& operator<<(std::ostream& out,                         \
                                  const Sequence<T1, T2, T3, T4, T5>& seq) { \
    google::PrintSequence(out, seq.begin(), seq.end());                      \
    return out;                                                              \
  }
OUTPUT_FIVE_ARG_CONTAINER(std::unordered_map)
OUTPUT_FIVE_ARG_CONTAINER(std::unordered_multimap)
#undef OUTPUT_FIVE_ARG_CONTAINER
template <class First, class Second>
inline std::ostream& operator<<(std::ostream& out,
                                const std::pair<First, Second>& p) {
  out << '(' << p.first << ", " << p.second << ')';
  return out;
}
namespace google {
template <class Iter>
inline void PrintSequence(std::ostream& out, Iter begin, Iter end) {
  for (int i = 0; begin != end && i < 100; ++i, ++begin) {
    if (i > 0) out << ' ';
    out << *begin;
  }
  if (begin != end) {
    out << " ...";
  }
}
}  
namespace std {
using ::operator<<;
}
#endif  