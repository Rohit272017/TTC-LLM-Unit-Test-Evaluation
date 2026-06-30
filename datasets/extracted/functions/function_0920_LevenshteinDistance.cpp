#ifndef TENSORFLOW_CORE_LIB_GTL_EDIT_DISTANCE_H_
#define TENSORFLOW_CORE_LIB_GTL_EDIT_DISTANCE_H_
#include <numeric>
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
namespace tensorflow {
namespace gtl {
template <typename T, typename Cmp>
inline int64_t LevenshteinDistance(const gtl::ArraySlice<T> s,
                                   const gtl::ArraySlice<T> t, const Cmp& cmp) {
  const int64_t s_size = s.size();
  const int64_t t_size = t.size();
  if (t_size > s_size) return LevenshteinDistance(t, s, cmp);
  const T* s_data = s.data();
  const T* t_data = t.data();
  if (t_size == 0) return s_size;
  if (s == t) return 0;
  absl::InlinedVector<int64_t, 32UL> scratch_holder(t_size);
  int64_t* scratch = scratch_holder.data();
  for (size_t j = 1; j < t_size; ++j) scratch[j - 1] = j;
  for (size_t i = 1; i <= s_size; ++i) {
    int substitution_base_cost = i - 1;
    int insertion_cost = i + 1;
    for (size_t j = 1; j <= t_size; ++j) {
      const int replacement_cost = cmp(s_data[i - 1], t_data[j - 1]) ? 0 : 1;
      const int substitution_cost = substitution_base_cost + replacement_cost;
      const int deletion_cost = scratch[j - 1] + 1;
      const int cheapest =  
          std::min(deletion_cost, std::min(insertion_cost, substitution_cost));
      substitution_base_cost = scratch[j - 1];  
      scratch[j - 1] = cheapest;                
      insertion_cost = cheapest + 1;            
    }
  }
  return scratch[t_size - 1];
}
template <typename Container1, typename Container2, typename Cmp>
inline int64_t LevenshteinDistance(const Container1& s, const Container2& t,
                                   const Cmp& cmp) {
  return LevenshteinDistance(
      gtl::ArraySlice<typename Container1::value_type>(s.data(), s.size()),
      gtl::ArraySlice<typename Container1::value_type>(t.data(), t.size()),
      cmp);
}
}  
}  
#endif  