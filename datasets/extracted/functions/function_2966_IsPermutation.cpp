#include "xla/permutation_util.h"
#include <vector>
#include "absl/container/inlined_vector.h"
namespace xla {
bool IsPermutation(absl::Span<const int64_t> permutation) {
  absl::InlinedVector<bool, 8> seen(permutation.size(), false);
  for (int64_t p : permutation) {
    if (p < 0 || p >= permutation.size() || seen[p]) {
      return false;
    }
    seen[p] = true;
  }
  return true;
}
std::vector<int64_t> InversePermutation(
    absl::Span<const int64_t> input_permutation) {
  DCHECK(IsPermutation(input_permutation));
  std::vector<int64_t> output_permutation(input_permutation.size(), -1);
  for (size_t i = 0; i < input_permutation.size(); ++i) {
    output_permutation[input_permutation[i]] = i;
  }
  return output_permutation;
}
std::vector<int64_t> ComposePermutations(absl::Span<const int64_t> p1,
                                         absl::Span<const int64_t> p2) {
  CHECK_EQ(p1.size(), p2.size());
  std::vector<int64_t> output;
  output.reserve(p1.size());
  for (size_t i = 0; i < p1.size(); ++i) {
    output.push_back(p1.at(p2.at(i)));
  }
  return output;
}
bool IsIdentityPermutation(absl::Span<const int64_t> permutation) {
  for (int64_t i = 0; i < permutation.size(); ++i) {
    if (permutation[i] != i) {
      return false;
    }
  }
  return true;
}
}  