#ifndef TENSORSTORE_INTERNAL_CONTAINER_HETEROGENEOUS_CONTAINER_H_
#define TENSORSTORE_INTERNAL_CONTAINER_HETEROGENEOUS_CONTAINER_H_
#include <functional>
#include "absl/container/flat_hash_set.h"
namespace tensorstore {
namespace internal {
template <typename T>
struct SupportsHeterogeneous : public T {
  using is_transparent = void;
};
template <typename EntryPointer, typename T, auto Getter>
struct KeyAdapter {
  template <typename U,
            typename = std::enable_if_t<std::is_convertible_v<U, T>>>
  KeyAdapter(U&& key) : value(std::forward<U>(key)) {}
  KeyAdapter(const EntryPointer& e) : value(std::invoke(Getter, *e)) {}
  template <typename H>
  friend H AbslHashValue(H h, const KeyAdapter& key) {
    return H::combine(std::move(h), key.value);
  }
  friend bool operator==(const KeyAdapter& a, const KeyAdapter& b) {
    return a.value == b.value;
  }
  T value;
};
template <typename EntryPointer, typename T, auto Getter>
using HeterogeneousHashSet = absl::flat_hash_set<
    EntryPointer,
    SupportsHeterogeneous<absl::Hash<KeyAdapter<EntryPointer, T, Getter>>>,
    SupportsHeterogeneous<std::equal_to<KeyAdapter<EntryPointer, T, Getter>>>>;
}  
}  
#endif  