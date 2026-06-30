#ifndef XLA_SERVICE_MAPPED_PTR_CONTAINER_SORTER_H_
#define XLA_SERVICE_MAPPED_PTR_CONTAINER_SORTER_H_
#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
namespace xla {
template <typename PointedToTy>
class MappedPtrContainerSorter {
 public:
  using MapPtrFn = absl::FunctionRef<const PointedToTy*(const PointedToTy*)>;
  using UnmappedPtrIndexFn = absl::FunctionRef<size_t(const PointedToTy*)>;
  static UnmappedPtrIndexFn IndexBeforeMappedElementsFn();
  static UnmappedPtrIndexFn IndexAfterMappedElementsFn();
  static UnmappedPtrIndexFn InvalidIndexFn();
  template <typename OrderedTy, typename UnorderedTy>
  static absl::Status Sort(MapPtrFn map_ptr, UnmappedPtrIndexFn unmapped_index,
                           const OrderedTy& ordered_container,
                           UnorderedTy& unordered_container);
 private:
  class SortedIndices {
   public:
    SortedIndices(size_t max_partial_order_exclusive,
                  size_t unordered_container_size)
        : max_partial_order_exclusive_(max_partial_order_exclusive),
          unordered_container_size_(unordered_container_size),
          mapped_element_indices_by_partial_order_(
              max_partial_order_exclusive) {}
    absl::Status AddMappedElement(size_t unordered_container_index,
                                  size_t partial_order);
    void AddUnmappedElement(size_t unordered_container_index,
                            size_t target_index_amongst_mapped_elements);
    std::string ToString() const;
    absl::StatusOr<std::vector<size_t>> Flatten() const;
   private:
    SortedIndices() = delete;
    size_t max_partial_order_exclusive_;
    size_t unordered_container_size_;
    std::vector<std::vector<size_t>> mapped_element_indices_by_partial_order_;
    absl::flat_hash_map<size_t, std::vector<size_t>>
        target_index_to_unmapped_element_index_;
  };
  static size_t IndexBeforeMappedElements() {
    return std::numeric_limits<size_t>::max() - 2;
  }
  static size_t IndexAfterMappedElements() {
    return std::numeric_limits<size_t>::max() - 1;
  }
  static size_t InvalidIndex() { return std::numeric_limits<size_t>::max(); }
  template <typename OrderedTy, typename UnorderedTy>
  static absl::StatusOr<std::vector<size_t>> ComputeNewIndices(
      MapPtrFn map_ptr, UnmappedPtrIndexFn unmapped_index,
      const OrderedTy& ordered_container,
      const UnorderedTy& unordered_container);
  template <typename UnorderedTy>
  static void Reorder(std::vector<size_t> new_indices,
                      UnorderedTy& unordered_container);
};
namespace mapped_ptr_container_sorter_internal {
template <typename I, typename O>
struct PtrGetter {
  static O Get(I i);
};
template <typename T>
struct PtrGetter<T* const&, const T*> {
  static const T* Get(T* const& p) { return p; }
};
template <typename T>
struct PtrGetter<T const* const&, const T*> {
  static const T* Get(T const* const& p) { return p; }
};
template <typename T>
struct PtrGetter<T*&, T*> {
  static T* Get(T*& p) { return p; }
};
template <typename T>
struct PtrGetter<const std::unique_ptr<T>&, const T*> {
  static const T* Get(const std::unique_ptr<T>& p) { return p.get(); }
};
template <typename T>
struct PtrGetter<std::unique_ptr<T>&, T*> {
  static T* Get(std::unique_ptr<T>& p) { return p.get(); }
};
}  
template <typename PointedToTy>
typename MappedPtrContainerSorter<PointedToTy>::UnmappedPtrIndexFn
MappedPtrContainerSorter<PointedToTy>::IndexBeforeMappedElementsFn() {
  static const auto fn = [](const PointedToTy*) {
    return IndexBeforeMappedElements();
  };
  return fn;
}
template <typename PointedToTy>
typename MappedPtrContainerSorter<PointedToTy>::UnmappedPtrIndexFn
MappedPtrContainerSorter<PointedToTy>::IndexAfterMappedElementsFn() {
  static const auto fn = [](const PointedToTy*) {
    return IndexAfterMappedElements();
  };
  return fn;
}
template <typename PointedToTy>
typename MappedPtrContainerSorter<PointedToTy>::UnmappedPtrIndexFn
MappedPtrContainerSorter<PointedToTy>::InvalidIndexFn() {
  static const auto fn = [](const PointedToTy*) { return InvalidIndex(); };
  return fn;
}
template <typename PointedToTy>
absl::Status
MappedPtrContainerSorter<PointedToTy>::SortedIndices::AddMappedElement(
    size_t unordered_container_index, size_t partial_order) {
  if (partial_order >= mapped_element_indices_by_partial_order_.size()) {
    return InternalStrCat("invalid partial order: ", partial_order, " v max(",
                          mapped_element_indices_by_partial_order_.size(), ")");
  }
  mapped_element_indices_by_partial_order_[partial_order].push_back(
      unordered_container_index);
  return absl::OkStatus();
}
template <typename PointedToTy>
void MappedPtrContainerSorter<PointedToTy>::SortedIndices::AddUnmappedElement(
    size_t unordered_container_index,
    size_t target_index_amongst_mapped_elements) {
  target_index_to_unmapped_element_index_[target_index_amongst_mapped_elements]
      .push_back(unordered_container_index);
}
template <typename PointedToTy>
std::string MappedPtrContainerSorter<PointedToTy>::SortedIndices::ToString()
    const {
  std::vector<std::string> mapped_element_strs;
  mapped_element_strs.reserve(mapped_element_indices_by_partial_order_.size());
  for (const auto& indices : mapped_element_indices_by_partial_order_) {
    mapped_element_strs.push_back(
        absl::StrCat("[", absl::StrJoin(indices, ", "), "]"));
  }
  std::vector<std::string> unmapped_element_strs;
  unmapped_element_strs.reserve(target_index_to_unmapped_element_index_.size());
  for (const auto& kv : target_index_to_unmapped_element_index_) {
    std::string key = absl::StrCat(kv.first);
    if (kv.first == IndexBeforeMappedElements()) {
      key = "before_mapped";
    }
    if (kv.first == IndexAfterMappedElements()) {
      key = "after_mapped";
    }
    if (kv.first == InvalidIndex()) {
      key = "invalid";
    }
    unmapped_element_strs.push_back(
        absl::StrCat(key, ": [", absl::StrJoin(kv.second, ", "), "]"));
  }
  return absl::StrCat(
      "max_partial_order_exclusive_: ", max_partial_order_exclusive_, "\n",
      "unordered_container_size_: ", unordered_container_size_, "\n",
      "mapped_element_indices_by_partial_order_: [",
      absl::StrJoin(mapped_element_strs, ", "), "]\n",
      "target_index_to_unmapped_element_index_: {",
      absl::StrJoin(unmapped_element_strs, ", "), "}\n");
}
template <typename PointedToTy>
absl::StatusOr<std::vector<size_t>>
MappedPtrContainerSorter<PointedToTy>::SortedIndices::Flatten() const {
  std::vector<size_t> result(unordered_container_size_, InvalidIndex());
  size_t next_available_index = 0;
  auto next_index_fn = [&]() -> absl::StatusOr<size_t> {
    if (next_available_index >= unordered_container_size_) {
      return InternalStrCat(
          "invalid unordered_container index: ", next_available_index,
          " v size(", unordered_container_size_, ")");
    }
    return next_available_index++;
  };
  if (target_index_to_unmapped_element_index_.contains(
          IndexBeforeMappedElements())) {
    const auto& indices =
        target_index_to_unmapped_element_index_.at(IndexBeforeMappedElements());
    for (size_t index : indices) {
      TF_ASSIGN_OR_RETURN(result[index], next_index_fn());
    }
  }
  size_t num_inserted_mapped_elements = 0;
  for (const auto& mapped_element_indices :
       mapped_element_indices_by_partial_order_) {
    for (size_t mapped_element_index : mapped_element_indices) {
      TF_ASSIGN_OR_RETURN(result[mapped_element_index], next_index_fn());
      ++num_inserted_mapped_elements;
      if (target_index_to_unmapped_element_index_.contains(
              num_inserted_mapped_elements - 1)) {
        const auto& unmapped_element_indices =
            target_index_to_unmapped_element_index_.at(
                num_inserted_mapped_elements - 1);
        for (size_t unmapped_element_index : unmapped_element_indices) {
          TF_ASSIGN_OR_RETURN(result[unmapped_element_index], next_index_fn());
        }
      }
    }
  }
  if (target_index_to_unmapped_element_index_.contains(
          IndexAfterMappedElements())) {
    const auto& indices =
        target_index_to_unmapped_element_index_.at(IndexAfterMappedElements());
    for (size_t index : indices) {
      TF_ASSIGN_OR_RETURN(result[index], next_index_fn());
    }
  }
  absl::flat_hash_set<size_t> used_indices;
  for (size_t index : result) {
    if (used_indices.contains(index)) {
      return InternalStrCat(
          "2 elements in unordered_container are destined for the same "
          "index: ",
          index);
    }
    if (index >= unordered_container_size_) {
      return InvalidArgumentStrCat("invalid unordered_container index: ", index,
                                   " v size(", unordered_container_size_, ")");
    }
  }
  return result;
}
template <typename PointedToTy>
template <typename OrderedTy, typename UnorderedTy>
absl::StatusOr<std::vector<size_t>>
MappedPtrContainerSorter<PointedToTy>::ComputeNewIndices(
    MapPtrFn map_ptr, UnmappedPtrIndexFn unmapped_index,
    const OrderedTy& ordered_container,
    const UnorderedTy& unordered_container) {
  using UnorderedPtrGetter = mapped_ptr_container_sorter_internal::PtrGetter<
      typename UnorderedTy::const_reference, const PointedToTy*>;
  using OrderedPtrGetter = mapped_ptr_container_sorter_internal::PtrGetter<
      typename OrderedTy::const_reference, const PointedToTy*>;
  if (unordered_container.size() >= IndexBeforeMappedElements()) {
    return InvalidArgumentStrCat("Unordered container is too large to sort.");
  }
  absl::flat_hash_set<const PointedToTy*> unordered_ptrs;
  for (const auto& unordered_element : unordered_container) {
    const PointedToTy* ptr = UnorderedPtrGetter::Get(unordered_element);
    unordered_ptrs.insert(ptr);
  }
  absl::flat_hash_map<const PointedToTy*, std::list<size_t>>
      mapped_ptr_to_partial_order;
  size_t next_partial_order_value = 0;
  for (const auto& ordered_element : ordered_container) {
    const PointedToTy* ordered_ptr = OrderedPtrGetter::Get(ordered_element);
    const PointedToTy* unordered_ptr = map_ptr(ordered_ptr);
    if (!unordered_ptr) {
      continue;
    }
    if (!unordered_ptrs.contains(unordered_ptr)) {
      continue;
    }
    mapped_ptr_to_partial_order[unordered_ptr].push_back(
        next_partial_order_value);
    ++next_partial_order_value;
  }
  SortedIndices result(next_partial_order_value, unordered_container.size());
  for (size_t i = 0; i < unordered_container.size(); ++i) {
    const PointedToTy* ptr = UnorderedPtrGetter::Get(unordered_container[i]);
    if (!mapped_ptr_to_partial_order.contains(ptr)) {
      result.AddUnmappedElement(i, unmapped_index(ptr));
      continue;
    }
    auto& index_list = mapped_ptr_to_partial_order[ptr];
    TF_RETURN_IF_ERROR(result.AddMappedElement(i, index_list.front()));
    if (index_list.size() > 1) {
      index_list.pop_front();
    }
  }
  VLOG(5) << "Pre flatten unordered_container result:\n" << result.ToString();
  return result.Flatten();
}
template <typename PointedToTy>
template <typename UnorderedTy>
void MappedPtrContainerSorter<PointedToTy>::Reorder(
    std::vector<size_t> new_indices, UnorderedTy& unordered_container) {
  size_t old_pos = 0;
  while (old_pos < new_indices.size()) {
    size_t new_pos = new_indices[old_pos];
    if (old_pos == new_pos) {
      ++old_pos;
      continue;
    }
    std::swap(new_indices[old_pos], new_indices[new_pos]);
    std::swap(unordered_container[old_pos], unordered_container[new_pos]);
  }
}
template <typename PointedToTy>
template <typename OrderedTy, typename UnorderedTy>
absl::Status MappedPtrContainerSorter<PointedToTy>::Sort(
    MapPtrFn map_ptr, UnmappedPtrIndexFn unmapped_index,
    const OrderedTy& ordered_container, UnorderedTy& unordered_container) {
  std::vector<size_t> indices;
  TF_ASSIGN_OR_RETURN(
      indices, ComputeNewIndices(map_ptr, unmapped_index, ordered_container,
                                 unordered_container));
  Reorder(std::move(indices), unordered_container);
  return absl::OkStatus();
}
}  
#endif  