#ifndef XLA_SERVICE_GRAPHCYCLES_ORDERED_SET_H_
#define XLA_SERVICE_GRAPHCYCLES_ORDERED_SET_H_
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "tsl/platform/logging.h"
namespace xla {
template <typename T>
class OrderedSet {
 public:
  bool Insert(T value) {
    bool new_insertion =
        value_to_index_.insert({value, value_sequence_.size()}).second;
    if (new_insertion) {
      value_sequence_.push_back(value);
    }
    return new_insertion;
  }
  void Erase(T value) {
    auto it = value_to_index_.find(value);
    DCHECK(it != value_to_index_.end());
    value_to_index_[value_sequence_.back()] = it->second;
    std::swap(value_sequence_[it->second], value_sequence_.back());
    value_sequence_.pop_back();
    value_to_index_.erase(it);
  }
  void Reserve(size_t new_size) {
    value_to_index_.reserve(new_size);
    value_sequence_.reserve(new_size);
  }
  void Clear() {
    value_to_index_.clear();
    value_sequence_.clear();
  }
  bool Contains(T value) const { return value_to_index_.contains(value); }
  size_t Size() const { return value_sequence_.size(); }
  absl::Span<T const> GetSequence() const { return value_sequence_; }
 private:
  std::vector<T> value_sequence_;
  absl::flat_hash_map<T, int> value_to_index_;
};
}  
#endif  