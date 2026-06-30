#ifndef TENSORSTORE_KVSTORE_KVSTACK_RANGE_MAP_H_
#define TENSORSTORE_KVSTORE_KVSTACK_RANGE_MAP_H_
#include <cassert>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include "absl/container/btree_set.h"
#include "tensorstore/kvstore/key_range.h"
namespace tensorstore {
namespace internal_kvstack {
template <typename V>
class KeyRangeMap {
  struct Compare;
 public:
  struct Value {
    KeyRange range;
    V value;
  };
  using value_type = Value;
  using const_iterator =
      typename absl::btree_set<Value, Compare>::const_iterator;
  const_iterator begin() const { return table_.begin(); }
  const_iterator end() const { return table_.end(); }
  const_iterator range_containing(std::string_view key) const {
    auto it = range_containing_impl(key);
    return Contains(it->range, key) ? it : table_.end();
  }
  template <typename V2>
  void Set(KeyRange range, V2&& value) {
    Erase(range);
    [[maybe_unused]] auto insert_result =
        table_.insert(Value{std::move(range), std::forward<V2>(value)});
    assert(insert_result.second);
  }
  void Erase(const KeyRange& range) {
    const_iterator it = range_containing_impl(range.inclusive_min);
    if (it != table_.end()) {
      if (range.inclusive_min > it->range.inclusive_min) {
        std::string tmp = range.inclusive_min;
        std::swap(const_cast<KeyRange&>(it->range).exclusive_max, tmp);
        it = table_.insert(
            it,
            Value{KeyRange(range.inclusive_min, std::move(tmp)), it->value});
      }
      while (it != table_.end() && Contains(range, it->range)) {
        it = table_.erase(it);
      }
      if (it != table_.end() && it->range.inclusive_min < range.exclusive_max) {
#if 0
        auto node = table_.extract(it);
        node.value().range.inclusive_min = range.exclusive_max;
        auto insert_result = table_.insert(std::move(node));
        assert(insert_result.inserted);
#endif
        const_cast<KeyRange&>(it->range).inclusive_min = range.exclusive_max;
      }
    }
  }
  template <typename Fn>
  void VisitRange(const KeyRange& range, Fn&& fn) const {
    if (range.empty()) return;
    auto it = range_containing_impl(range.inclusive_min);
    auto end = range.exclusive_max.empty()
                   ? table_.end()
                   : table_.lower_bound(range.exclusive_max);
    for (; it != end; ++it) {
      KeyRange intersect = Intersect(range, it->range);
      if (!intersect.empty()) {
        fn(intersect, it->value);
      }
    }
  }
 private:
  const_iterator range_containing_impl(std::string_view key) const {
    const_iterator it = table_.lower_bound(key);
    if (it == table_.end() || it->range.inclusive_min > key) {
      if (it != table_.begin() && !table_.empty()) {
        return std::prev(it);
      }
    }
    return it;
  }
  struct Compare {
    using is_transparent = void;
    bool operator()(const Value& a, const Value& b) const {
      return a.range.inclusive_min < b.range.inclusive_min;
    }
    bool operator()(const Value& a, std::string_view b) const {
      return a.range.inclusive_min < b;
    }
    bool operator()(std::string_view a, const Value& b) const {
      return a < b.range.inclusive_min;
    }
  };
  absl::btree_set<Value, Compare> table_;
};
}  
}  
#endif  