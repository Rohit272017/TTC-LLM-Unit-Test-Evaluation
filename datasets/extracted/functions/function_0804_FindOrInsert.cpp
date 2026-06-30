#ifndef TENSORSTORE_INTERNAL_CONTAINER_HASH_SET_OF_ANY_H_
#define TENSORSTORE_INTERNAL_CONTAINER_HASH_SET_OF_ANY_H_
#include <stddef.h>
#include <cassert>
#include <functional>
#include <memory>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include "absl/container/flat_hash_set.h"
namespace tensorstore {
namespace internal {
class HashSetOfAny {
 public:
  class Entry {
   public:
    virtual ~Entry() = default;
   private:
    friend class HashSetOfAny;
    size_t hash_;
  };
  template <typename DerivedEntry, typename MakeEntry>
  std::pair<DerivedEntry*, bool> FindOrInsert(
      typename DerivedEntry::KeyParam key, MakeEntry make_entry,
      const std::type_info& derived_type = typeid(DerivedEntry)) {
    static_assert(std::is_base_of_v<Entry, DerivedEntry>);
    KeyFor<DerivedEntry> key_wrapper{derived_type, key};
    size_t hash = Hash{}(key_wrapper);
    auto it = entries_.find(key_wrapper);
    if (it != entries_.end()) {
      return {static_cast<DerivedEntry*>(*it), false};
    }
    std::unique_ptr<DerivedEntry> derived_entry = make_entry();
    auto* entry = static_cast<Entry*>(derived_entry.get());
    assert(derived_type == typeid(*entry));
    entry->hash_ = hash;
    [[maybe_unused]] auto inserted = entries_.insert(entry).second;
    assert(inserted);
    return {derived_entry.release(), true};
  }
  void erase(Entry* entry) { entries_.erase(entry); }
  void clear() { entries_.clear(); }
  auto begin() { return entries_.begin(); }
  auto begin() const { return entries_.begin(); }
  auto end() { return entries_.end(); }
  auto end() const { return entries_.end(); }
  size_t size() const { return entries_.size(); }
  bool empty() const { return entries_.empty(); }
 private:
  template <typename DerivedEntry>
  struct KeyFor {
    const std::type_info& derived_type;
    typename DerivedEntry::KeyParam key;
    friend bool operator==(Entry* entry, const KeyFor<DerivedEntry>& other) {
      return typeid(*entry) == other.derived_type &&
             static_cast<DerivedEntry*>(entry)->key() == other.key;
    }
  };
  struct Hash {
    using is_transparent = void;
    template <typename DerivedEntry>
    size_t operator()(KeyFor<DerivedEntry> key) const {
      return absl::HashOf(std::type_index(key.derived_type), key.key);
    }
    size_t operator()(Entry* entry) const { return entry->hash_; }
  };
  struct Eq : public std::equal_to<void> {
    using is_transparent = void;
  };
  absl::flat_hash_set<Entry*, Hash, Eq> entries_;
};
}  
}  
#endif  