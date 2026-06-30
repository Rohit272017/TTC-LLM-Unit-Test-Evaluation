#include "tensorflow/core/graph/edgeset.h"
namespace tensorflow {
std::pair<EdgeSet::const_iterator, bool> EdgeSet::insert(value_type value) {
  RegisterMutation();
  const_iterator ci;
  ci.Init(this);
  auto s = get_set();
  if (!s) {
    for (int i = 0; i < kInline; i++) {
      if (ptrs_[i] == value) {
        ci.array_iter_ = &ptrs_[i];
        return std::make_pair(ci, false);
      }
    }
    for (int i = 0; i < kInline; i++) {
      if (ptrs_[i] == nullptr) {
        ptrs_[i] = value;
        ci.array_iter_ = &ptrs_[i];
        return std::make_pair(ci, true);
      }
    }
    s = new gtl::FlatSet<const Edge*>;
    s->insert(reinterpret_cast<const Edge**>(std::begin(ptrs_)),
              reinterpret_cast<const Edge**>(std::end(ptrs_)));
    ptrs_[0] = this;
    ptrs_[1] = s;
  }
  auto p = s->insert(value);
  ci.tree_iter_ = p.first;
  return std::make_pair(ci, p.second);
}
EdgeSet::size_type EdgeSet::erase(key_type key) {
  RegisterMutation();
  auto s = get_set();
  if (!s) {
    for (int i = 0; i < kInline; i++) {
      if (ptrs_[i] == key) {
        size_t n = size();
        ptrs_[i] = ptrs_[n - 1];
        ptrs_[n - 1] = nullptr;
        return 1;
      }
    }
    return 0;
  } else {
    return s->erase(key);
  }
}
}  