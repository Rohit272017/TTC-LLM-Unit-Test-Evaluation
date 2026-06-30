#ifndef QUICHE_COMMON_QUICHE_INTRUSIVE_LIST_H_
#define QUICHE_COMMON_QUICHE_INTRUSIVE_LIST_H_
#include <stddef.h>
#include <cstddef>
#include <iterator>
#include "quiche/common/platform/api/quiche_export.h"
namespace quiche {
template <typename T, typename ListID>
class QuicheIntrusiveList;
template <typename T, typename ListID = void>
class QUICHE_EXPORT QuicheIntrusiveLink {
 protected:
  QuicheIntrusiveLink() : next_(nullptr), prev_(nullptr) {}
#ifndef SWIG
  QuicheIntrusiveLink(const QuicheIntrusiveLink&) = delete;
  QuicheIntrusiveLink& operator=(const QuicheIntrusiveLink&) = delete;
#endif  
 private:
  friend class QuicheIntrusiveList<T, ListID>;
  T* cast_to_derived() { return static_cast<T*>(this); }
  const T* cast_to_derived() const { return static_cast<const T*>(this); }
  QuicheIntrusiveLink* next_;
  QuicheIntrusiveLink* prev_;
};
template <typename T, typename ListID = void>
class QUICHE_EXPORT QuicheIntrusiveList {
  template <typename QualifiedT, typename QualifiedLinkT>
  class iterator_impl;
 public:
  typedef T value_type;
  typedef value_type* pointer;
  typedef const value_type* const_pointer;
  typedef value_type& reference;
  typedef const value_type& const_reference;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef QuicheIntrusiveLink<T, ListID> link_type;
  typedef iterator_impl<T, link_type> iterator;
  typedef iterator_impl<const T, const link_type> const_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  QuicheIntrusiveList() { clear(); }
#ifndef SWIG
  QuicheIntrusiveList(QuicheIntrusiveList&& src) noexcept {
    clear();
    if (src.empty()) return;
    sentinel_link_.next_ = src.sentinel_link_.next_;
    sentinel_link_.prev_ = src.sentinel_link_.prev_;
    sentinel_link_.prev_->next_ = &sentinel_link_;
    sentinel_link_.next_->prev_ = &sentinel_link_;
    src.clear();
  }
#endif  
  iterator begin() { return iterator(sentinel_link_.next_); }
  const_iterator begin() const { return const_iterator(sentinel_link_.next_); }
  iterator end() { return iterator(&sentinel_link_); }
  const_iterator end() const { return const_iterator(&sentinel_link_); }
  reverse_iterator rbegin() { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }
  reverse_iterator rend() { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }
  bool empty() const { return (sentinel_link_.next_ == &sentinel_link_); }
  size_type size() const { return std::distance(begin(), end()); }
  size_type max_size() const { return size_type(-1); }
  reference front() { return *begin(); }
  const_reference front() const { return *begin(); }
  reference back() { return *(--end()); }
  const_reference back() const { return *(--end()); }
  static iterator insert(iterator position, T* obj) {
    return insert_link(position.link(), obj);
  }
  void push_front(T* obj) { insert(begin(), obj); }
  void push_back(T* obj) { insert(end(), obj); }
  static iterator erase(T* obj) {
    link_type* obj_link = obj;
    obj_link->next_->prev_ = obj_link->prev_;
    obj_link->prev_->next_ = obj_link->next_;
    link_type* next_link = obj_link->next_;
    obj_link->next_ = nullptr;
    obj_link->prev_ = nullptr;
    return iterator(next_link);
  }
  static iterator erase(iterator position) {
    return erase(position.operator->());
  }
  void pop_front() { erase(begin()); }
  void pop_back() { erase(--end()); }
  static bool is_linked(const T* obj) {
    return obj->link_type::next_ != nullptr;
  }
  void clear() {
    sentinel_link_.next_ = sentinel_link_.prev_ = &sentinel_link_;
  }
  void swap(QuicheIntrusiveList& x) {
    QuicheIntrusiveList tmp;
    tmp.splice(tmp.begin(), *this);
    this->splice(this->begin(), x);
    x.splice(x.begin(), tmp);
  }
  void splice(iterator pos, QuicheIntrusiveList& src) {
    splice(pos, src.begin(), src.end());
  }
  void splice(iterator pos, iterator i) { splice(pos, i, std::next(i)); }
  void splice(iterator pos, iterator first, iterator last) {
    if (first == last) return;
    link_type* const last_prev = last.link()->prev_;
    first.link()->prev_->next_ = last.operator->();
    last.link()->prev_ = first.link()->prev_;
    first.link()->prev_ = pos.link()->prev_;
    pos.link()->prev_->next_ = first.operator->();
    last_prev->next_ = pos.operator->();
    pos.link()->prev_ = last_prev;
  }
 private:
  static iterator insert_link(link_type* next_link, T* obj) {
    link_type* obj_link = obj;
    obj_link->next_ = next_link;
    link_type* const initial_next_prev = next_link->prev_;
    obj_link->prev_ = initial_next_prev;
    initial_next_prev->next_ = obj_link;
    next_link->prev_ = obj_link;
    return iterator(obj_link);
  }
  template <typename QualifiedT, typename QualifiedLinkT>
  class QUICHE_EXPORT iterator_impl {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = QualifiedT;
    using difference_type = std::ptrdiff_t;
    using pointer = QualifiedT*;
    using reference = QualifiedT&;
    iterator_impl() = default;
    iterator_impl(QualifiedLinkT* link) : link_(link) {}
    iterator_impl(const iterator_impl& x) = default;
    iterator_impl& operator=(const iterator_impl& x) = default;
    template <typename U, typename V>
    iterator_impl(const iterator_impl<U, V>& x) : link_(x.link_) {}
    template <typename U, typename V>
    bool operator==(const iterator_impl<U, V>& x) const {
      return link_ == x.link_;
    }
    template <typename U, typename V>
    bool operator!=(const iterator_impl<U, V>& x) const {
      return link_ != x.link_;
    }
    reference operator*() const { return *operator->(); }
    pointer operator->() const { return link_->cast_to_derived(); }
    QualifiedLinkT* link() const { return link_; }
#ifndef SWIG  
    iterator_impl& operator++() {
      link_ = link_->next_;
      return *this;
    }
    iterator_impl operator++(int ) {
      iterator_impl tmp = *this;
      ++*this;
      return tmp;
    }
    iterator_impl& operator--() {
      link_ = link_->prev_;
      return *this;
    }
    iterator_impl operator--(int ) {
      iterator_impl tmp = *this;
      --*this;
      return tmp;
    }
#endif  
   private:
    template <typename U, typename V>
    friend class iterator_impl;
    QualifiedLinkT* link_ = nullptr;
  };
  link_type sentinel_link_;
  QuicheIntrusiveList(const QuicheIntrusiveList&);
  void operator=(const QuicheIntrusiveList&);
};
}  
#endif  