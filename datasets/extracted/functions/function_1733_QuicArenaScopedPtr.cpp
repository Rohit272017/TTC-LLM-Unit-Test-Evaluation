#ifndef QUICHE_QUIC_CORE_QUIC_ARENA_SCOPED_PTR_H_
#define QUICHE_QUIC_CORE_QUIC_ARENA_SCOPED_PTR_H_
#include <cstdint>  
#include "quiche/quic/platform/api/quic_export.h"
#include "quiche/quic/platform/api/quic_logging.h"
namespace quic {
template <typename T>
class QUICHE_NO_EXPORT QuicArenaScopedPtr {
  static_assert(alignof(T*) > 1,
                "QuicArenaScopedPtr can only store objects that are aligned to "
                "greater than 1 byte.");
 public:
  QuicArenaScopedPtr();
  explicit QuicArenaScopedPtr(T* value);
  template <typename U>
  QuicArenaScopedPtr(QuicArenaScopedPtr<U>&& other);  
  template <typename U>
  QuicArenaScopedPtr& operator=(QuicArenaScopedPtr<U>&& other);
  ~QuicArenaScopedPtr();
  T* get() const;
  T& operator*() const;
  T* operator->() const;
  void swap(QuicArenaScopedPtr& other);
  void reset(T* value = nullptr);
  bool is_from_arena();
 private:
  template <typename U>
  friend class QuicArenaScopedPtr;
  template <uint32_t ArenaSize>
  friend class QuicOneBlockArena;
  enum class ConstructFrom { kHeap, kArena };
  QuicArenaScopedPtr(void* value, ConstructFrom from);
  QuicArenaScopedPtr(const QuicArenaScopedPtr&) = delete;
  QuicArenaScopedPtr& operator=(const QuicArenaScopedPtr&) = delete;
  static const uintptr_t kFromArenaMask = 0x1;
  void* value_;
};
template <typename T>
bool operator==(const QuicArenaScopedPtr<T>& left,
                const QuicArenaScopedPtr<T>& right) {
  return left.get() == right.get();
}
template <typename T>
bool operator!=(const QuicArenaScopedPtr<T>& left,
                const QuicArenaScopedPtr<T>& right) {
  return left.get() != right.get();
}
template <typename T>
bool operator==(std::nullptr_t, const QuicArenaScopedPtr<T>& right) {
  return nullptr == right.get();
}
template <typename T>
bool operator!=(std::nullptr_t, const QuicArenaScopedPtr<T>& right) {
  return nullptr != right.get();
}
template <typename T>
bool operator==(const QuicArenaScopedPtr<T>& left, std::nullptr_t) {
  return left.get() == nullptr;
}
template <typename T>
bool operator!=(const QuicArenaScopedPtr<T>& left, std::nullptr_t) {
  return left.get() != nullptr;
}
template <typename T>
QuicArenaScopedPtr<T>::QuicArenaScopedPtr() : value_(nullptr) {}
template <typename T>
QuicArenaScopedPtr<T>::QuicArenaScopedPtr(T* value)
    : QuicArenaScopedPtr(value, ConstructFrom::kHeap) {}
template <typename T>
template <typename U>
QuicArenaScopedPtr<T>::QuicArenaScopedPtr(QuicArenaScopedPtr<U>&& other)
    : value_(other.value_) {
  static_assert(
      std::is_base_of<T, U>::value || std::is_same<T, U>::value,
      "Cannot construct QuicArenaScopedPtr; type is not derived or same.");
  other.value_ = nullptr;
}
template <typename T>
template <typename U>
QuicArenaScopedPtr<T>& QuicArenaScopedPtr<T>::operator=(
    QuicArenaScopedPtr<U>&& other) {
  static_assert(
      std::is_base_of<T, U>::value || std::is_same<T, U>::value,
      "Cannot assign QuicArenaScopedPtr; type is not derived or same.");
  swap(other);
  return *this;
}
template <typename T>
QuicArenaScopedPtr<T>::~QuicArenaScopedPtr() {
  reset();
}
template <typename T>
T* QuicArenaScopedPtr<T>::get() const {
  return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(value_) &
                              ~kFromArenaMask);
}
template <typename T>
T& QuicArenaScopedPtr<T>::operator*() const {
  return *get();
}
template <typename T>
T* QuicArenaScopedPtr<T>::operator->() const {
  return get();
}
template <typename T>
void QuicArenaScopedPtr<T>::swap(QuicArenaScopedPtr& other) {
  using std::swap;
  swap(value_, other.value_);
}
template <typename T>
bool QuicArenaScopedPtr<T>::is_from_arena() {
  return (reinterpret_cast<uintptr_t>(value_) & kFromArenaMask) != 0;
}
template <typename T>
void QuicArenaScopedPtr<T>::reset(T* value) {
  if (value_ != nullptr) {
    if (is_from_arena()) {
      get()->~T();
    } else {
      delete get();
    }
  }
  QUICHE_DCHECK_EQ(0u, reinterpret_cast<uintptr_t>(value) & kFromArenaMask);
  value_ = value;
}
template <typename T>
QuicArenaScopedPtr<T>::QuicArenaScopedPtr(void* value, ConstructFrom from_arena)
    : value_(value) {
  QUICHE_DCHECK_EQ(0u, reinterpret_cast<uintptr_t>(value_) & kFromArenaMask);
  switch (from_arena) {
    case ConstructFrom::kHeap:
      break;
    case ConstructFrom::kArena:
      value_ = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(value_) |
                                       QuicArenaScopedPtr<T>::kFromArenaMask);
      break;
  }
}
}  
#endif  