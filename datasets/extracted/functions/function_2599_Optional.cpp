#ifndef LANGSVR_OPTIONAL_H_
#define LANGSVR_OPTIONAL_H_
#include <cassert>
#include <type_traits>
#include <utility>
namespace langsvr {
template <typename T>
struct Optional {
    Optional() = default;
    ~Optional() { Reset(); }
    Optional(const Optional& other) { *this = other; }
    Optional(Optional&& other) { *this = std::move(other); }
    Optional(const T& other) { *this = other; }
    Optional(T&& other) { *this = std::move(other); }
    void Reset() {
        if (ptr) {
            delete ptr;
            ptr = nullptr;
        }
    }
    Optional& operator=(const Optional& other) {
        Reset();
        if (other.ptr) {
            ptr = new T(*other);
        }
        return *this;
    }
    Optional& operator=(Optional&& other) {
        Reset();
        ptr = other.ptr;
        other.ptr = nullptr;
        return *this;
    }
    Optional& operator=(const T& value) {
        Reset();
        if (!ptr) {
            ptr = new T(value);
        }
        return *this;
    }
    Optional& operator=(T&& value) {
        Reset();
        if (!ptr) {
            ptr = new T(std::move(value));
        }
        return *this;
    }
    operator bool() const { return ptr != nullptr; }
    bool operator!() const { return ptr == nullptr; }
    T* operator->() { return &Get(); }
    const T* operator->() const { return &Get(); }
    T& operator*() { return Get(); }
    const T& operator*() const { return Get(); }
    template <typename V>
    bool operator==(V&& value) const {
        if constexpr (std::is_same_v<Optional, std::decay_t<V>>) {
            return (!*this && !value) || (*this && value && (Get() == value.Get()));
        } else {
            if (!ptr) {
                return false;
            }
            return Get() == std::forward<V>(value);
        }
    }
    template <typename V>
    bool operator!=(V&& value) const {
        return !(*this == std::forward<V>(value));
    }
  private:
    T& Get() {
        assert(ptr);
        return *ptr;
    }
    const T& Get() const {
        assert(ptr);
        return *ptr;
    }
    T* ptr = nullptr;
};
}  
#endif  