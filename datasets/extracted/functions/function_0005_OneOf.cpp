#ifndef LANGSVR_ONE_OF_H_
#define LANGSVR_ONE_OF_H_
#include <memory>
#include <type_traits>
#include <utility>
#include "langsvr/traits.h"
namespace langsvr {
template <typename... TYPES>
struct OneOf {
    template <typename T>
    static constexpr bool IsValidType = TypeIsIn<std::decay_t<T>, TYPES...>;
    OneOf() = default;
    ~OneOf() { Reset(); }
    template <typename T, typename = std::enable_if_t<IsValidType<T>>>
    OneOf(T&& value) {
        Set(std::forward<T>(value));
    }
    OneOf(const OneOf& other) { *this = other; }
    OneOf(OneOf&& other) {
        ptr = other.ptr;
        kind = other.kind;
        other.ptr = nullptr;
    }
    OneOf& operator=(const OneOf& other) {
        Reset();
        auto copy = [this](auto* p) {
            if (p) {
                this->ptr = new std::decay_t<decltype(*p)>(*p);
                return true;
            }
            return false;
        };
        (copy(other.Get<TYPES>()) || ...);
        kind = other.kind;
        return *this;
    }
    template <typename T, typename = std::enable_if_t<IsValidType<T>>>
    OneOf& operator=(T&& value) {
        Set(std::forward<T>(value));
        return *this;
    }
    bool operator==(const OneOf& other) const {
        return Visit([&](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if (auto* other_value = other.Get<T>()) {
                return value == *other_value;
            }
            return false;
        });
    }
    bool operator!=(const OneOf& other) const { return !(*this == other); }
    void Reset() {
        (delete Get<TYPES>(), ...);
        ptr = nullptr;
        kind = kNoKind;
    }
    template <typename T, typename = std::enable_if_t<IsValidType<T>>>
    void Set(T&& value) {
        using D = std::decay_t<T>;
        Reset();
        kind = static_cast<uint8_t>(TypeIndex<D, TYPES...>);
        ptr = new D(std::forward<T>(value));
    }
    template <typename T>
    bool Is() const {
        return kind == TypeIndex<T, TYPES...>;
    }
    template <typename T>
    T* Get() {
        return this->Is<T>() ? static_cast<T*>(ptr) : nullptr;
    }
    template <typename T>
    const T* Get() const {
        return this->Is<T>() ? static_cast<const T*>(ptr) : nullptr;
    }
    template <typename F>
    auto Visit(F&& cb) const {
        using FIRST = typename std::tuple_element<0, std::tuple<TYPES...>>::type;
        using RET = decltype(cb(std::declval<FIRST&>()));
        if constexpr (std::is_void_v<RET>) {
            auto call = [&](auto* p) {
                if (p) {
                    cb(*p);
                }
            };
            (call(Get<TYPES>()), ...);
        } else {
            RET ret{};
            auto call = [&](auto* p) {
                if (p) {
                    ret = cb(*p);
                }
            };
            (call(Get<TYPES>()), ...);
            return ret;
        }
    }
  private:
    static_assert(sizeof...(TYPES) < 255);
    static constexpr uint8_t kNoKind = 0xff;
    void* ptr = nullptr;
    uint8_t kind = 0xff;
};
}  
#endif  