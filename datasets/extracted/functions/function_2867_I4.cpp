#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_I4_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_I4_H_
#include <cstdint>
#include <limits>
#include <ostream>
#include <type_traits>
namespace shlo_ref {
struct I4 {
  int8_t data = 0;
  constexpr I4() = default;
  constexpr I4(const I4&) = default;
  constexpr I4& operator=(const I4&) = default;
  template <class T>
  constexpr I4(T v) : data(v) {}
  template <class T>
  constexpr operator T() const {
    return static_cast<T>(data);
  }
  friend I4& operator++(I4& lhs) {
    ++lhs.data;
    return lhs;
  }
  friend I4& operator--(I4& lhs) {
    --lhs.data;
    return lhs;
  }
  friend I4 operator++(I4& lhs, int) {
    I4 ret = lhs;
    ++lhs.data;
    return ret;
  }
  friend I4 operator--(I4& lhs, int) {
    I4 ret = lhs;
    --lhs.data;
    return ret;
  }
  friend I4& operator+=(I4& lhs, I4 rhs) {
    lhs.data += rhs.data;
    return lhs;
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend I4& operator+=(I4& lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    lhs.data += static_cast<C>(rhs);
    return lhs;
  }
  friend I4& operator-=(I4& lhs, I4 rhs) {
    lhs.data -= rhs.data;
    return lhs;
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend I4& operator-=(I4& lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    lhs.data -= static_cast<C>(rhs);
    return lhs;
  }
  friend I4& operator*=(I4& lhs, I4 rhs) {
    lhs.data *= rhs.data;
    return lhs;
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend I4& operator*=(I4& lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    lhs.data *= static_cast<C>(rhs);
    return lhs;
  }
  friend I4& operator/=(I4& lhs, I4 rhs) {
    lhs.data /= rhs.data;
    return lhs;
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend I4& operator/=(I4& lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    lhs.data /= static_cast<C>(rhs);
    return lhs;
  }
  friend I4& operator%=(I4& lhs, I4 rhs) {
    lhs.data %= rhs.data;
    return lhs;
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend I4& operator%=(I4& lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    lhs.data %= static_cast<C>(rhs);
    return lhs;
  }
  friend I4& operator&=(I4& lhs, I4 rhs) {
    lhs.data &= rhs.data;
    return lhs;
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend I4& operator&=(I4& lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    lhs.data &= static_cast<C>(rhs);
    return lhs;
  }
  friend I4& operator|=(I4& lhs, I4 rhs) {
    lhs.data |= rhs.data;
    return lhs;
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend I4& operator|=(I4& lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    lhs.data |= static_cast<C>(rhs);
    return lhs;
  }
  friend I4& operator^=(I4& lhs, I4 rhs) {
    lhs.data ^= rhs.data;
    return lhs;
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend I4& operator^=(I4& lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    lhs.data ^= static_cast<C>(rhs);
    return lhs;
  }
  friend I4& operator<<=(I4& lhs, I4 rhs) {
    lhs.data <<= rhs.data;
    return lhs;
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend I4& operator<<=(I4& lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    lhs.data <<= static_cast<C>(rhs);
    return lhs;
  }
  friend I4& operator>>=(I4& lhs, I4 rhs) {
    lhs.data >>= rhs.data;
    return lhs;
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend I4& operator>>=(I4& lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    lhs.data >>= static_cast<C>(rhs);
    return lhs;
  }
  friend auto operator+(I4 lhs) { return +lhs.data; }
  friend auto operator-(I4 lhs) { return -lhs.data; }
  friend auto operator+(I4 lhs, I4 rhs) { return lhs.data + rhs.data; }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend auto operator+(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data + static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend auto operator+(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) + rhs.data;
  }
  friend auto operator-(I4 lhs, I4 rhs) { return lhs.data - rhs.data; }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend auto operator-(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data - static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend auto operator-(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) - rhs.data;
  }
  friend auto operator*(I4 lhs, I4 rhs) { return lhs.data * rhs.data; }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend auto operator*(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data * static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend auto operator*(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) * rhs.data;
  }
  friend auto operator/(I4 lhs, I4 rhs) { return lhs.data / rhs.data; }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend auto operator/(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data / static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend auto operator/(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) / rhs.data;
  }
  friend auto operator%(I4 lhs, I4 rhs) { return lhs.data % rhs.data; }
  template <class T, class = std::enable_if_t<std::is_integral_v<T>>>
  friend auto operator%(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data % static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_integral_v<T>>>
  friend auto operator%(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) % rhs.data;
  }
  friend auto operator~(I4 lhs) { return ~lhs.data; }
  friend auto operator&(I4 lhs, I4 rhs) { return lhs.data & rhs.data; }
  template <class T, class = std::enable_if_t<std::is_integral_v<T>>>
  friend auto operator&(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data & static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_integral_v<T>>>
  friend auto operator&(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) & rhs.data;
  }
  friend auto operator|(I4 lhs, I4 rhs) { return lhs.data | rhs.data; }
  template <class T, class = std::enable_if_t<std::is_integral_v<T>>>
  friend auto operator|(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data | static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_integral_v<T>>>
  friend auto operator|(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) | rhs.data;
  }
  friend auto operator^(I4 lhs, I4 rhs) { return lhs.data ^ rhs.data; }
  template <class T, class = std::enable_if_t<std::is_integral_v<T>>>
  friend auto operator^(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data ^ static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_integral_v<T>>>
  friend auto operator^(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) ^ rhs.data;
  }
  friend auto operator<<(I4 lhs, I4 rhs) { return lhs.data << rhs.data; }
  template <class T, class = std::enable_if_t<std::is_integral_v<T>>>
  friend auto operator<<(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data << static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_integral_v<T>>>
  friend auto operator<<(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) << rhs.data;
  }
  friend auto operator>>(I4 lhs, I4 rhs) { return lhs.data >> rhs.data; }
  template <class T, class = std::enable_if_t<std::is_integral_v<T>>>
  friend auto operator>>(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data >> static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_integral_v<T>>>
  friend auto operator>>(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) >> rhs.data;
  }
  friend bool operator!(I4 v) { return !v.data; }
  friend auto operator&&(I4 lhs, I4 rhs) { return lhs.data && rhs.data; }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend auto operator&&(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data && static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend auto operator&&(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) && rhs.data;
  }
  friend auto operator||(I4 lhs, I4 rhs) { return lhs.data || rhs.data; }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend auto operator||(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data || static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend auto operator||(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) || rhs.data;
  }
  friend bool operator==(I4 lhs, I4 rhs) { return lhs.data == rhs.data; }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend bool operator==(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data == static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend bool operator==(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) == rhs.data;
  }
  friend bool operator!=(I4 lhs, I4 rhs) { return lhs.data != rhs.data; }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend bool operator!=(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data != static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend bool operator!=(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) != rhs.data;
  }
  friend bool operator<(I4 lhs, I4 rhs) { return lhs.data < rhs.data; }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend bool operator<(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data < static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend bool operator<(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) < rhs.data;
  }
  friend bool operator>(I4 lhs, I4 rhs) { return lhs.data > rhs.data; }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend bool operator>(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data > static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend bool operator>(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) > rhs.data;
  }
  friend bool operator<=(I4 lhs, I4 rhs) { return lhs.data <= rhs.data; }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend bool operator<=(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data <= static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend bool operator<=(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) <= rhs.data;
  }
  friend bool operator>=(I4 lhs, I4 rhs) { return lhs.data >= rhs.data; }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend bool operator>=(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data >= static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend bool operator>=(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) >= rhs.data;
  }
  friend std::ostream& operator<<(std::ostream& os, I4 v) { return os << +v; }
};
}  
namespace std {
template <>
struct numeric_limits<shlo_ref::I4> : std::numeric_limits<int8_t> {
  static constexpr shlo_ref::I4 min() noexcept { return shlo_ref::I4(-8); }
  static constexpr shlo_ref::I4 lowest() noexcept { return min(); }
  static constexpr shlo_ref::I4 max() noexcept { return shlo_ref::I4(7); }
};
}  
#endif  