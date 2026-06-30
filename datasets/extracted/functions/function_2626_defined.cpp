#ifndef QUICHE_COMMON_QUICHE_ENDIAN_H_
#define QUICHE_COMMON_QUICHE_ENDIAN_H_
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include "quiche/common/platform/api/quiche_export.h"
namespace quiche {
enum Endianness {
  NETWORK_BYTE_ORDER,  
  HOST_BYTE_ORDER      
};
class QUICHE_EXPORT QuicheEndian {
 public:
#if defined(__clang__) || \
    (defined(__GNUC__) && \
     ((__GNUC__ == 4 && __GNUC_MINOR__ >= 8) || __GNUC__ >= 5))
  static uint16_t HostToNet16(uint16_t x) { return __builtin_bswap16(x); }
  static uint32_t HostToNet32(uint32_t x) { return __builtin_bswap32(x); }
  static uint64_t HostToNet64(uint64_t x) { return __builtin_bswap64(x); }
#else
  static uint16_t HostToNet16(uint16_t x) { return PortableByteSwap(x); }
  static uint32_t HostToNet32(uint32_t x) { return PortableByteSwap(x); }
  static uint64_t HostToNet64(uint64_t x) { return PortableByteSwap(x); }
#endif
  static uint16_t NetToHost16(uint16_t x) { return HostToNet16(x); }
  static uint32_t NetToHost32(uint32_t x) { return HostToNet32(x); }
  static uint64_t NetToHost64(uint64_t x) { return HostToNet64(x); }
  template <typename T>
  static T PortableByteSwap(T input) {
    static_assert(std::is_unsigned<T>::value, "T has to be uintNN_t");
    union {
      T number;
      char bytes[sizeof(T)];
    } value;
    value.number = input;
    std::reverse(&value.bytes[0], &value.bytes[sizeof(T)]);
    return value.number;
  }
};
enum QuicheVariableLengthIntegerLength : uint8_t {
  VARIABLE_LENGTH_INTEGER_LENGTH_0 = 0,
  VARIABLE_LENGTH_INTEGER_LENGTH_1 = 1,
  VARIABLE_LENGTH_INTEGER_LENGTH_2 = 2,
  VARIABLE_LENGTH_INTEGER_LENGTH_4 = 4,
  VARIABLE_LENGTH_INTEGER_LENGTH_8 = 8,
  kQuicheDefaultLongHeaderLengthLength = VARIABLE_LENGTH_INTEGER_LENGTH_2,
};
}  
#endif  