#ifndef TENSORFLOW_TSL_PLATFORM_DEFAULT_INTEGRAL_TYPES_H_
#define TENSORFLOW_TSL_PLATFORM_DEFAULT_INTEGRAL_TYPES_H_
#include <cstdint>
namespace tsl {
typedef signed char int8;
typedef short int16;
typedef int int32;
typedef ::std::int64_t int64;
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef std::uint64_t uint64;
}  
#endif  