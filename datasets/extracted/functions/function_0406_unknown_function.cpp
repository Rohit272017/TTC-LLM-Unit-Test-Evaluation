#ifndef THIRD_PARTY_CEL_CPP_BASE_INTERNAL_MESSAGE_WRAPPER_H_
#define THIRD_PARTY_CEL_CPP_BASE_INTERNAL_MESSAGE_WRAPPER_H_
#include <cstdint>
namespace cel::base_internal {
inline constexpr uintptr_t kMessageWrapperTagMask = 0b1;
inline constexpr uintptr_t kMessageWrapperPtrMask = ~kMessageWrapperTagMask;
inline constexpr int kMessageWrapperTagSize = 1;
inline constexpr uintptr_t kMessageWrapperTagTypeInfoValue = 0b0;
inline constexpr uintptr_t kMessageWrapperTagMessageValue = 0b1;
}  
#endif  