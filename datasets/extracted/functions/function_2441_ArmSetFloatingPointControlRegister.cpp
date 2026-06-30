#include "tsl/platform/denormal.h"
#include <cstdint>
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/platform.h"
#if !defined(__SSE3__) && !defined(__clang__) && \
    (defined(__GNUC__) && (__GNUC__ < 4) ||      \
     ((__GNUC__ == 4) && (__GNUC_MINOR__ < 9)))
#define GCC_WITHOUT_INTRINSICS
#endif
#if defined(PLATFORM_IS_X86) && !defined(IS_MOBILE_PLATFORM) && \
    !defined(GCC_WITHOUT_INTRINSICS)
#define X86_DENORM_USE_INTRINSICS
#endif
#ifdef X86_DENORM_USE_INTRINSICS
#include <pmmintrin.h>
#endif
#if defined(PLATFORM_IS_ARM) && defined(__ARM_FP) && (__ARM_FP > 0)
#define ARM_DENORM_AVAILABLE
#define ARM_FPCR_FZ (1 << 24)
#endif
namespace tsl {
namespace port {
bool DenormalState::operator==(const DenormalState& other) const {
  return flush_to_zero() == other.flush_to_zero() &&
         denormals_are_zero() == other.denormals_are_zero();
}
bool DenormalState::operator!=(const DenormalState& other) const {
  return !(this->operator==(other));
}
#ifdef ARM_DENORM_AVAILABLE
static inline void ArmSetFloatingPointControlRegister(uint32_t fpcr) {
#ifdef PLATFORM_IS_ARM64
  __asm__ __volatile__("msr fpcr, %[fpcr]"
                       :
                       : [fpcr] "r"(static_cast<uint64_t>(fpcr)));
#else
  __asm__ __volatile__("vmsr fpscr, %[fpcr]" : : [fpcr] "r"(fpcr));
#endif
}
static inline uint32_t ArmGetFloatingPointControlRegister() {
  uint32_t fpcr;
#ifdef PLATFORM_IS_ARM64
  uint64_t fpcr64;
  __asm__ __volatile__("mrs %[fpcr], fpcr" : [fpcr] "=r"(fpcr64));
  fpcr = static_cast<uint32_t>(fpcr64);
#else
  __asm__ __volatile__("vmrs %[fpcr], fpscr" : [fpcr] "=r"(fpcr));
#endif
  return fpcr;
}
#endif  
bool SetDenormalState(const DenormalState& state) {
#ifdef X86_DENORM_USE_INTRINSICS
  if (TestCPUFeature(SSE3)) {
    _MM_SET_FLUSH_ZERO_MODE(state.flush_to_zero() ? _MM_FLUSH_ZERO_ON
                                                  : _MM_FLUSH_ZERO_OFF);
    _MM_SET_DENORMALS_ZERO_MODE(state.denormals_are_zero()
                                    ? _MM_DENORMALS_ZERO_ON
                                    : _MM_DENORMALS_ZERO_OFF);
    return true;
  }
#endif
#ifdef ARM_DENORM_AVAILABLE
  if (state.flush_to_zero() == state.denormals_are_zero()) {
    uint32_t fpcr = ArmGetFloatingPointControlRegister();
    if (state.flush_to_zero()) {
      fpcr |= ARM_FPCR_FZ;
    } else {
      fpcr &= ~ARM_FPCR_FZ;
    }
    ArmSetFloatingPointControlRegister(fpcr);
    return true;
  }
#endif
  return false;
}
DenormalState GetDenormalState() {
#ifdef X86_DENORM_USE_INTRINSICS
  if (TestCPUFeature(SSE3)) {
    bool flush_zero_mode = _MM_GET_FLUSH_ZERO_MODE() == _MM_FLUSH_ZERO_ON;
    bool denormals_zero_mode =
        _MM_GET_DENORMALS_ZERO_MODE() == _MM_DENORMALS_ZERO_ON;
    return DenormalState(flush_zero_mode, denormals_zero_mode);
  }
#endif
#ifdef ARM_DENORM_AVAILABLE
  uint32_t fpcr = ArmGetFloatingPointControlRegister();
  if ((fpcr & ARM_FPCR_FZ) != 0) {
    return DenormalState(true, true);
  }
#endif
  return DenormalState(false, false);
}
ScopedRestoreFlushDenormalState::ScopedRestoreFlushDenormalState()
    : denormal_state_(GetDenormalState()) {}
ScopedRestoreFlushDenormalState::~ScopedRestoreFlushDenormalState() {
  SetDenormalState(denormal_state_);
}
ScopedFlushDenormal::ScopedFlushDenormal() {
  SetDenormalState(
      DenormalState(true, true));
}
ScopedDontFlushDenormal::ScopedDontFlushDenormal() {
  SetDenormalState(
      DenormalState(false, false));
}
}  
}  