#ifndef TENSORFLOW_TSL_PLATFORM_FINGERPRINT_H_
#define TENSORFLOW_TSL_PLATFORM_FINGERPRINT_H_
#include "tsl/platform/platform.h"
#include "tsl/platform/stringpiece.h"
#include "tsl/platform/types.h"
#if TSL_IS_IN_OSS
#define USE_OSS_FARMHASH
#endif  
#ifdef USE_OSS_FARMHASH
#include <farmhash.h>
#else
#include "util/hash/farmhash_fingerprint.h"
#endif
namespace tsl {
struct Fprint128 {
  uint64_t low64;
  uint64_t high64;
};
inline bool operator==(const Fprint128& lhs, const Fprint128& rhs) {
  return lhs.low64 == rhs.low64 && lhs.high64 == rhs.high64;
}
struct Fprint128Hasher {
  size_t operator()(const Fprint128& v) const {
    return static_cast<size_t>(v.low64);
  }
};
namespace internal {
inline uint64_t ShiftMix(const uint64_t val) { return val ^ (val >> 47); }
}  
inline uint64_t FingerprintCat64(const uint64_t fp1, const uint64_t fp2) {
  static const uint64_t kMul = 0xc6a4a7935bd1e995ULL;
  uint64_t result = fp1 ^ kMul;
  result ^= internal::ShiftMix(fp2 * kMul) * kMul;
  result *= kMul;
  result = internal::ShiftMix(result) * kMul;
  result = internal::ShiftMix(result);
  return result;
}
inline uint64_t Fingerprint64(const absl::string_view s) {
#ifdef USE_OSS_FARMHASH
  return ::util::Fingerprint64(s.data(), s.size());
#else
  return farmhash::Fingerprint64(s.data(), s.size());
#endif
}
inline uint32_t Fingerprint32(const absl::string_view s) {
#ifdef USE_OSS_FARMHASH
  return ::util::Fingerprint32(s.data(), s.size());
#else
  return farmhash::Fingerprint32(s.data(), s.size());
#endif
}
inline Fprint128 Fingerprint128(const absl::string_view s) {
#ifdef USE_OSS_FARMHASH
  const auto fingerprint = ::util::Fingerprint128(s.data(), s.size());
  return {::util::Uint128Low64(fingerprint),
          ::util::Uint128High64(fingerprint)};
#else
  const auto fingerprint = farmhash::Fingerprint128(s.data(), s.size());
  return {absl::Uint128Low64(fingerprint), absl::Uint128High64(fingerprint)};
#endif
}
inline Fprint128 FingerprintCat128(const Fprint128& a, const Fprint128& b) {
  return {FingerprintCat64(a.low64, b.low64),
          FingerprintCat64(a.high64, b.high64)};
}
inline Fprint128 FingerprintCat128(const Fprint128& a, const uint64_t b) {
  auto x = FingerprintCat64(a.low64, b);
  return {x, FingerprintCat64(a.high64, x)};
}
}  
#endif  