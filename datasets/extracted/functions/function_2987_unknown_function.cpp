#ifndef TENSORFLOW_CORE_PLATFORM_FINGERPRINT_H_
#define TENSORFLOW_CORE_PLATFORM_FINGERPRINT_H_
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/fingerprint.h"
namespace tensorflow {
using Fprint128 = tsl::Fprint128;
using Fprint128Hasher = tsl::Fprint128Hasher;
using tsl::Fingerprint128;
using tsl::Fingerprint32;
using tsl::Fingerprint64;
using tsl::FingerprintCat64;
}  
#endif  