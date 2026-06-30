#include "quiche/quic/core/crypto/aes_256_gcm_decrypter.h"
#include "openssl/aead.h"
#include "openssl/tls1.h"
#include "quiche/quic/platform/api/quic_flag_utils.h"
#include "quiche/quic/platform/api/quic_flags.h"
namespace quic {
namespace {
const size_t kKeySize = 32;
const size_t kNonceSize = 12;
}  
Aes256GcmDecrypter::Aes256GcmDecrypter()
    : AesBaseDecrypter(EVP_aead_aes_256_gcm, kKeySize, kAuthTagSize, kNonceSize,
                        true) {
  static_assert(kKeySize <= kMaxKeySize, "key size too big");
  static_assert(kNonceSize <= kMaxNonceSize, "nonce size too big");
}
Aes256GcmDecrypter::~Aes256GcmDecrypter() {}
uint32_t Aes256GcmDecrypter::cipher_id() const {
  return TLS1_CK_AES_256_GCM_SHA384;
}
}  