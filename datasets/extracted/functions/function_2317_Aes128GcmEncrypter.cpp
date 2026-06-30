#include "quiche/quic/core/crypto/aes_128_gcm_encrypter.h"
#include "openssl/evp.h"
namespace quic {
namespace {
const size_t kKeySize = 16;
const size_t kNonceSize = 12;
}  
Aes128GcmEncrypter::Aes128GcmEncrypter()
    : AesBaseEncrypter(EVP_aead_aes_128_gcm, kKeySize, kAuthTagSize, kNonceSize,
                        true) {
  static_assert(kKeySize <= kMaxKeySize, "key size too big");
  static_assert(kNonceSize <= kMaxNonceSize, "nonce size too big");
}
Aes128GcmEncrypter::~Aes128GcmEncrypter() {}
}  