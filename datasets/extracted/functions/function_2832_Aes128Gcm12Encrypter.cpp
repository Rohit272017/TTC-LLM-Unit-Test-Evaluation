#include "quiche/quic/core/crypto/aes_128_gcm_12_encrypter.h"
#include "openssl/evp.h"
namespace quic {
namespace {
const size_t kKeySize = 16;
const size_t kNonceSize = 12;
}  
Aes128Gcm12Encrypter::Aes128Gcm12Encrypter()
    : AesBaseEncrypter(EVP_aead_aes_128_gcm, kKeySize, kAuthTagSize, kNonceSize,
                        false) {
  static_assert(kKeySize <= kMaxKeySize, "key size too big");
  static_assert(kNonceSize <= kMaxNonceSize, "nonce size too big");
}
Aes128Gcm12Encrypter::~Aes128Gcm12Encrypter() {}
}  