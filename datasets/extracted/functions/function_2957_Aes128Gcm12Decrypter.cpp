#include "quiche/quic/core/crypto/aes_128_gcm_12_decrypter.h"
#include "openssl/aead.h"
#include "openssl/tls1.h"
namespace quic {
namespace {
const size_t kKeySize = 16;
const size_t kNonceSize = 12;
}  
Aes128Gcm12Decrypter::Aes128Gcm12Decrypter()
    : AesBaseDecrypter(EVP_aead_aes_128_gcm, kKeySize, kAuthTagSize, kNonceSize,
                        false) {
  static_assert(kKeySize <= kMaxKeySize, "key size too big");
  static_assert(kNonceSize <= kMaxNonceSize, "nonce size too big");
}
Aes128Gcm12Decrypter::~Aes128Gcm12Decrypter() {}
uint32_t Aes128Gcm12Decrypter::cipher_id() const {
  return TLS1_CK_AES_128_GCM_SHA256;
}
}  