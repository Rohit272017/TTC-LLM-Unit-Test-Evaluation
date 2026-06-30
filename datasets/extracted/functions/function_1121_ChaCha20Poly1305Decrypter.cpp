#include "quiche/quic/core/crypto/chacha20_poly1305_decrypter.h"
#include "openssl/aead.h"
#include "openssl/tls1.h"
namespace quic {
namespace {
const size_t kKeySize = 32;
const size_t kNonceSize = 12;
}  
ChaCha20Poly1305Decrypter::ChaCha20Poly1305Decrypter()
    : ChaChaBaseDecrypter(EVP_aead_chacha20_poly1305, kKeySize, kAuthTagSize,
                          kNonceSize,
                           false) {
  static_assert(kKeySize <= kMaxKeySize, "key size too big");
  static_assert(kNonceSize <= kMaxNonceSize, "nonce size too big");
}
ChaCha20Poly1305Decrypter::~ChaCha20Poly1305Decrypter() {}
uint32_t ChaCha20Poly1305Decrypter::cipher_id() const {
  return TLS1_CK_CHACHA20_POLY1305_SHA256;
}
QuicPacketCount ChaCha20Poly1305Decrypter::GetIntegrityLimit() const {
  static_assert(kMaxIncomingPacketSize < 16384,
                "This key limit requires limits on decryption payload sizes");
  return 68719476736U;
}
}  