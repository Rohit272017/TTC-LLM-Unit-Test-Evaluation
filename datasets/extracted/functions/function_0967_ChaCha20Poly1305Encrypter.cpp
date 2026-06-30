#include "quiche/quic/core/crypto/chacha20_poly1305_encrypter.h"
#include <limits>
#include "openssl/evp.h"
namespace quic {
namespace {
const size_t kKeySize = 32;
const size_t kNonceSize = 12;
}  
ChaCha20Poly1305Encrypter::ChaCha20Poly1305Encrypter()
    : ChaChaBaseEncrypter(EVP_aead_chacha20_poly1305, kKeySize, kAuthTagSize,
                          kNonceSize,
                           false) {
  static_assert(kKeySize <= kMaxKeySize, "key size too big");
  static_assert(kNonceSize <= kMaxNonceSize, "nonce size too big");
}
ChaCha20Poly1305Encrypter::~ChaCha20Poly1305Encrypter() {}
QuicPacketCount ChaCha20Poly1305Encrypter::GetConfidentialityLimit() const {
  return std::numeric_limits<QuicPacketCount>::max();
}
}  