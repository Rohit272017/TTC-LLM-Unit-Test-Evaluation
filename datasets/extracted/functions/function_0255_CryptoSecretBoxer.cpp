#include "quiche/quic/core/crypto/crypto_secret_boxer.h"
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "absl/strings/string_view.h"
#include "openssl/aead.h"
#include "openssl/err.h"
#include "quiche/quic/core/crypto/quic_random.h"
#include "quiche/quic/platform/api/quic_logging.h"
#include "quiche/common/platform/api/quiche_mutex.h"
namespace quic {
static const size_t kSIVNonceSize = 12;
static const size_t kBoxKeySize = 32;
struct CryptoSecretBoxer::State {
  std::vector<bssl::UniquePtr<EVP_AEAD_CTX>> ctxs;
};
CryptoSecretBoxer::CryptoSecretBoxer() {}
CryptoSecretBoxer::~CryptoSecretBoxer() {}
size_t CryptoSecretBoxer::GetKeySize() { return kBoxKeySize; }
static const EVP_AEAD* (*const kAEAD)() = EVP_aead_aes_256_gcm_siv;
bool CryptoSecretBoxer::SetKeys(const std::vector<std::string>& keys) {
  if (keys.empty()) {
    QUIC_LOG(DFATAL) << "No keys supplied!";
    return false;
  }
  const EVP_AEAD* const aead = kAEAD();
  std::unique_ptr<State> new_state(new State);
  for (const std::string& key : keys) {
    QUICHE_DCHECK_EQ(kBoxKeySize, key.size());
    bssl::UniquePtr<EVP_AEAD_CTX> ctx(
        EVP_AEAD_CTX_new(aead, reinterpret_cast<const uint8_t*>(key.data()),
                         key.size(), EVP_AEAD_DEFAULT_TAG_LENGTH));
    if (!ctx) {
      ERR_clear_error();
      QUIC_LOG(DFATAL) << "EVP_AEAD_CTX_init failed";
      return false;
    }
    new_state->ctxs.push_back(std::move(ctx));
  }
  quiche::QuicheWriterMutexLock l(&lock_);
  state_ = std::move(new_state);
  return true;
}
std::string CryptoSecretBoxer::Box(QuicRandom* rand,
                                   absl::string_view plaintext) const {
  size_t out_len =
      kSIVNonceSize + plaintext.size() + EVP_AEAD_max_overhead(kAEAD());
  std::string ret;
  ret.resize(out_len);
  uint8_t* out = reinterpret_cast<uint8_t*>(const_cast<char*>(ret.data()));
  rand->RandBytes(out, kSIVNonceSize);
  const uint8_t* const nonce = out;
  out += kSIVNonceSize;
  out_len -= kSIVNonceSize;
  size_t bytes_written;
  {
    quiche::QuicheReaderMutexLock l(&lock_);
    if (!EVP_AEAD_CTX_seal(state_->ctxs[0].get(), out, &bytes_written, out_len,
                           nonce, kSIVNonceSize,
                           reinterpret_cast<const uint8_t*>(plaintext.data()),
                           plaintext.size(), nullptr, 0)) {
      ERR_clear_error();
      QUIC_LOG(DFATAL) << "EVP_AEAD_CTX_seal failed";
      return "";
    }
  }
  QUICHE_DCHECK_EQ(out_len, bytes_written);
  return ret;
}
bool CryptoSecretBoxer::Unbox(absl::string_view in_ciphertext,
                              std::string* out_storage,
                              absl::string_view* out) const {
  if (in_ciphertext.size() < kSIVNonceSize) {
    return false;
  }
  const uint8_t* const nonce =
      reinterpret_cast<const uint8_t*>(in_ciphertext.data());
  const uint8_t* const ciphertext = nonce + kSIVNonceSize;
  const size_t ciphertext_len = in_ciphertext.size() - kSIVNonceSize;
  out_storage->resize(ciphertext_len);
  bool ok = false;
  {
    quiche::QuicheReaderMutexLock l(&lock_);
    for (const bssl::UniquePtr<EVP_AEAD_CTX>& ctx : state_->ctxs) {
      size_t bytes_written;
      if (EVP_AEAD_CTX_open(ctx.get(),
                            reinterpret_cast<uint8_t*>(
                                const_cast<char*>(out_storage->data())),
                            &bytes_written, ciphertext_len, nonce,
                            kSIVNonceSize, ciphertext, ciphertext_len, nullptr,
                            0)) {
        ok = true;
        *out = absl::string_view(out_storage->data(), bytes_written);
        break;
      }
      ERR_clear_error();
    }
  }
  return ok;
}
}  