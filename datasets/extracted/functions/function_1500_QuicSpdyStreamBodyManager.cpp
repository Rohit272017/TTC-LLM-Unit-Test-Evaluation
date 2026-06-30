#include "quiche/quic/core/http/quic_spdy_stream_body_manager.h"
#include <algorithm>
#include "absl/strings/string_view.h"
#include "quiche/quic/platform/api/quic_logging.h"
namespace quic {
QuicSpdyStreamBodyManager::QuicSpdyStreamBodyManager()
    : total_body_bytes_received_(0) {}
size_t QuicSpdyStreamBodyManager::OnNonBody(QuicByteCount length) {
  QUICHE_DCHECK_NE(0u, length);
  if (fragments_.empty()) {
    return length;
  }
  fragments_.back().trailing_non_body_byte_count += length;
  return 0;
}
void QuicSpdyStreamBodyManager::OnBody(absl::string_view body) {
  QUICHE_DCHECK(!body.empty());
  fragments_.push_back({body, 0});
  total_body_bytes_received_ += body.length();
}
size_t QuicSpdyStreamBodyManager::OnBodyConsumed(size_t num_bytes) {
  QuicByteCount bytes_to_consume = 0;
  size_t remaining_bytes = num_bytes;
  while (remaining_bytes > 0) {
    if (fragments_.empty()) {
      QUIC_BUG(quic_bug_10394_1) << "Not enough available body to consume.";
      return 0;
    }
    Fragment& fragment = fragments_.front();
    const absl::string_view body = fragment.body;
    if (body.length() > remaining_bytes) {
      bytes_to_consume += remaining_bytes;
      fragment.body = body.substr(remaining_bytes);
      return bytes_to_consume;
    }
    remaining_bytes -= body.length();
    bytes_to_consume += body.length() + fragment.trailing_non_body_byte_count;
    fragments_.pop_front();
  }
  return bytes_to_consume;
}
int QuicSpdyStreamBodyManager::PeekBody(iovec* iov, size_t iov_len) const {
  QUICHE_DCHECK(iov);
  QUICHE_DCHECK_GT(iov_len, 0u);
  if (fragments_.empty()) {
    iov[0].iov_base = nullptr;
    iov[0].iov_len = 0;
    return 0;
  }
  size_t iov_filled = 0;
  while (iov_filled < fragments_.size() && iov_filled < iov_len) {
    absl::string_view body = fragments_[iov_filled].body;
    iov[iov_filled].iov_base = const_cast<char*>(body.data());
    iov[iov_filled].iov_len = body.size();
    iov_filled++;
  }
  return iov_filled;
}
size_t QuicSpdyStreamBodyManager::ReadableBytes() const {
  size_t count = 0;
  for (auto const& fragment : fragments_) {
    count += fragment.body.length();
  }
  return count;
}
size_t QuicSpdyStreamBodyManager::ReadBody(const struct iovec* iov,
                                           size_t iov_len,
                                           size_t* total_bytes_read) {
  *total_bytes_read = 0;
  QuicByteCount bytes_to_consume = 0;
  size_t index = 0;
  char* dest = reinterpret_cast<char*>(iov[index].iov_base);
  size_t dest_remaining = iov[index].iov_len;
  while (!fragments_.empty()) {
    Fragment& fragment = fragments_.front();
    const absl::string_view body = fragment.body;
    const size_t bytes_to_copy =
        std::min<size_t>(body.length(), dest_remaining);
    if (bytes_to_copy > 0) {
      memcpy(dest, body.data(), bytes_to_copy);
    }
    bytes_to_consume += bytes_to_copy;
    *total_bytes_read += bytes_to_copy;
    if (bytes_to_copy == body.length()) {
      bytes_to_consume += fragment.trailing_non_body_byte_count;
      fragments_.pop_front();
    } else {
      fragment.body = body.substr(bytes_to_copy);
    }
    if (bytes_to_copy == dest_remaining) {
      ++index;
      if (index == iov_len) {
        break;
      }
      dest = reinterpret_cast<char*>(iov[index].iov_base);
      dest_remaining = iov[index].iov_len;
    } else {
      dest += bytes_to_copy;
      dest_remaining -= bytes_to_copy;
    }
  }
  return bytes_to_consume;
}
}  