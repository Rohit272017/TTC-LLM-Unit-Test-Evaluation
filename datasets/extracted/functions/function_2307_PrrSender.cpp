#include "quiche/quic/core/congestion_control/prr_sender.h"
#include "quiche/quic/core/quic_packets.h"
namespace quic {
PrrSender::PrrSender()
    : bytes_sent_since_loss_(0),
      bytes_delivered_since_loss_(0),
      ack_count_since_loss_(0),
      bytes_in_flight_before_loss_(0) {}
void PrrSender::OnPacketSent(QuicByteCount sent_bytes) {
  bytes_sent_since_loss_ += sent_bytes;
}
void PrrSender::OnPacketLost(QuicByteCount prior_in_flight) {
  bytes_sent_since_loss_ = 0;
  bytes_in_flight_before_loss_ = prior_in_flight;
  bytes_delivered_since_loss_ = 0;
  ack_count_since_loss_ = 0;
}
void PrrSender::OnPacketAcked(QuicByteCount acked_bytes) {
  bytes_delivered_since_loss_ += acked_bytes;
  ++ack_count_since_loss_;
}
bool PrrSender::CanSend(QuicByteCount congestion_window,
                        QuicByteCount bytes_in_flight,
                        QuicByteCount slowstart_threshold) const {
  if (bytes_sent_since_loss_ == 0 || bytes_in_flight < kMaxSegmentSize) {
    return true;
  }
  if (congestion_window > bytes_in_flight) {
    if (bytes_delivered_since_loss_ + ack_count_since_loss_ * kMaxSegmentSize <=
        bytes_sent_since_loss_) {
      return false;
    }
    return true;
  }
  if (bytes_delivered_since_loss_ * slowstart_threshold >
      bytes_sent_since_loss_ * bytes_in_flight_before_loss_) {
    return true;
  }
  return false;
}
}  