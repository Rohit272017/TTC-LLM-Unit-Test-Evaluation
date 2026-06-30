#include "quiche/quic/core/congestion_control/hybrid_slow_start.h"
#include <algorithm>
#include "quiche/quic/platform/api/quic_logging.h"
namespace quic {
const int64_t kHybridStartLowWindow = 16;
const uint32_t kHybridStartMinSamples = 8;
const int kHybridStartDelayFactorExp = 3;  
const int64_t kHybridStartDelayMinThresholdUs = 4000;
const int64_t kHybridStartDelayMaxThresholdUs = 16000;
HybridSlowStart::HybridSlowStart()
    : started_(false),
      hystart_found_(NOT_FOUND),
      rtt_sample_count_(0),
      current_min_rtt_(QuicTime::Delta::Zero()) {}
void HybridSlowStart::OnPacketAcked(QuicPacketNumber acked_packet_number) {
  if (IsEndOfRound(acked_packet_number)) {
    started_ = false;
  }
}
void HybridSlowStart::OnPacketSent(QuicPacketNumber packet_number) {
  last_sent_packet_number_ = packet_number;
}
void HybridSlowStart::Restart() {
  started_ = false;
  hystart_found_ = NOT_FOUND;
}
void HybridSlowStart::StartReceiveRound(QuicPacketNumber last_sent) {
  QUIC_DVLOG(1) << "Reset hybrid slow start @" << last_sent;
  end_packet_number_ = last_sent;
  current_min_rtt_ = QuicTime::Delta::Zero();
  rtt_sample_count_ = 0;
  started_ = true;
}
bool HybridSlowStart::IsEndOfRound(QuicPacketNumber ack) const {
  return !end_packet_number_.IsInitialized() || end_packet_number_ <= ack;
}
bool HybridSlowStart::ShouldExitSlowStart(QuicTime::Delta latest_rtt,
                                          QuicTime::Delta min_rtt,
                                          QuicPacketCount congestion_window) {
  if (!started_) {
    StartReceiveRound(last_sent_packet_number_);
  }
  if (hystart_found_ != NOT_FOUND) {
    return true;
  }
  rtt_sample_count_++;
  if (rtt_sample_count_ <= kHybridStartMinSamples) {
    if (current_min_rtt_.IsZero() || current_min_rtt_ > latest_rtt) {
      current_min_rtt_ = latest_rtt;
    }
  }
  if (rtt_sample_count_ == kHybridStartMinSamples) {
    int64_t min_rtt_increase_threshold_us =
        min_rtt.ToMicroseconds() >> kHybridStartDelayFactorExp;
    min_rtt_increase_threshold_us = std::min(min_rtt_increase_threshold_us,
                                             kHybridStartDelayMaxThresholdUs);
    QuicTime::Delta min_rtt_increase_threshold =
        QuicTime::Delta::FromMicroseconds(std::max(
            min_rtt_increase_threshold_us, kHybridStartDelayMinThresholdUs));
    if (current_min_rtt_ > min_rtt + min_rtt_increase_threshold) {
      hystart_found_ = DELAY;
    }
  }
  return congestion_window >= kHybridStartLowWindow &&
         hystart_found_ != NOT_FOUND;
}
}  